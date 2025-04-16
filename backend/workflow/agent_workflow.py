import logging
import json 
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from openai import AsyncOpenAI
from backend.agents.query_planner import QueryPlanningAgent
from backend.agents.retriever import RetrievalAgent
from backend.agents.response_generator import ResponseGenerationAgent
from backend.data.vectordb_qdrant import VectorDBManager
from backend.workflow.state import ChatState

logger = logging.getLogger(__name__)


class AgentWorkflow:
    """
    Coordinates the flow between different agents for e-commerce queries. (History Removed)
    Initializes required clients and agents.
    """

    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.vector_db_manager: VectorDBManager = None
        self.query_planner: QueryPlanningAgent = None
        self.retriever: RetrievalAgent = None
        self.response_generator: ResponseGenerationAgent = None
        self.workflow = None

        self._initialize_components()

        if self.query_planner and self.retriever and self.response_generator:
            self.workflow = self._create_workflow()
            logger.info("Agent workflow created successfully.")
        else:
             logger.error("Workflow creation skipped due to failed component initialization.")

    def _initialize_components(self):
        """Initializes VectorDBManager and Agents."""
        try:
            logger.info("Initializing VectorDBManager (Qdrant)...")
            self.vector_db_manager = VectorDBManager(openai_client=self.openai_client)

            logger.info("Initializing QueryPlanningAgent...")
            self.query_planner = QueryPlanningAgent(openai_client=self.openai_client)

            logger.info("Initializing RetrievalAgent...")
            # Pass dependencies needed by RetrievalAgent (excluding history)
            self.retriever = RetrievalAgent(
                vector_db_manager=self.vector_db_manager,
                openai_client=self.openai_client # Keep if needed for other BaseAgent methods
            )

            logger.info("Initializing ResponseGenerationAgent...")
            self.response_generator = ResponseGenerationAgent(openai_client=self.openai_client)

            logger.info("All workflow components initialized.")

        except Exception as e:
            logger.exception(f"Failed to initialize workflow components: {e}")
            self.vector_db_manager = None
            self.query_planner = None
            self.retriever = None
            self.response_generator = None

    def _create_workflow(self):
        """Create and compile the Langgraph workflow (History Removed)"""
        if not all([self.query_planner, self.retriever, self.response_generator]):
             logger.error("Cannot create workflow: One or more agents failed to initialize.")
             return None

        # Define workflow steps (node functions)
        async def plan_query(state: ChatState) -> Dict[str, Any]:
            logger.info("--- Workflow Step: Plan Query ---")
            if not state["messages"] or not isinstance(state["messages"][0], HumanMessage):
                 logger.error("State is missing the initial user message.")
                 return {"error": "Initial user message missing", "query_plan": {}}

            latest_message = state["messages"][0]

            try:

                query_plan_result = await self.query_planner.run(
                    message=latest_message.content
                )

                if isinstance(query_plan_result, dict) and "error" in query_plan_result:
                    error_msg = query_plan_result['error']
                    logger.error(f"Query planner agent returned an error: {error_msg}")
                    return {"error": error_msg, "query_plan": {"error": error_msg, "is_in_scope": False, "request_type": "error"}}
                elif not isinstance(query_plan_result, dict): # Planner should return a dict (the plan)
                     logger.error(f"Query planner agent returned unexpected type: {type(query_plan_result)}. Expected dict.")
                     error_msg = "Planner returned invalid output type."
                     return {"error": error_msg, "query_plan": {"error": error_msg, "is_in_scope": False, "request_type": "error"}}
                else:
                    query_plan = query_plan_result
                    if "error" in query_plan:
                         logger.error(f"Query plan contains an error flag: {query_plan['error']}")
                         # Propagate the error within the plan structure
                         return {"error": query_plan['error'], "query_plan": query_plan}
                    else:
                         logger.debug(f"Successfully received query plan from agent: {json.dumps(query_plan, indent=2)}")
                         return {"query_plan": query_plan, "error": None}

            except Exception as e:
                logger.exception(f"Unexpected error in query planning node execution: {str(e)}")
                error_msg = f"Query planning node failed unexpectedly: {str(e)}"
                return {"error": error_msg, "query_plan": {"error": error_msg, "is_in_scope": False, "request_type": "error"}}


        async def retrieve_information(state: ChatState) -> Dict[str, Any]:
            logger.info("--- Workflow Step: Retrieve Information ---")
            if state.get("error"):
                logger.warning(f"Skipping retrieval due to previous error: {state['error']}")
                return {"product_info": {"products": [], "result_count": 0, "search_successful": False, "error": state['error']}}

            query_plan = state.get("query_plan", {})
            if not query_plan or query_plan.get("error") or not query_plan.get("is_in_scope", False):
                 error_reason = query_plan.get("error", "Query out of scope or planning failed.")
                 logger.info(f"Skipping retrieval: {error_reason}")
                 return {
                     "product_info": {
                         "products": [], "result_count": 0, "search_successful": False,
                         "error": error_reason
                     }
                 }

            if not state["messages"] or not isinstance(state["messages"][0], HumanMessage):
                 logger.error("State is missing the user message for retrieval context.")
                 return {"product_info": {"products": [], "result_count": 0, "search_successful": False, "error": "User message missing"}}

            latest_message = state["messages"][0]

            try:

                product_info = await self.retriever.retrieve_information(
                    message=latest_message.content,
                    query_plan=state["query_plan"]
                )
                logger.debug(f"Retrieved product info: Found {product_info.get('result_count', 0)} items.")
                return {"product_info": product_info}
            except Exception as e:
                logger.exception(f"Error in retrieval stage: {str(e)}")
                error_msg = f"Information retrieval failed: {str(e)}"
                return {
                    "product_info": {
                        "products": [], "result_count": 0, "search_successful": False,
                        "error": error_msg
                    }
                }

        async def generate_response(state: ChatState) -> Dict[str, Any]:
            logger.info("--- Workflow Step: Generate Response ---")
            final_response = ""

            if not state["messages"] or not isinstance(state["messages"][0], HumanMessage):
                 logger.error("State is missing the user message for response generation.")
                 final_response = "Sorry, I lost the context of your message. Please try again."
                 new_messages = [AIMessage(content=final_response)] # State only has AI response now
                 return {
                     "messages": new_messages,
                     "current_response": final_response
                 }

            latest_message = state["messages"][0]

            if state.get("error"):
                error_message = state["error"]
                logger.error(f"Generating error response due to previous error: {error_message}")
                final_response = f"I'm sorry, but I encountered an error: {error_message}. Please try again."
            else:
                product_info = state.get("product_info", {})
                query_plan = state.get("query_plan", {})

                # Handle cases where retrieval failed or was skipped
                if product_info.get("error") or not product_info.get("search_successful", True):
                    if query_plan.get("request_type") == "out_of_scope" or query_plan.get("is_in_scope") is False:
                        final_response = "I specialize in Refrigerator and Dishwasher parts from PartSelect. Could you ask something related to those?"
                    elif not product_info.get("search_successful", True):
                        final_response = product_info.get("no_results_reason", "I couldn't find specific parts matching your request. Could you provide more details?")
                    else:
                        final_response = f"I encountered an issue retrieving information: {product_info.get('error', 'Unknown retrieval error')}. Please try again."
                    logger.warning(f"Generating response based on retrieval failure/scope issue: {final_response}")
                else:
                    try:

                        final_response = await self.response_generator.generate_response(
                            message=latest_message.content,
                            query_plan=state["query_plan"],
                            product_info=state["product_info"]
                        )
                        if not isinstance(final_response, str):
                             logger.error(f"Response generator returned non-string: {type(final_response)}. Using fallback.")
                             final_response = "Sorry, I encountered an internal error formatting the response."

                        logger.debug(f"Generated final response: {final_response[:100]}...")

                    except Exception as e:
                        logger.exception(f"Error in response generation stage: {str(e)}")
                        final_response = "I'm sorry, but I encountered an error while generating a response after finding information. Please try again."

            # --- Update message history and state ---
            new_messages = [latest_message, AIMessage(content=final_response)]
            return {
                "messages": new_messages,
                "current_response": final_response
            }

        # Create and compile the graph
        workflow_graph = StateGraph(ChatState)
        workflow_graph.add_node("plan_query", plan_query)
        workflow_graph.add_node("retrieve_information", retrieve_information)
        workflow_graph.add_node("generate_response", generate_response)

        workflow_graph.add_edge("plan_query", "retrieve_information")
        workflow_graph.add_edge("retrieve_information", "generate_response")
        workflow_graph.add_edge("generate_response", END)

        workflow_graph.set_entry_point("plan_query")

        return workflow_graph.compile()

    # Update process_message signature and initial state
    async def process_message(self, user_message: str) -> str:
        """Process a user message through the workflow (History Removed)"""
        if not self.workflow:
             logger.error("Workflow is not compiled or failed during initialization. Cannot process message.")
             return "I'm sorry, my internal systems are not ready. Please try again later."

        logger.info(f"--- Starting Workflow for Message: '{user_message}' ---")
        # Initial state only contains the current user message
        messages = [HumanMessage(content=user_message)]

        initial_state: ChatState = {
            "messages": messages, # Only the current user message
            "query_plan": {},
            "product_info": {},
            "current_response": "",
            "error": None
        }

        try:
            final_state = await self.workflow.ainvoke(initial_state)
            response = final_state.get("current_response", "An unexpected error occurred, and no response was generated.")
            logger.info(f"--- Workflow Finished. Final Response: '{response[:100]}...' ---")
            return response
        except Exception as e:
            logger.exception(f"Workflow invocation error: {str(e)}")
            return "I'm sorry, but I encountered an unexpected error processing your request. Please try again later."

    async def close(self):
        """Gracefully close resources, like the VectorDBManager client."""
        # ... (close method remains the same) ...
        if self.vector_db_manager:
            try:
                logger.info("Closing VectorDBManager (Qdrant client)...")
                await self.vector_db_manager.close()
                logger.info("VectorDBManager closed successfully.")
            except Exception as e:
                logger.error(f"Error closing VectorDBManager: {e}")
        else:
            logger.info("No VectorDBManager instance to close.")

