import logging
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from openai import AsyncOpenAI
from backend.agents.base import BaseAgent

logger = logging.getLogger(__name__)

class QueryPlanningAgent(BaseAgent):
    """
    Agent responsible for analyzing user queries related to PartSelect
    (Refrigerator & Dishwasher parts) and planning the retrieval strategy.
    Generates a validated query plan dictionary or an error dictionary.
    """

    PART_NUMBER_REGEX = re.compile(r'\b((?:PS|AP|WP|PD|EBX|W|WR|WD|DA|DE|DG|DD|EAP|TJ|WB|WH|WR|WS|WW)\d{4,}|[A-Z0-9]{6,})\b', re.IGNORECASE)
    MODEL_NUMBER_REGEX = re.compile(
        r'\b(?=.{8,10}\b)(?:(?!.*[-/].*[-/])[A-Z0-9]+(?:[-/][A-Z0-9]+)?)\b'
    )

    def __init__(self, openai_client: AsyncOpenAI):
        super().__init__(openai_client)

    def _get_system_prompt(self) -> str:
        return """
        You are a Query Planning Agent for the PartSelect e-commerce website, specializing in **Refrigerator** and **Dishwasher** parts. Your primary function is to analyze user requests (including conversation history) and generate a structured JSON plan to guide the product retrieval and information process.

        **Instructions:**
        1.  **Analyze:** Carefully read the latest user message and relevant conversation history to understand the user's intent. Focus on identifying specific parts, appliance models, symptoms, or requested actions (like compatibility checks or installation help).
        2.  **Domain Focus:** Determine if the request is related to Refrigerator or Dishwasher parts/models. If the request is clearly outside this scope (e.g., asking about lawnmowers, cars, general knowledge), set "is_in_scope" to false.
        3.  **Extract Key Information:** Identify and extract the following details if present:
            *   `part_number`: Specific part number mentioned (e.g., "PS11752778", "WP8544771").
            *   `model_number`: Specific appliance model number mentioned (e.g., "WDT780SAEM1", "WRF560SEYM05").
            *   `appliance_type`: The type of appliance ("Refrigerator" or "Dishwasher"). Infer if possible from context or model/part numbers.
            *   `brand`: The brand of the appliance (e.g., "Whirlpool", "GE", "Samsung").
            *   `symptom`: Description of the problem or symptom if the user is troubleshooting (e.g., "ice maker not working", "dishwasher not draining", "fridge too warm").
            *   `component`: The specific component mentioned if troubleshooting (e.g., "ice maker", "water valve", "heating element", "door gasket").
            *   `request_type`: Classify the user's primary goal:
                - 'search_part': Looking for a specific part or parts for a symptom/model.
                - 'compatibility_check': Asking if a part fits a specific model.
                - 'installation_guide': Asking how to install a specific part.
                - 'troubleshooting': Asking how to fix a problem/symptom.
                - 'product_details': Asking for more information about a specific part.
                - 'other': A request within scope but not fitting the above categories.
                - 'out_of_scope': The request is not about Refrigerator or Dishwasher parts.
        4.  **Format Output:** Generate **ONLY** a single JSON object containing the plan. Do not include any explanatory text before or after the JSON. The JSON object should have the following structure (include keys only if relevant information is found/inferred):

            ```json
            {
              "is_in_scope": true, // boolean: true if related to Fridge/Dishwasher parts, false otherwise
              "request_type": "...", // e.g., "compatibility_check", "search_part", "troubleshooting", "installation_guide", "out_of_scope"
              "search_query": "A concise query for semantic search, especially for troubleshooting. Example: 'Whirlpool refrigerator ice maker not working'",
              "part_number": "...", // Extracted part number string, if any
              "model_number": "...", // Extracted model number string, if any
              "appliance_type": "Refrigerator" | "Dishwasher" | null,
              "brand": "...", // Extracted brand string, if any
              "symptom": "...", // Description of the problem
              "component": "...", // Specific component mentioned
              "potential_parts": ["part_num1", "part_num2"], // Optional: If user mentions multiple parts
              "target_part_context": "...", // Optional: Part number identified from previous turn for context
              "target_model_context": "..." // Optional: Model number identified from previous turn for context
            }
            ```

        **Important Rules:**
        *   **JSON Only:** Your entire output MUST be a single, valid JSON object.
        *   **Prioritize IDs:** Give precedence to extracting `part_number` and `model_number` if present.
        *   **Scope:** If `is_in_scope` is false, set `request_type` to `out_of_scope` and minimize other fields.
        *   **Conciseness:** Keep `search_query` focused, especially for troubleshooting or general part searches. If a specific `part_number` is given, the `search_query` might be less critical or simply echo the part number/name.

        **Example User Queries & Outputs:**

        *   **Query:** "How can I install part number PS11752778?"
            **Output:**
            ```json
            {
              "is_in_scope": true,
              "request_type": "installation_guide",
              "part_number": "PS11752778",
              "search_query": "Installation guide for PS11752778"
            }
            ```
        *   **Query:** "Is part WP8544771 compatible with my WDT780SAEM1 model?"
            **Output:**
            ```json
            {
              "is_in_scope": true,
              "request_type": "compatibility_check",
              "part_number": "WP8544771",
              "model_number": "WDT780SAEM1",
              "search_query": "Check compatibility WP8544771 with WDT780SAEM1"
            }
            ```
        *   **Query:** "The ice maker on my Whirlpool fridge is not working. How can I fix it?"
            **Output:**
            ```json
            {
              "is_in_scope": true,
              "request_type": "troubleshooting",
              "appliance_type": "Refrigerator",
              "brand": "Whirlpool",
              "symptom": "ice maker not working",
              "component": "ice maker",
              "search_query": "Whirlpool refrigerator ice maker not working fix"
            }
            ```
        *   **Query:** "Do you sell tires?"
            **Output:**
            ```json
            {
              "is_in_scope": false,
              "request_type": "out_of_scope",
              "search_query": "User asking about tires"
            }
            ```
        """

    def _extract_with_regex(self, text: str) -> Dict[str, Optional[str]]:
        extracted = {}
        part_match = self.PART_NUMBER_REGEX.search(text)
        if part_match:
            extracted["part_number_regex"] = part_match.group(1)

        # Avoid matching part numbers as model numbers if possible
        remaining_text = self.PART_NUMBER_REGEX.sub("", text)
        model_match = self.MODEL_NUMBER_REGEX.search(remaining_text)
        if model_match:
             # Simple check to avoid overly short/numeric strings being models
             potential_model = model_match.group(1)
             if not potential_model.isdigit() or len(potential_model) > 5:
                 extracted["model_number_regex"] = potential_model

        return extracted

    def _parse_llm_response_to_plan(self, response_text: str, original_message: str) -> Dict[str, Any]:
        """
        Cleans, parses, and validates the LLM response string into a JSON plan.
        Returns the plan dictionary or an error dictionary.
        """
        query_plan = {}
        cleaned_text = response_text

        try:
            logger.debug(f"Raw LLM planner response: {response_text[:500]}")
            cleaned_text = re.sub(r'^```json\s*', '', response_text, flags=re.IGNORECASE)
            cleaned_text = re.sub(r'\s*```$', '', cleaned_text)
            cleaned_text = cleaned_text.strip()

            # Handle potential extra braces (simple fix)
            if cleaned_text.startswith('{{'):
                logger.warning("Detected double opening brace in planner output, attempting correction.")
                cleaned_text = cleaned_text[1:]

            query_plan = json.loads(cleaned_text)

            if not isinstance(query_plan, dict):
                raise ValueError("Parsed query plan is not a dictionary.")

            if "is_in_scope" not in query_plan:
                if any(k in query_plan for k in ["part_number", "model_number", "appliance_type", "symptom"]):
                    query_plan["is_in_scope"] = True
                    logger.warning("LLM plan missing 'is_in_scope', inferred as true.")
                else:
                    query_plan["is_in_scope"] = False
                    logger.warning("LLM plan missing 'is_in_scope', defaulted to false.")

            if "request_type" not in query_plan:
                 if query_plan.get("is_in_scope") == False:
                      query_plan["request_type"] = "out_of_scope"
                 else:
                      query_plan["request_type"] = "search_part"
                 logger.warning(f"LLM plan missing 'request_type', defaulted to '{query_plan['request_type']}'.")

            if "search_query" not in query_plan and query_plan.get("is_in_scope"):
                 if query_plan.get("part_number"):
                     query_plan["search_query"] = f"Details for part {query_plan['part_number']}"
                 elif query_plan.get("symptom"):
                      query_plan["search_query"] = query_plan["symptom"]
                 else:
                      query_plan["search_query"] = original_message
                 logger.warning(f"LLM plan missing 'search_query', generated fallback: '{query_plan['search_query']}'")


            logger.info("Successfully parsed and validated LLM response into query plan.")
            return query_plan

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response into JSON: {e}")
            logger.debug(f"Cleaned text causing parse error:\n{cleaned_text[:500]}")
            return {"error": "Failed to parse query plan from LLM response", "raw_response": response_text, "is_in_scope": False, "request_type": "error"}
        except ValueError as e:
             logger.error(f"Validation error after parsing LLM response: {e}")
             logger.debug(f"Parsed structure causing validation error:\n{query_plan}")
             return {"error": f"Query plan validation failed: {e}", "raw_response": response_text, "is_in_scope": False, "request_type": "error"}
        except Exception as e:
            logger.exception(f"Unexpected error parsing/validating LLM response: {e}")
            return {"error": f"Unexpected error processing plan: {e}", "raw_response": response_text, "is_in_scope": False, "request_type": "error"}


    async def run(self, **kwargs) -> Dict[str, Any]:
        """
        Generates, cleans, parses, and validates a query plan.

        Returns:
            A dictionary containing the validated query plan,
            or an error dictionary if generation/parsing/validation fails.
        """
        message = kwargs.get("message")
        logger.info(f"Planning query for message: '{message}'")

        if not message:
             logger.error("QueryPlanningAgent.run called without 'message'.")
             return {"error": "Input message missing", "is_in_scope": False, "request_type": "error"}

        try:
            raw_llm_output = await super().run(**kwargs)

            if not isinstance(raw_llm_output, str):
                 if isinstance(raw_llm_output, dict) and "error" in raw_llm_output:
                      logger.error(f"BaseAgent run returned an error: {raw_llm_output['error']}")
                      return raw_llm_output
                 else:
                      logger.error(f"LLM response was not a string: {type(raw_llm_output)}. Output: {str(raw_llm_output)[:200]}")
                      return {"error": "LLM returned unexpected output type", "is_in_scope": False, "request_type": "error"}

            if not raw_llm_output:
                 logger.error("LLM returned an empty response string for query planning.")
                 return {"error": "LLM returned empty response", "is_in_scope": False, "request_type": "error"}

            query_plan = self._parse_llm_response_to_plan(raw_llm_output, message)

            logger.debug(f"Generated Query Plan: {json.dumps(query_plan, indent=2)}")
            return query_plan

        except Exception as e:
            logger.exception(f"Error during query planning process for message '{message}': {str(e)}")
            return {
                "error": f"Query planning failed: {str(e)}",
                "is_in_scope": False,
                "request_type": "error"
            }

