import logging
import json
from typing import Dict, Any, List, Tuple, Optional
from agents.base import BaseAgent
from data.vectordb_qdrant import VectorDBManager
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class RetrievalAgent(BaseAgent):
    """
    Agent responsible for retrieving PartSelect product information (Refrigerator & Dishwasher parts)
    from the vector database based on a query plan.
    """

    def __init__(self, vector_db_manager: VectorDBManager, openai_client: AsyncOpenAI):
        # Pass vector_db_manager to the BaseAgent constructor
        super().__init__(openai_client=openai_client)
        self.vector_db_manager = vector_db_manager
        self.default_top_k = 10
        self.specific_part_top_k = 3

    def _get_system_prompt(self) -> str:
        return """
        You are a Retrieval Agent for the PartSelect e-commerce website, specializing in Refrigerator and Dishwasher parts. Your role is internal processing.
        1. Receive a query plan detailing the user's request (e.g., part number, model number, symptom, request type).
        2. Formulate search parameters (text query, filters) for the vector database based on the plan.
        3. Execute the search against the vector database.
        4. Process the raw database results:
           - Extract relevant product data (name, description, price, brand, specs, URLs).
           - Perform compatibility checks if requested in the plan.
           - Extract installation information if requested.
           - Format the results clearly.
        5. Return the structured results.

        Prioritize accuracy based on the provided plan. Handle specific request types like compatibility checks and installation guide requests appropriately.
        """

    def _prepare_search_query_and_filters(self, query_plan: Dict[str, Any], message: str) -> Tuple[str, Optional[Dict[str, Any]], int]:
        search_query = query_plan.get("search_query", message)
        filters = {}
        top_k = self.default_top_k

        # 1. Prioritize Part Number Filter (most specific)
        part_number = query_plan.get("part_number")
        if part_number:
            # Use exact match filter for part number
            filters["part_number"] = str(part_number).upper()
            search_query = f"Part {part_number}"
            top_k = self.specific_part_top_k
            logger.info(f"Prioritizing filter for specific part_number: {part_number}")

        # 2. Add other filters if no specific part number or for broader context
        else:
            appliance_type = query_plan.get("appliance_type")
            if appliance_type and appliance_type in ["Refrigerator", "Dishwasher"]:
                filters["appliance_type"] = appliance_type

            brand = query_plan.get("brand")
            if brand:
                filters["brand"] = brand 

            model_number = query_plan.get("model_number")
            if model_number and "model" not in search_query.lower():
                 search_query += f" for model {model_number}"

        # Add component/symptom to search query if relevant for troubleshooting
        if query_plan.get("request_type") == "troubleshooting":
            component = query_plan.get("component")
            symptom = query_plan.get("symptom")
            if component and component not in search_query.lower():
                search_query += f" {component}"
            if symptom and symptom not in search_query.lower():
                 search_query += f" {symptom}"


        # Fallback if search query is still empty
        if not search_query:
            search_query = message

        logger.info(f"Prepared search: query='{search_query}', filters={filters if filters else 'None'}, top_k={top_k}")
        return search_query, filters if filters else None, top_k


    def _perform_compatibility_check(self, product_data: Dict[str, Any], target_model: str) -> bool:
        if not target_model:
            return False


        compatible_models = product_data.get("compatibleModels")
        if isinstance(compatible_models, list):
            return target_model.upper() in [m.upper() for m in compatible_models]

        return False


    def _extract_installation_info(self, product_data: Dict[str, Any]) -> Optional[str]:
        """
        Extracts installation guide URL or relevant info from product data.
        Adapt based on your data structure.
        """
        video_tutorials = product_data.get("videoTutorials")
        if isinstance(video_tutorials, list) and video_tutorials:
            first_video = video_tutorials[0]
            if isinstance(first_video, dict) and first_video.get("videoUrl"):
                logger.debug(f"Extracted installation video URL: {first_video['videoUrl']}")
                return first_video["videoUrl"] # Return the URL of the first video

        # Fallback: Check installationGuides (though empty in sample)
        installation_guides = product_data.get("installationGuides")
        if isinstance(installation_guides, list) and installation_guides:
             # Assuming guides might be strings (URLs) or objects with a URL
             first_guide = installation_guides[0]
             if isinstance(first_guide, str):
                 logger.debug(f"Extracted installation guide URL (string): {first_guide}")
                 return first_guide
             elif isinstance(first_guide, dict) and first_guide.get("url"): # Check for a common 'url' key
                 logger.debug(f"Extracted installation guide URL (object): {first_guide['url']}")
                 return first_guide["url"]

        logger.debug("No specific installation guide URL found in videoTutorials or installationGuides.")
        return None


    def _process_search_results(
        self,
        raw_db_results: List[Dict[str, Any]],
        query_plan: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Processes raw search results, performs checks based on query_plan.
        """
        processed_results = []
        if not raw_db_results:
            return []

        request_type = query_plan.get("request_type")
        target_model = query_plan.get("model_number")
        target_part = query_plan.get("part_number") 

        logger.info(f"Processing {len(raw_db_results)} raw results for request_type='{request_type}'")

        for result in raw_db_results:
            product_data = result.get('document', {})
            metadata = result.get('metadata', {})
            product_id = result.get('id') 
            distance = result.get('distance')
            similarity_score = 1 / (1 + distance) if distance is not None and distance >= 0 else 0

            if not product_data:
                 logger.warning(f"Skipping result with ID {product_id} due to missing document data.")
                 continue
            if not metadata:
                 logger.warning(f"Result with ID {product_id} has missing metadata. Proceeding with caution.")

            # --- Compatibility Check (if requested) ---
            is_compatible = None
            if request_type == "compatibility_check" and target_model:
                is_compatible = self._perform_compatibility_check(product_data, target_model)
                if not is_compatible:
                     logger.debug(f"Filtering out part {product_id} - failed compatibility check for model {target_model}")
                     continue

            # --- Installation Info Extraction (if requested) ---
            installation_info = None
            if request_type == "installation_guide":
                installation_info = self._extract_installation_info(product_data)

            # --- Structure the final output ---
            processed_product = {
                "id": product_id,
                "part_number": metadata.get("part_number", product_data.get("partSelectNumber", product_id)),
                "manufacturer_part_number": metadata.get("manufacturer_part_number", product_data.get("manufacturerPartNumber")),
                "name": product_data.get('title', 'N/A'),
                "description": product_data.get('description', ''),
                "price": metadata.get('price', product_data.get('price')),
                "appliance_type": metadata.get("appliance_type"),
                "availability": metadata.get("availability", product_data.get('stockStatus', 'Inquire')),
                "url": product_data.get('sourceUrl'),
                "image_url": product_data.get('imageUrl'),
                "retrieval_score": similarity_score,
                "raw_distance": distance,
                "compatibility_result": is_compatible if is_compatible is not None else "N/A",
                "installation_guide": installation_info if installation_info is not None else "N/A",
                "video_tutorials": product_data.get("videoTutorials", []),
                "compatible_models_list": product_data.get("compatibleModels", []),
            }

            processed_results.append(processed_product)

        # Re-sort (might have been filtered) - sorting by score is good default
        processed_results.sort(key=lambda x: x['retrieval_score'], reverse=True)

        logger.info(f"Returning {len(processed_results)} processed results after filtering/checks.")
        return processed_results

    async def retrieve_information(self, message: str, query_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Retrieve product information based on the PartSelect-specific query plan.
        """
        logger.info(f"RetrievalAgent received message: '{message}'")
        logger.debug(f"Query Plan: {json.dumps(query_plan, indent=2)}")

        # --- Initial Plan Check ---
        if not query_plan or query_plan.get("request_type") == "error":
             logger.error(f"Received invalid or error query plan: {query_plan.get('error', 'Unknown error')}")
             return {"error": "Failed to retrieve due to invalid query plan.", "products": [], "result_count": 0, "search_successful": False}

        if not query_plan.get("is_in_scope", False):
             logger.info("Query determined to be out of scope by planner. Skipping retrieval.")
             return {"error": "Query is out of scope (not related to Refrigerator or Dishwasher parts).", "products": [], "result_count": 0, "search_successful": False, "is_out_of_scope": True}

        search_query = message
        filters = None
        top_k = self.default_top_k

        try:
            # 1. Prepare search query, filters, and top_k from the plan
            search_query, filters, top_k = self._prepare_search_query_and_filters(query_plan, message)

            # 2. Search the vector database
            logger.info(f"Searching vector DB with query='{search_query}', top_k={top_k}, filters={filters}")
            raw_products = await self.vector_db_manager.search_products(
                query_text=search_query,
                top_k=top_k,
                filters=filters
            )
            logger.info(f"Vector DB returned {len(raw_products)} raw results.")

            # 3. Process results (includes compatibility/installation checks)
            processed_results = self._process_search_results(raw_products, query_plan)

            # 4. Package the results
            retrieval_result = {
                "query": message,
                "query_plan": query_plan,
                "search_query_used": search_query,
                "filters_applied": filters,
                "products": processed_results,
                "result_count": len(processed_results),
                "search_successful": len(processed_results) > 0
            }

            # Add specific notes based on request type and results
            request_type = query_plan.get("request_type")
            if not retrieval_result["search_successful"]:
                retrieval_result["no_results_reason"] = f"No matching products found for the specified criteria (type: {request_type})."
                logger.warning(f"No products found for query: '{search_query}' with filters: {filters}")
            elif request_type == "compatibility_check" and query_plan.get("model_number"):
                 retrieval_result["compatibility_note"] = f"Found {len(processed_results)} part(s) potentially compatible with model {query_plan['model_number']}."
            elif request_type == "installation_guide":
                 guides_found = sum(1 for p in processed_results if p.get("installation_guide") and p["installation_guide"] != "N/A")
                 retrieval_result["installation_note"] = f"Found installation information for {guides_found} out of {len(processed_results)} retrieved part(s)."


            logger.info(f"Successfully retrieved and processed {retrieval_result['result_count']} products for request type '{request_type}'.")
            return retrieval_result

        except Exception as e:
            logger.exception(f"Error during information retrieval for query '{message}': {str(e)}")
            return {
                "query": message,
                "query_plan": query_plan,
                "search_query_used": search_query,
                "filters_applied": filters,
                "products": [],
                "result_count": 0,
                "search_successful": False,
                "error": f"Information retrieval failed: {str(e)}"
            }
