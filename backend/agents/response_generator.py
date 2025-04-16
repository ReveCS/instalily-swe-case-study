import logging
import json
from typing import Dict, Any, List
from .base import BaseAgent
from chromadb import Client as ChromaClient
from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

class ResponseGenerationAgent(BaseAgent):
    """Agent responsible for generating natural language responses to user queries"""

    def __init__(self, openai_client: AsyncOpenAI):
        super().__init__(openai_client)
    
    def _get_system_prompt(self) -> str:
        return """
        You are a helpful appliance parts assistant for PartSelect, specializing in refrigerator and dishwasher parts. Your job is to:
        1. Craft natural, helpful responses using the product information provided
        2. Maintain a conversational and friendly tone
        3. ALWAYS reference specific part numbers (format: PS followed by numbers, e.g. PS12345678), model numbers, brands, and categories in your responses
        4. Highlight the most relevant product details based on the user's query
        5. Provide comparisons when multiple products are involved
        6. Suggest additional information that might be helpful
        
        Never include information that wasn't provided in the product data, and be transparent
        about any limitations in the available information.
        """
    
    def _format_context_for_llm(self, message: str, query_plan: Dict[str, Any], product_info: Dict[str, Any]) -> str:
        """Combines user message and retrieved context into a single prompt string."""
        context_parts = []
        context_parts.append(f"User Query: {message}")

        if query_plan:
            plan_summary = {k: v for k, v in query_plan.items() if k in ['request_type', 'part_number', 'model_number', 'appliance_type', 'brand', 'symptom', 'component'] and v}
            if plan_summary:
                 context_parts.append(f"\nQuery Context: {json.dumps(plan_summary)}")

        if product_info and product_info.get("products"):
            context_parts.append("\nRetrieved Product Information:")
            max_products_to_show = 3
            for i, product in enumerate(product_info["products"][:max_products_to_show]):
                details = {
                    "name": product.get("name"),
                    "part_number": product.get("part_number"),
                    "manufacturer_part_number": product.get("manufacturer_part_number"),
                    "price": product.get("price"),
                    "availability": product.get("availability"),
                    "description": (product.get("description") or "")[:200] + "...",
                    "url": product.get("url"),
                    "compatibility_result": product.get("compatibility_result") if product.get("compatibility_result") != "N/A" else None,
                    "installation_guide": product.get("installation_guide") if product.get("installation_guide") != "N/A" else None,
                }
                details = {k: v for k, v in details.items() if v is not None}
                context_parts.append(f"- Product {i+1}: {json.dumps(details)}")
            if len(product_info["products"]) > max_products_to_show:
                context_parts.append(f"- ... (and {len(product_info['products']) - max_products_to_show} more results not shown)")
        elif product_info and product_info.get("error"):
             context_parts.append(f"\nNote: Information retrieval failed or returned no results. Reason: {product_info.get('error')}")
        elif product_info and not product_info.get("products"):
             context_parts.append("\nNote: No specific products were found matching the query.")


        context_parts.append("\nBased on the above information, please generate a helpful and conversational response to the user's query.")
        return "\n".join(context_parts)

    async def generate_response(self, message: str, query_plan: Dict[str, Any], 
                                product_info: Dict[str, Any]) -> str:
        """
        Generate a response based on query plan and retrieved information
        
        Args:
            message: The user's message
            query_plan: The plan created by the QueryPlanningAgent
            product_info: The information retrieved by the RetrievalAgent
            
        Returns:
            A natural language response as a string
        """
        try:
            formatted_prompt = self._format_context_for_llm(message, query_plan, product_info)

            response = await self.run(
                message=formatted_prompt
            )
            return response
        except Exception as e:
            logger.error(f"Error in response generation: {str(e)}")
            raise RuntimeError(f"Response generation failed: {str(e)}")