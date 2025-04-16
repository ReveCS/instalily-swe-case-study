from typing import TypedDict, Annotated, Sequence, Dict, Any
from langchain_core.messages import BaseMessage

class ChatState(TypedDict):
    """State schema for the e-commerce chatbot workflow"""
    messages: Annotated[Sequence[BaseMessage], "The message history"]
    query_plan: Annotated[Dict[str, Any], "The plan for how to answer the query"]
    product_info: Annotated[Dict[str, Any], "Retrieved product information"] 
    current_response: Annotated[str, "The current response being built"]
    error: Annotated[str, "Error message if any step fails"]