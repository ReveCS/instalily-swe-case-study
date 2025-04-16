from typing import List, Dict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage

def convert_to_langchain_messages(history: List[Dict[str, str]]) -> List[BaseMessage]:
    """Convert the chat history from API format to Langchain message format"""
    messages = []
    for msg in history:
        if msg.get('role') == 'user':
            messages.append(HumanMessage(content=msg.get('content')))
        elif msg.get('role') == 'assistant':
            messages.append(AIMessage(content=msg.get('content')))
        elif msg.get('role') == 'system':
            messages.append(SystemMessage(content=msg.get('content')))
    return messages

def convert_from_langchain_messages(messages: List[BaseMessage]) -> List[Dict[str, str]]:
    """Convert from Langchain message format to API format"""
    history = []
    for msg in messages:
        if isinstance(msg, HumanMessage):
            history.append({'role': 'user', 'content': msg.content})
        elif isinstance(msg, AIMessage):
            history.append({'role': 'assistant', 'content': msg.content})
        elif isinstance(msg, SystemMessage):
            history.append({'role': 'system', 'content': msg.content})
    return history