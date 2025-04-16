from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from abc import ABC, abstractmethod
import logging
from chromadb import Client as ChromaClient
from openai import AsyncOpenAI
from langchain_core.runnables import Runnable
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.deepseek import DeepSeekProvider


logger = logging.getLogger(__name__)

class BaseAgent(ABC):
    """Base class for all agents in the system"""
    
    def __init__(self, openai_client: AsyncOpenAI):
        self.openai_client = openai_client
        self.agent: Optional[Runnable] = None
        self._create_agent_internal()
        
    def _create_model(self):
        return OpenAIModel(
            'deepseek-chat',
            provider=DeepSeekProvider(api_key='sk-c91d53ba31784680a1bd96c612c9bbad'),
        )   
        
    def _create_agent_internal(self) -> Agent:
        """Create the agent with the appropriate system prompt"""
        logger.debug(f"Creating pydantic_ai Agent for {self.__class__.__name__}")
        try:
            model = self._create_model()
            # Call the abstract method correctly
            system_prompt = self._get_system_prompt()

            # Initialize the Pydantic-AI Agent
            # Removed deps_type and deps assignment as they aren't standard for Agent.run input handling
            self.agent = Agent(
                model,
                system_prompt=system_prompt,
                retries=2 # Keep retries if desired
            )
            logger.info(f"Pydantic-AI Agent created successfully for {self.__class__.__name__}")
        except Exception as e:
             logger.exception(f"Error creating pydantic_ai Agent for {self.__class__.__name__}: {e}")
             self.agent = None # Ensure agent is None if creation fails
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Return the system prompt for this agent. To be implemented by subclasses."""
        pass

    async def run(self, **kwargs) -> Any:
        """Runs the pydantic_ai agent with formatted messages."""
        message = kwargs.get("message")
        # Expect history to be List[BaseMessage] now

        if not message:
            logger.error(f"{self.__class__.__name__}.run called without 'message'.")
            return {"error": "Input message missing"}

        try:
            result = await self.agent.run(
                user_prompt=message
            )
            logger.debug(f"Agent ({self.__class__.__name__}) returned result: {str(result)[:200]}...")
            return result.output

        except Exception as e:
            logger.exception(f"Exception during pydantic_ai Agent invocation for {self.__class__.__name__}: {str(e)}")
            return {"error": f"Agent invocation failed: {str(e)}"}
