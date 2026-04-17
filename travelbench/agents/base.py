"""
Base agent classes for the travel benchmark framework.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field

from ..core.messages import Message

class AgentState(BaseModel):
    """Base state for agents."""
    
    conversation_history: List[Message] = Field(default_factory=list)
    turn_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BaseAgent(ABC):
    """Base class for all agents in the framework."""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
    
    @abstractmethod
    def generate_response(
        self, 
        message: Message, 
        state: AgentState
    ) -> Union[Tuple[Message, AgentState], Tuple[Message, AgentState, int]]:
        """
        Generate a response to the input message.
        
        Args:
            message: Input message
            state: Current agent state
            
        Returns:
            Tuple of (response_message, updated_state) or 
            Tuple of (response_message, updated_state, num_steps) for assistants that track steps
        """
        pass
    
    @abstractmethod
    def get_initial_state(self) -> AgentState:
        """Get the initial state for the agent."""
        pass

class TravelAssistantState(AgentState):
    """Extended state for travel assistant."""
    query_time: str = ""
    context: str = ""  # Context information from JSONL data


class UserSimulatorState(AgentState):
    """State for user simulator."""
    query: str = ""
    user_profile: str = ""  # User profile from JSONL data
    query_time: str = ""  # Time information from JSONL data
