"""
Conversation management for multi-turn and single-turn interactions.
"""

import uuid
import time
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

from .messages import Message, UserMessage, AssistantMessage, ToolMessage
from .config import BenchmarkConfig
from ..agents.assistant import TravelAssistant, TravelAssistantState
from ..simulators.user_simulator import TravelUserSimulator, UserSimulatorState

class ConversationMode(Enum):
    """Different conversation modes supported by the framework."""
    MULTI_TURN = "multi_turn"  # Traditional back-and-forth conversation
    SINGLE_TURN = "single_turn"  # Single user input, multi-step assistant processing

class ConversationStep(BaseModel):
    """Represents a single step within a conversation turn."""
    
    step_id: int = Field(description="Step number within the turn")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    assistant_message: Optional[AssistantMessage] = Field(default=None)
    tool_messages: List[ToolMessage] = Field(default_factory=list)
    duration: float = Field(default=0.0, description="Step duration in seconds")


class ConversationTurn(BaseModel):
    """Represents a single turn in the conversation."""
    
    turn_id: int = Field(description="Turn number")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    user_message: Optional[UserMessage] = Field(default=None)
    assistant_message: Optional[AssistantMessage] = Field(default=None)
    steps: List[ConversationStep] = Field(default_factory=list, description="Assistant processing steps (for single-turn mode)")
    duration: float = Field(default=0.0, description="Turn duration in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    num_steps: int = Field(default=0, description="Number of processing steps (for single-turn mode)")
    
    @property
    def final_assistant_message(self) -> Optional[AssistantMessage]:
        """Get the final assistant message from this turn."""
        if self.steps:
            return self.steps[-1].assistant_message
        return self.assistant_message


class ConversationResult(BaseModel):
    """Results of a completed conversation."""
    
    conversation_id: str = Field(description="Unique conversation identifier")
    mode: ConversationMode = Field(default=ConversationMode.MULTI_TURN, description="Conversation mode used")
    total_turns: int = Field(description="Total number of turns")
    duration: float = Field(description="Total conversation duration in seconds")
    completion_reason: str = Field(description="Why the conversation ended")
    
    # Summary statistics
    user_messages_count: int = Field(default=0)
    assistant_messages_count: int = Field(default=0)
    tool_calls_count: int = Field(default=0)
    total_steps: int = Field(default=0, description="Total processing steps across all turns")
    
    # Tool execution statistics
    cache_hits: int = Field(default=0, description="Number of cache hits during tool execution")
    cache_misses: int = Field(default=0, description="Number of cache misses during tool execution")
    tool_errors: int = Field(default=0, description="Number of tool execution errors")
    
    # States
    final_assistant_state: Optional[Dict[str, Any]] = Field(default=None)
    final_user_state: Optional[Dict[str, Any]] = Field(default=None)
    
    # Full conversation history
    turns: List[ConversationTurn] = Field(default_factory=list)
    messages: List[Message] = Field(default_factory=list)
    
    # Performance metrics
    average_response_time: float = Field(default=0.0)
    total_cost: float = Field(default=0.0)
    total_tokens: int = Field(default=0)


class ConversationManager:
    """
    Manages both multi-turn and single-turn conversations.
    """
    
    def __init__(self, config: BenchmarkConfig, mode: ConversationMode = ConversationMode.MULTI_TURN):
        self.config = config
        self.mode = mode
        self.conversation_id = str(uuid.uuid4())
        
        # Conversation state
        self.turns: List[ConversationTurn] = []
        self.messages: List[Message] = []
        self.start_time: Optional[float] = None
        
        # Performance tracking
        self.total_cost = 0.0
        self.total_tokens = 0
        
    def run_conversation(
        self,
        assistant: TravelAssistant,
        user_simulator: Optional[TravelUserSimulator] = None,
        initial_user_message: Optional[str] = None,
        max_steps_per_turn: int = 10
    ) -> ConversationResult:
        """
        Run a conversation in the specified mode.
        
        Args:
            assistant: The travel assistant
            user_simulator: The user simulator (required for multi-turn mode)
            initial_user_message: Initial user message (for single-turn mode)
            max_turns: Maximum number of turns (overrides config)
            max_steps_per_turn: Maximum processing steps per turn (for single-turn mode)
            
        Returns:
            ConversationResult with complete conversation data
        """
        if self.mode == ConversationMode.MULTI_TURN:
            return self._run_multi_turn_conversation(assistant, user_simulator,max_steps_per_turn)
        else:
            return self._run_single_turn_conversation(assistant, initial_user_message, max_steps_per_turn)
    
    def _run_multi_turn_conversation(
        self,
        assistant: TravelAssistant,
        user_simulator: TravelUserSimulator,
        max_turns: Optional[int] = 10,
    ) -> ConversationResult:
        """Run traditional multi-turn conversation."""
        if user_simulator is None:
            raise ValueError("user_simulator is required for multi-turn mode")
        
        max_turns = max_turns or self.config.max_conversation_turns
        self.start_time = time.time()
        
        # Initialize agent states with JSONL data
        assistant_state = assistant.get_initial_state()
        user_state = user_simulator.get_initial_state()
        
        # Start conversation with user simulator's initial message
        initial_message = user_simulator.generate_initial_message(user_state)
        
        # For user_simulator's conversation_history, its own message should be stored as AssistantMessage
        # because user_simulator is essentially an assistant role
        assistant_initial_message = AssistantMessage(
            content=initial_message.content or "",
            turn_idx=initial_message.turn_idx
        )
        user_state.conversation_history.append(assistant_initial_message)
        
        # For conversation manager, keep normal perspective (user_simulator is user role)
        self.messages.append(initial_message)
        
        turn_count = 0
        completion_reason = "unknown"
        try:
            while turn_count < max_turns:
                turn_start_time = time.time()
                turn_count += 1
                current_turn = ConversationTurn(turn_id=turn_count)
                if self.messages:
                    last_message = self.messages[-1]
                    current_turn.user_message = last_message if isinstance(last_message, UserMessage) else None
                    
                    # Assistant processes the message (traditional single response)
                    # For multi-turn mode, we ignore the num_steps return value
                    assistant_message, assistant_state, _ = assistant.generate_response(
                        last_message, assistant_state
                    )
                    current_turn.assistant_message = assistant_message
                    self.messages.append(assistant_message)
                    
                    # Track costs and tokens
                    self._track_usage(assistant_message)

                    
                    # User simulator responds to assistant
                    user_message, user_state = user_simulator.generate_response(
                        assistant_message, user_state
                    )
                    self.messages.append(user_message)
                    
                    # Check if user wants to end conversation
                    if user_simulator.is_conversation_finished(user_message, user_state):
                        completion_reason = "user_ended"
                        current_turn.duration = time.time() - turn_start_time
                        self.turns.append(current_turn)
                        break
                
                current_turn.duration = time.time() - turn_start_time
                self.turns.append(current_turn)
            
            if turn_count >= max_turns:
                completion_reason = "max_turns_reached"
                
        except Exception as e:
            completion_reason = f"error: {str(e)}"
        return self._create_conversation_result(turn_count, completion_reason, assistant_state, user_state)
    
    def _run_single_turn_conversation(
        self,
        assistant: TravelAssistant,
        initial_user_message: Optional[str] = None,
        max_steps: int = 10
    ) -> ConversationResult:
        """Run single-turn conversation with multi-step processing."""
        if not initial_user_message:
            raise ValueError("initial_user_message is required for single-turn mode")
        
        self.start_time = time.time()
        turn_start_time = time.time()
        
        # Initialize assistant state
        assistant_state = assistant.get_initial_state()
        
        # Create initial user message
        user_message = UserMessage(content=initial_user_message, turn_idx=1)
        self.messages.append(user_message)
        
        # Create single turn
        current_turn = ConversationTurn(turn_id=1, user_message=user_message)
        completion_reason = "completed"
        try:
            # Use assistant's generate_response which already handles multi-step tool calling
            # It now returns the number of steps taken
            final_response, assistant_state, num_steps = assistant.generate_response(
                user_message, 
                assistant_state
            )
            
            # Simply record the number of steps
            current_turn.num_steps = num_steps
            
            # Track the final response
            current_turn.assistant_message = final_response
            self.messages.append(final_response)
            self._track_usage(final_response)
            
            current_turn.duration = time.time() - turn_start_time
            self.turns.append(current_turn)
            
        except Exception as e:
            completion_reason = f"error: {str(e)}"
            current_turn.duration = time.time() - turn_start_time
            self.turns.append(current_turn)
        
        return self._create_conversation_result(1, completion_reason, assistant_state, None)
    
    def _track_usage(self, message: AssistantMessage):
        """Track token usage and costs."""
        if hasattr(message, 'usage') and message.usage:
            usage = message.usage
            self.total_tokens += usage.get('total_tokens', 0)
        
        if hasattr(message, 'cost') and message.cost:
            self.total_cost += message.cost
    def _create_conversation_result(
        self,
        turn_count: int,
        completion_reason: str,
        assistant_state: Optional[TravelAssistantState],
        user_state: Optional[UserSimulatorState]
    ) -> ConversationResult:
        """Create the final conversation result."""
        total_duration = time.time() - self.start_time
        
        # Use assistant_state's conversation_history as the source of truth for messages
        # This includes all messages including tool calls and tool responses
        # However, we need to ensure we don't lose the user's last message if the conversation ended
        messages_to_save = self.messages
        if assistant_state and hasattr(assistant_state, 'conversation_history'):
            messages_to_save = assistant_state.conversation_history
            
            # Check if self.messages has newer messages (e.g., user's last message)
            # that are not in assistant_state.conversation_history
            if completion_reason == "user_ended":
                messages_to_save = list(messages_to_save) + [self.messages[-1]]
        
        # Calculate statistics from the complete message history
        user_messages_count = sum(1 for msg in messages_to_save if isinstance(msg, UserMessage))
        assistant_messages_count = sum(1 for msg in messages_to_save if isinstance(msg, AssistantMessage))
        tool_calls_count = sum(
            len(msg.tool_calls) for msg in messages_to_save 
            if isinstance(msg, AssistantMessage) and msg.tool_calls
        )
        
        total_steps = sum(turn.num_steps for turn in self.turns)
        
        average_response_time = (
            sum(turn.duration for turn in self.turns) / len(self.turns) 
            if self.turns else 0
        )
        
        # Extract tool execution statistics from assistant metadata
        cache_hits = 0
        cache_misses = 0
        tool_errors = 0
        
        if assistant_state and hasattr(assistant_state, 'metadata'):
            cache_hits = assistant_state.metadata.get('cache_hits', 0)
            cache_misses = assistant_state.metadata.get('cache_misses', 0)
            tool_errors = assistant_state.metadata.get('tool_errors', 0)
        
        # Extract final states
        final_assistant_state = None
        final_user_state = None
        
        if assistant_state:
            if hasattr(assistant_state, 'model_dump'):
                final_assistant_state = assistant_state.model_dump()
            elif hasattr(assistant_state, '__dict__'):
                final_assistant_state = assistant_state.__dict__
        
        if user_state:
            if hasattr(user_state, 'model_dump'):
                final_user_state = user_state.model_dump()
            elif hasattr(user_state, '__dict__'):
                final_user_state = user_state.__dict__
        
        return ConversationResult(
            conversation_id=self.conversation_id,
            mode=self.mode,
            total_turns=turn_count,
            duration=total_duration,
            completion_reason=completion_reason,
            user_messages_count=user_messages_count,
            assistant_messages_count=assistant_messages_count,
            tool_calls_count=tool_calls_count,
            total_steps=total_steps,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            tool_errors=tool_errors,
            final_assistant_state=final_assistant_state,
            final_user_state=final_user_state,
            turns=self.turns,
            messages=messages_to_save,
            average_response_time=average_response_time,
            total_cost=self.total_cost,
            total_tokens=self.total_tokens,
        )
    


class ConversationRunner:
    """
    High-level interface for running conversations and benchmarks.
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
    
    def run_multi_turn_conversation(
        self,
        assistant: TravelAssistant,
        user_simulator: TravelUserSimulator,
        max_turns: int = 10
    ) -> ConversationResult:
        """Run a multi-turn conversation with optional JSONL data."""
        
        manager = ConversationManager(self.config, ConversationMode.MULTI_TURN)
        result = manager.run_conversation(
            assistant=assistant, 
            user_simulator=user_simulator, 
            max_steps_per_turn=max_turns
        )
        
        return result
    
    def run_single_turn_conversation(
        self,
        assistant: TravelAssistant,
        user_message: str,
        max_steps: int = 10,
    ) -> ConversationResult:
        """Run a single-turn multi-step conversation with optional JSONL data."""
        
        manager = ConversationManager(self.config, ConversationMode.SINGLE_TURN)
        result = manager.run_conversation(
            assistant=assistant, 
            initial_user_message=user_message,
            max_steps_per_turn=max_steps,
        )
        
        return result