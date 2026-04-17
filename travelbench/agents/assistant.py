"""
Travel Assistant agent implementation.
"""

from typing import Tuple

from ..core.config import OpenAIConfig
from ..core.openai_client import OpenAIClient
from ..core.messages import (
    Message, SystemMessage, AssistantMessage, ToolMessage
)
from .prompt import MULTI_TURN_PROMPT,SINGLE_TURN_PROMPT,UNSOLVED_PROMPT
from ..core.tools import sandbox_tool_registry
from .base import BaseAgent, TravelAssistantState


class TravelAssistant(BaseAgent):
    """
    Travel Assistant agent that can help users with travel-related tasks.
    Supports tool calling and multi-turn conversations.
    """
    
    # Default system prompt for travel assistant

    
    def __init__(
        self,
        config: OpenAIConfig,
        time: str = "",
        context: str = "",
        multi_turn: bool = True,
        unsolved: bool = False
    ):
        super().__init__("","")
        
        self.config = config
        self.client = OpenAIClient(config)
        self.time = time
        self.context = context
        self.system_prompt = self._build_system_prompt(multi_turn,unsolved)
        # Tool management - use sandbox tool registry for cached tools
        self.tool_registry = sandbox_tool_registry
        
        # Available tools for this assistant
        self.available_tools = self.tool_registry.get_tools()

    def _build_system_prompt(self,multi_turn: bool,unsolved: bool) -> str:
        """Build system prompt with context and time"""
        if multi_turn:
            return MULTI_TURN_PROMPT.format(context=self.context, time=self.time)
        elif not unsolved:
            return SINGLE_TURN_PROMPT.format(context=self.context, time=self.time)
        else:
            return UNSOLVED_PROMPT.format(context=self.context, time=self.time)
        
        
    def get_initial_state(
        self,
    ) -> TravelAssistantState:
        """Get initial state for the assistant with optional context from JSONL data."""
        
        # Build system prompt with context if provided
        system_content = self.system_prompt
        
        return TravelAssistantState(
            conversation_history=[
                SystemMessage(content=system_content)
            ],
            turn_count=0,
            context=self.context,
            query_time=self.time,
        )
    
    def generate_response(
        self, 
        message: Message, 
        state: TravelAssistantState
    ) -> Tuple[AssistantMessage, TravelAssistantState, int]:
        """
        Generate a response to user message, potentially using tools.
        Supports multiple rounds of tool calls until completion.
        
        Returns:
            Tuple of (assistant_message, updated_state, num_steps)
            where num_steps is the number of iterations/steps taken
        """
        # Add user message to conversation history
        state.conversation_history.append(message)
        state.turn_count += 1
        
        try:
            # Loop until we get a response without tool calls
            max_iterations = 20  # Prevent infinite loops
            iteration = 0
            
            while iteration < max_iterations:
                # Get response from OpenAI
                assistant_message, usage_info = self.client.generate_response(
                    messages=state.conversation_history,
                    tools=self.available_tools
                )
                
                # Set turn index
                assistant_message.turn_idx = state.turn_count
                
                # Update metadata with usage info
                state.metadata.update({
                    "last_usage": usage_info,
                    "last_turn": state.turn_count
                })
                
                # Increment iteration counter
                iteration += 1
                
                # If no tool calls, we're done
                if not assistant_message.has_tool_calls():
                    state.conversation_history.append(assistant_message)
                    return assistant_message, state, iteration
                
                # Handle tool calls and continue loop
                state = self._handle_tool_calls(assistant_message, state)
            
            # If we hit max iterations, return a message
            error_message = AssistantMessage(
                content="I apologize, but I've reached the maximum number of tool call iterations. Please try rephrasing your request.",
                turn_idx=state.turn_count
            )
            state.conversation_history.append(error_message)
            return error_message, state, iteration
            
        except Exception as e:
            # Return error message if something goes wrong
            error_message = AssistantMessage(
                content=f"I apologize, but I encountered an error: {str(e)}. Please try again.",
                turn_idx=state.turn_count
            )
            state.conversation_history.append(error_message)
            return error_message, state, 0
    
    def _handle_tool_calls(
        self, 
        assistant_message: AssistantMessage, 
        state: TravelAssistantState
    ) -> TravelAssistantState:
        """
        Execute tool calls and add results to conversation history.
        Returns updated state for the next iteration.
        """
        if not assistant_message.tool_calls:
            return state
        
        # Add the assistant message with tool calls to history
        state.conversation_history.append(assistant_message)
        
        # Initialize metadata counters if not present
        if "cache_hits" not in state.metadata:
            state.metadata["cache_hits"] = 0
        if "cache_misses" not in state.metadata:
            state.metadata["cache_misses"] = 0
        if "tool_errors" not in state.metadata:
            state.metadata["tool_errors"] = 0
        if "tool_calls_executed" not in state.metadata:
            state.metadata["tool_calls_executed"] = 0
        
        # Execute each tool call
        tool_messages = []
        for tool_call in assistant_message.tool_calls:
            try:
                # Execute the tool - returns dict with result, cache_hit, has_error
                tool_result_dict = self.tool_registry.execute_tool(
                    tool_call.name, 
                    self.time,
                    **tool_call.arguments
                )
                
                # Extract information from result dict
                result_content = tool_result_dict.get("result", "")
                cache_hit = tool_result_dict.get("cache_hit", False)
                has_error = tool_result_dict.get("has_error", False)
                
                # Update cache statistics
                if cache_hit:
                    state.metadata["cache_hits"] += 1
                else:
                    state.metadata["cache_misses"] += 1
                
                # Update error statistics
                if has_error:
                    state.metadata["tool_errors"] += 1
                
                # Create tool message with result content
                tool_message = ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=result_content,
                    error=has_error,
                    turn_idx=state.turn_count
                )
            except Exception as e:
                # Handle exceptions
                state.metadata["tool_errors"] += 1
                state.metadata["cache_misses"] += 1  # Exception means no cache hit
                
                tool_message = ToolMessage(
                    tool_call_id=tool_call.id,
                    name=tool_call.name,
                    content=f"Error executing tool: {str(e)}",
                    error=True,
                    turn_idx=state.turn_count
                )
            
            tool_messages.append(tool_message)
            state.conversation_history.append(tool_message)
        
        # Update total tool calls count
        state.metadata["tool_calls_executed"] += len(tool_messages)
        
        return state

