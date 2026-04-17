"""
OpenAI API client for the travel benchmark framework.
Handles all interactions with OpenAI API including tool calling.
"""

import json
import time
from typing import List, Dict, Any, Optional, Tuple, Union
from openai import OpenAI

from .config import OpenAIConfig
from .messages import Message, AssistantMessage, ToolCall, to_openai_messages
from .tools import Tool


class OpenAIClient:
    """OpenAI API client with tool calling support."""
    
    def __init__(self, config: OpenAIConfig):
        self.config = config
        # Check if using OpenRouter API
        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
            timeout=config.timeout
        )
        self.enable_thinking = config.enable_thinking
    
    def generate_response(
        self,
        messages: List[Message],
        tools: Optional[List[Tool]] = None,
        **kwargs
    ) -> Tuple[AssistantMessage, Dict[str, Any]]:
        """
        Generate response from OpenAI API.
        
        Args:
            messages: Conversation messages
            tools: Available tools for the assistant
            **kwargs: Additional arguments to override config
            
        Returns:
            Tuple of (AssistantMessage, usage_info)
        """
        # Convert messages to OpenAI format
        openai_messages = to_openai_messages(messages)
        
        # Prepare request parameters
        request_params = {
            "model": self.config.model_name,
            "messages": openai_messages,
            "temperature": kwargs.get("temperature", self.config.temperature),
        }
        if self.config.max_tokens is not None:
            request_params["max_tokens"] = kwargs.get("max_tokens", self.config.max_tokens)
        
        # Add tools if provided
        if tools:
            request_params["tools"] = [tool.to_openai_format() for tool in tools]
            request_params["tool_choice"] = "auto"
        
        # Add extra_bodyto enable reasoning
        if self.enable_thinking:
            request_params["extra_body"] = {"enable_thinking": True}
        # Retry mechanism
        max_retries = 5
        retry_delay = 5  # seconds
        
        for attempt in range(max_retries):
            try:
                # Make API call
                response = self.client.chat.completions.create(**request_params)
                # Extract response data
                choice = response.choices[0]
                message = choice.message
                # Handle content - it might be a string, list, or None
                if message.content is None:
                    content = None
                elif isinstance(message.content, str):
                    content = message.content.strip()
                elif isinstance(message.content, list):
                    # Convert list to string (e.g., for structured outputs)
                    content = json.dumps(message.content, ensure_ascii=False)
                else:
                    # Fallback: convert to string
                    content = str(message.content)
                # Create AssistantMessage
                assistant_message = AssistantMessage(
                    content=content,
                    usage=response.usage.model_dump() if response.usage else None
                )
                
                # Handle tool calls if present
                if message.tool_calls:
                    tool_calls = []
                    for tc in message.tool_calls:
                        try:
                            arguments = json.loads(tc.function.arguments)
                        except json.JSONDecodeError:
                            # Handle malformed JSON
                            arguments = {"raw_arguments": tc.function.arguments}
                        
                        tool_calls.append(ToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments
                        ))
                    assistant_message.tool_calls = tool_calls
                
                # Usage information
                usage_info = {
                    "usage": response.usage.model_dump() if response.usage else None,
                    "model": response.model,
                    "finish_reason": choice.finish_reason
                }
                
                return assistant_message, usage_info
                
            except Exception as e:
                if attempt < max_retries - 1:
                    # Wait before retrying
                    time.sleep(retry_delay)
                else:
                    # Last attempt failed, raise exception
                    raise OpenAIClientError(
                        f"OpenAI API call failed after {max_retries} attempts: {str(e)}"
                    ) from e
    
    def create_embeddings(
        self,
        texts: Union[str, List[str]],
        model: Optional[str] = None,
        **kwargs
    ) -> Tuple[List[List[float]], Dict[str, Any]]:
        """
        Create embeddings using OpenAI-compatible API.
        
        Args:
            texts: Text or list of texts to embed
            model: Model name (defaults to config model)
            **kwargs: Additional arguments
            
        Returns:
            Tuple of (embeddings, usage_info)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Retry mechanism
        max_retries = 5
        retry_delay = 5
        
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    input=texts,
                    model=model or self.config.model_name,
                    **kwargs
                )
                
                # Extract embeddings
                embeddings = [item.embedding for item in response.data]
                
                # Usage info
                usage_info = {
                    "usage": response.usage.model_dump() if hasattr(response, 'usage') else None,
                    "model": response.model
                }
                
                return embeddings, usage_info
                
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(retry_delay)
                else:
                    raise OpenAIClientError(
                        f"Embeddings API call failed after {max_retries} attempts: {str(e)}"
                    ) from e

class OpenAIClientError(Exception):
    """Exception raised for OpenAI client errors."""
    pass

