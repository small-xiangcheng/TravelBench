"""
Configuration management for the travel benchmark framework.
"""

import os
from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field
from enum import Enum

# Default cache directory for sandbox environment
# Can be overridden by SANDBOX_CACHE_DIR environment variable
DEFAULT_CACHE_DIR = os.environ.get("SANDBOX_CACHE_DIR", "./sandbox_cache")

class SandboxMode(Enum):
    """Sandbox operation modes."""
    ONLINE = "online"           # deprecated
    ISOLATED = "isolated"       # Cache only + LLM simulation for misses

class SandboxConfig(BaseModel):
    """Sandbox environment configuration."""
    
    enabled: bool = Field(default=False, description="Enable sandbox environment")
    mode: SandboxMode = Field(default=SandboxMode.ISOLATED, description="Sandbox operation mode")
    cache_dir: str = Field(default_factory=lambda: DEFAULT_CACHE_DIR, description="Directory for cache storage")
    max_examples: int = Field(default=8, description="Max examples to use for LLM simulation")
    simulation_temperature: float = Field(default=0.7, description="Temperature for LLM simulation")
    auto_setup_llm_simulator: bool = Field(default=True, description="Auto setup LLM simulator for isolated mode")
    use_similarity_retrieval: bool = Field(default=True, description="Use similarity-based retrieval instead of random sampling")
    
    # Remote embedding service configuration (独立于 agent/user-simulator)
    use_remote_embedding: bool = Field(default=False, description="Use remote embedding service instead of local model")
    embedding_service_url: Optional[str] = Field(default=None, description="Remote embedding service URL (e.g., http://localhost:8001/v1)")
    embedding_model_name: Optional[str] = Field(default=None, description="Embedding model name for remote service")


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""
    
    model_name: str = Field(default="gpt-4", description="Model name to use")
    api_key: str = Field(description="OpenAI API key")
    api_base: str = Field(default="your-openai-api-base-here", description="API base URL")
    temperature: float = Field(default=0.7, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")
    timeout: Optional[int] = Field(default=None, description="Request timeout in seconds (None = no timeout)")
    enable_thinking: bool = Field(default=False, description="Enable thinking")


class BenchmarkConfig(BaseModel):
    """Main benchmark configuration."""
    
    assistant_config: OpenAIConfig = Field(description="Configuration for the assistant")
    user_simulator_config: OpenAIConfig = Field(description="Configuration for the user simulator")
    tool_simulator_config: OpenAIConfig = Field(description="Configuration for the tool simulator")
    max_conversation_turns: int = Field(default=20, description="Maximum conversation turns")
    conversation_timeout: Optional[int] = Field(default=None, description="Conversation timeout in seconds (None = no timeout)")
    seed: Optional[int] = Field(default=None, description="Random seed for reproducibility")
    
    # Travel domain settings
    supported_languages: List[str] = Field(default=["en"], description="Supported languages")
    
    # Sandbox environment settings
    sandbox_config: SandboxConfig = Field(default_factory=SandboxConfig, description="Sandbox environment configuration")
