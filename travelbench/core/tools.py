"""
Comprehensive tool system for the travel benchmark framework.
Provides core tool classes, registry, and sandbox caching functionality.
"""

import json
import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from .sandbox_cache import SandboxCacheManager
from .config import SandboxMode, DEFAULT_CACHE_DIR

# Global sandbox state
_sandbox_cache_manager = None
_current_sandbox_mode = SandboxMode.ISOLATED
_llm_simulator = None

def get_sandbox_cache_manager(cache_dir: Optional[str] = None) -> SandboxCacheManager:
    """Get global sandbox cache manager."""
    global _sandbox_cache_manager
    if _sandbox_cache_manager is None:
        _sandbox_cache_manager = SandboxCacheManager(cache_dir or DEFAULT_CACHE_DIR)
    return _sandbox_cache_manager

def set_sandbox_mode(mode: SandboxMode):
    """Set global sandbox mode."""
    global _current_sandbox_mode
    _current_sandbox_mode = mode

def set_llm_simulator(simulator_func):
    """
    Set LLM simulator function for isolated mode.
    
    Args:
        simulator_func: Function that takes (tool_name, params) and returns simulated result.
    """
    global _llm_simulator
    _llm_simulator = simulator_func

def create_and_set_llm_simulator(tool_simulator_config,
                                cache_dir: Optional[str] = None,
                                max_examples: int = 8,
                                use_similarity_retrieval: bool = False,
                                embedding_service_url: Optional[str] = None,
                                embedding_model_name: Optional[str] = None):
    """Create and set an LLM simulator for the sandbox environment.
    
    Args:
        tool_simulator_config: OpenAIConfig for the tool simulator
        cache_dir: Directory containing tool caches
        max_examples: Maximum number of examples to include in prompt
        use_similarity_retrieval: Whether to use similarity-based retrieval (default: False)
        embedding_service_url: URL of remote embedding service (optional)
        embedding_model_name: Model name for embedding service (optional)
        
    Returns:
        Configured LLMToolSimulator instance
    """
    from ..simulators.tool_simulator import create_llm_simulator
    simulator = create_llm_simulator(
        tool_simulator_config=tool_simulator_config,
        cache_dir=cache_dir or DEFAULT_CACHE_DIR,
        max_examples=max_examples,
        use_similarity_retrieval=use_similarity_retrieval,
        embedding_service_url=embedding_service_url,
        embedding_model_name=embedding_model_name
    )
    set_llm_simulator(simulator.simulate_tool_response)
    return simulator

class Tool(ABC):
    """Base class for all tools."""
    
    def __init__(self, name: str, description: str, input_schema: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.input_schema = input_schema or {
            "type": "object",
            "properties": {},
            "required": []
        }
    
    @abstractmethod
    def execute(self, **kwargs) -> str:
        """Execute the tool with given parameters."""
        pass
    
    def to_openai_format(self) -> Dict[str, Any]:
        """Convert tool to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema
            }
        }

class SandboxBaseTool(Tool):
    """
    Base class for sandbox-enabled tools.
    
    Args:
        name: Tool name.
        description: Tool description.
        input_schema: JSON schema for input parameters.
        cache_dir: Directory for cache storage.
    """
    
    def __init__(self, name: str, description: str, input_schema: Optional[Dict[str, Any]] = None, 
                 cache_dir: Optional[str] = None):
        super().__init__(name, description, input_schema)
        self.cache_dir = cache_dir
        self._cache_manager = get_sandbox_cache_manager(cache_dir)
    
    def _has_error(self, result: str) -> bool:
        """Check if result contains error field."""
        try:
            result_dict = json.loads(result)
            return 'error' in result_dict
        except (json.JSONDecodeError, TypeError):
            return False
    
    def _cached_execute(self, time: str, params: str) -> tuple[bool, str]:
        """
        Execute with caching based on sandbox mode.
        
        Args:
            time: Current time string.
            params: JSON string of tool parameters.
        
        Returns:
            Tuple of (cache_hit, result).
        """
        global _current_sandbox_mode, _llm_simulator
        
        cache_key = json.dumps({"time": time, "params": params}, sort_keys=True, ensure_ascii=False)
        
        if _current_sandbox_mode == SandboxMode.ONLINE:
            raise NotImplementedError("Sandbox does not support online mode")
        
        elif _current_sandbox_mode == SandboxMode.ISOLATED:
            cached_result = self._cache_manager.get(self.name, cache_key)
            if cached_result is not None:
                return True, cached_result
            if _llm_simulator:
                try:
                    simulated_result = _llm_simulator(self.name, params)
                    if simulated_result == "":
                        simulated_result = "没有找到相关信息，请确认输入参数是否正确；如无误，可能当前暂无相关数据"
                    self._cache_manager.save_missed_call(self.name, cache_key, simulated_result)
                    return False, simulated_result
                except Exception as e:
                    error_result = json.dumps({'error': f'LLM simulation failed: {str(e)}'}, ensure_ascii=False)
                    self._cache_manager.save_missed_call(self.name, cache_key, error_result)
                    return False, error_result
            else:
                error_result = json.dumps({'error': 'No cached data and no LLM simulator configured'}, ensure_ascii=False)
                self._cache_manager.save_missed_call(self.name, cache_key, error_result)
                return False, error_result
        
        result = self._real_execute(time, params)
        if result == "":
            result = "没有找到相关信息，请确认输入参数是否正确；如无误，可能当前暂无相关数据"
        return False, result
    
    @abstractmethod
    def _real_execute(self, time: str, params: str) -> str:
        """Execute the actual tool logic."""
        pass

    @abstractmethod
    def _validate_parameters(self, time: str, params: str) -> str:
        """Validate tool parameters. Returns error message or empty string if valid."""
        pass

    def _pre_process(self, time: str, params: str) -> str:
        """Pre-process the tool call params."""
        return params
    
    def _post_process(self, time: str, params: str, result: str) -> str:
        """Post-process the tool call result."""
        return result
    
    def execute(self, time: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the tool with caching.
        
        Args:
            time: Current time string.
            **kwargs: Tool parameters.
        
        Returns:
            Dict containing result, cache_hit, and has_error.
        """
        params = json.dumps(kwargs, sort_keys=True, ensure_ascii=False)
        result = self._validate_parameters(time, params)
        # Validation passes if result is empty string
        if result:
            return {
                "result": result,
                "cache_hit": False,
                "has_error": True
            }
        cache_hit, result = self._cached_execute(time, params)
        return {
            "result": result,
            "cache_hit": cache_hit,
            "has_error": False
        }


class ToolRegistry:
    """Registry for managing available tools."""
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool) -> None:
        """Register a tool."""
        if tool.name in self._tools:
            raise ValueError(f"Tool {tool.name} already registered")
        
        self._tools[tool.name] = tool
    
    def get_tool(self, name: str) -> Optional[Tool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def get_tools(self) -> List[Tool]:
        """Get all tools."""
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """Get tool names."""
        return list(self._tools.keys())
    
    def execute_tool(self, name: str, time: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a tool by name.
        
        Args:
            name: Tool name.
            time: Current time string.
            **kwargs: Tool parameters.
        
        Returns:
            Dict containing result, cache_hit, and has_error.
        """
        tool = self.get_tool(name)
        if tool is None:
            raise ToolNotFoundError(f"Tool not found: {name}")
        
        return tool.execute(time, **kwargs)
    
    
    def to_openai_format(self) -> List[Dict[str, Any]]:
        """Get tools in OpenAI format."""
        tools = self.get_tools()
        return [tool.to_openai_format() for tool in tools]

class ToolValidationError(Exception):
    """Exception raised for tool parameter validation errors."""
    pass

class ToolNotFoundError(Exception):
    """Exception raised when a tool is not found."""
    pass

class SandboxToolRegistry(ToolRegistry):
    """Enhanced tool registry with sandbox capabilities."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        super().__init__()
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self._sandbox_mode = SandboxMode.ISOLATED
    
    def set_sandbox_mode(self, mode: SandboxMode):
        """Set sandbox mode for this registry."""
        self._sandbox_mode = mode
        set_sandbox_mode(mode)
    
    def get_cache_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get cache statistics for all tools."""
        from .sandbox_cache import get_cache_stats
        return get_cache_stats(self.cache_dir)
    
    def clear_all_caches(self):
        """Clear all caches for all tools."""
        cache_manager = get_sandbox_cache_manager(self.cache_dir)
        cache_manager.force_save_all()
        
        # Remove all cache files (.json format)
        if os.path.exists(self.cache_dir):
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('_cache.json') or filename.endswith('_missed.json'):
                    try:
                        os.remove(os.path.join(self.cache_dir, filename))
                    except Exception as e:
                        logging.error(f"Failed to remove cache file {filename}: {e}")

# Global tool registry instance
sandbox_tool_registry = SandboxToolRegistry()


# Convenience function for getting cache stats
def get_cache_stats(cache_dir: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Get cache statistics."""
    from .sandbox_cache import get_cache_stats as _get_cache_stats
    return _get_cache_stats(cache_dir or DEFAULT_CACHE_DIR)

