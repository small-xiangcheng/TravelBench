"""
Sandbox environment manager for the travel benchmark framework.
Provides easy initialization and management of sandbox environments.
"""

import os
import logging

from .config import BenchmarkConfig, SandboxMode
from .tools import (
    sandbox_tool_registry, set_sandbox_mode, create_and_set_llm_simulator, SandboxToolRegistry
)

class SandboxManager:
    """
    Manages sandbox environment initialization and configuration.
    Provides seamless integration with existing benchmark framework.
    """
    
    def __init__(self, config: BenchmarkConfig):
        """
        Initialize sandbox manager with benchmark configuration.
        
        Args:
            config: Benchmark configuration including sandbox settings
        """
        self.config = config
        self.sandbox_config = config.sandbox_config
        self._initialized = False
        self._llm_simulator = None
        self.logger = logging.getLogger(__name__)
        
    def initialize_sandbox(self) -> bool:
        """
        Initialize sandbox environment based on configuration.
        
        Returns:
            True if sandbox was successfully initialized, False otherwise
        """
        if not self.sandbox_config.enabled:
            return False
        
        try:
            # Set sandbox mode directly (no conversion needed)
            set_sandbox_mode(self.sandbox_config.mode)
            
            # Initialize LLM simulator for ISOLATED mode if needed
            if (self.sandbox_config.mode == SandboxMode.ISOLATED and 
                self.sandbox_config.auto_setup_llm_simulator):
                
                self._setup_llm_simulator()
            
            # Ensure cache directory exists
            os.makedirs(self.sandbox_config.cache_dir, exist_ok=True)
            
            self._initialized = True
            self.logger.info(
                f"Sandbox initialized successfully in {self.sandbox_config.mode.value} mode. "
                f"Cache directory: {self.sandbox_config.cache_dir}"
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sandbox: {e}")
            return False
    
    
    def _setup_llm_simulator(self):
        """Setup LLM simulator for isolated mode."""
        try:
            self._llm_simulator = create_and_set_llm_simulator(
                tool_simulator_config=self.config.tool_simulator_config,
                cache_dir=self.sandbox_config.cache_dir,
                max_examples=self.sandbox_config.max_examples,
                use_similarity_retrieval=self.sandbox_config.use_similarity_retrieval,
                embedding_service_url=self.sandbox_config.embedding_service_url,
                embedding_model_name=self.sandbox_config.embedding_model_name
            )
            
            if self._llm_simulator:
                retrieval_mode = "similarity-based" if self.sandbox_config.use_similarity_retrieval else "random"
                self.logger.info(f"LLM simulator configured successfully with {retrieval_mode} retrieval")
            else:
                self.logger.warning("LLM simulator setup failed, using fallback")
                
        except Exception as e:
            self.logger.error(f"Failed to setup LLM simulator: {e}")
    
    def get_tool_registry(self) -> SandboxToolRegistry:
        """
        Get the appropriate tool registry based on sandbox configuration.
        
        Returns:
            SandboxToolRegistry instance configured for current sandbox mode
        """
        if not self._initialized:
            self.initialize_sandbox()
        
        if self.sandbox_config.enabled:
            return sandbox_tool_registry
        else:
            registry = SandboxToolRegistry(self.sandbox_config.cache_dir)
            registry.set_sandbox_mode(SandboxMode.ISOLATED)
            return registry
    
    def get_cache_statistics(self) -> dict:
        """Get cache statistics for all tools."""
        if not self._initialized:
            return {"error": "Sandbox not initialized"}
        
        try:
            from .tools import get_cache_stats
            return get_cache_stats(self.sandbox_config.cache_dir)
        except Exception as e:
            self.logger.error(f"Failed to get cache statistics: {e}")
            return {"error": str(e)}
    
    def clear_cache(self):
        """Clear all cached data."""
        if not self._initialized:
            self.logger.warning("Sandbox not initialized")
            return
        
        try:
            registry = self.get_tool_registry()
            registry.clear_all_caches()
            self.logger.info("Cache cleared successfully")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def switch_mode(self, new_mode: SandboxMode):
        """
        Switch sandbox to a different mode.
        
        Args:
            new_mode: New sandbox mode to switch to
        """
        if not self._initialized:
            self.logger.warning("Sandbox not initialized")
            return
        
        try:
            self.sandbox_config.mode = new_mode
            set_sandbox_mode(new_mode)
            
            # Setup LLM simulator if switching to isolated mode
            if (new_mode == SandboxMode.ISOLATED and 
                self.sandbox_config.auto_setup_llm_simulator and
                not self._llm_simulator):
                self._setup_llm_simulator()
            
            self.logger.info(f"Switched to {new_mode.value} mode")
            
        except Exception as e:
            self.logger.error(f"Failed to switch mode: {e}")

def create_sandbox_manager(config: BenchmarkConfig, auto_init: bool = True) -> SandboxManager:
    """
    Create and optionally initialize a sandbox manager.
    
    Args:
        config: Benchmark configuration with sandbox settings
        auto_init: Whether to automatically initialize the sandbox
        
    Returns:
        Configured SandboxManager instance
    """
    manager = SandboxManager(config)
    
    if auto_init:
        manager.initialize_sandbox()
    
    return manager
