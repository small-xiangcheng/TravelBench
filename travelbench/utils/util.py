"""
Utility functions for the TravelBench framework.

This module provides common utilities including:
- Configuration creation
- File loading
- Task execution helpers
- Logging utilities
"""

import json
import os
from typing import List, Dict, Any, Optional, Callable, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

__all__ = [
    # Constants
    'DEFAULT_MAX_STEPS',
    'DEFAULT_MAX_TURNS', 
    'DEFAULT_MODELS',
    # Logging utilities
    'log_task_result',
    # File utilities
    'load_datas_from_file',
    # Configuration utilities
    'create_default_config',
    # Execution utilities
    'execute_tasks_with_concurrency',
    # Tool utilities
    'get_available_tools_for_openai',
    # Result summary utilities
    'create_results_summary',
    'print_results_summary',
]

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MAX_STEPS = 20
DEFAULT_MAX_TURNS = 10

DEFAULT_MODELS = {
    "agent": "gpt-4",
    "user": "gpt-3.5-turbo",
    "tool": "gpt-4",
}

def log_task_result(result: Dict[str, Any], current: int, total: int, debug: bool = False):
    """Unified task result logging function."""
    mode = result.get("mode", "unknown")
    query = result.get("query", "")
    
    if result["success"]:
        steps_info = f", {result['total_steps']} steps" if mode == "single_turn" else ""
        print(f"✅ [{current}/{total}] {mode.upper()}: {query[:45]}... | "
              f"{result['turns']} turns, {result['duration']:.1f}s{steps_info}, ")
        
        # Debug mode: show detailed conversation info
        if debug:
            print(f"   🔍 Debug - Conversation ID: {result['conversation_id']}")
            print(f"   🔍 Debug - Completion reason: {result['completion_reason']}")
            print(f"   🔍 Debug - Tool calls: {result['tool_calls_count']}")
            if 'messages' in result:
                print(f"   🔍 Debug - Total messages: {len(result['messages'])}")
                # Show message details in debug mode
                for i, msg in enumerate(result['messages'][:5]):  # Show first 5 messages
                    msg_type = msg.get('type', 'unknown')
                    content_preview = str(msg.get('content', ''))[:100] + '...' if len(str(msg.get('content', ''))) > 100 else str(msg.get('content', ''))
                    print(f"   🔍 Debug - Message {i+1} ({msg_type}): {content_preview}")
                if len(result['messages']) > 5:
                    print(f"   🔍 Debug - ... and {len(result['messages'])-5} more messages")
    else:
        print(f"❌ [{current}/{total}] {mode.upper()}: {query[:45]}... | "
              f"Error: {result['error']}")
        
        # Debug mode: show error details
        if debug:
            import logging
            logging.error(f"Task failed for query: {query}")
            logging.error(f"Error details: {result['error']}")



def load_datas_from_file(file_path: str) -> List[Dict[str, Any]]:
    """Load data entries from a JSONL file.
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of parsed JSON objects from the file
    """
    datas = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    datas.append(json.loads(line))
        print(f"✅ Loaded {len(datas)} queries from {file_path}")
        return datas

    except Exception as e:
        print(f"❌ Error loading file {file_path}: {e}")
        return []


# ============================================================================
# Configuration Utilities
# ============================================================================

def create_default_config(
    agent_llm: str = "gpt-4",
    user_llm: str = "gpt-3.5-turbo", 
    tool_llm: str = "gpt-4",
    sandbox_mode: str = "isolated",
    cache_dir: Optional[str] = None
):
    """Create default benchmark configuration with overridable parameters.
    
    Args:
        agent_llm: Model name for the travel assistant agent
        user_llm: Model name for the user simulator
        tool_llm: Model name for tool simulation
        sandbox_mode: Sandbox execution mode ("isolated")
        cache_dir: Directory for caching sandbox results
        
    Returns:
        BenchmarkConfig: Configured benchmark settings
    """
    from ..core.config import (
        BenchmarkConfig, OpenAIConfig, SandboxConfig, SandboxMode
    )
    
    # Get API configuration from environment
    api_key = os.getenv('OPENAI_API_KEY', 'your-openai-api-key-here')
    api_base = os.getenv('OPENAI_API_BASE', 'your-openai-api-base-here')

    # Convert sandbox mode string to enum
    sandbox_mode_map = {
        "isolated": SandboxMode.ISOLATED,
    }
    
    # Read embedding service configuration from environment variables
    embedding_service_url = os.getenv('EMBEDDING_SERVICE_URL')
    embedding_model_name = os.getenv('EMBEDDING_MODEL_NAME')
    use_remote_embedding = bool(embedding_service_url and embedding_model_name)
    
    # Build sandbox config with optional cache_dir
    sandbox_kwargs = {
        "enabled": True,
        "mode": sandbox_mode_map.get(sandbox_mode, SandboxMode.ISOLATED),
        "max_examples": 8,
        "simulation_temperature": 0.0,
        "auto_setup_llm_simulator": True,
        "use_remote_embedding": use_remote_embedding,
        "embedding_service_url": embedding_service_url,
        "embedding_model_name": embedding_model_name
    }
    
    # Only set cache_dir if explicitly provided
    if cache_dir is not None:
        sandbox_kwargs["cache_dir"] = cache_dir
    
    config = BenchmarkConfig(
        assistant_config=OpenAIConfig(
            model_name=agent_llm,
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=8192
        ),
        user_simulator_config=OpenAIConfig(
            model_name=user_llm,
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=8192
        ),
        tool_simulator_config=OpenAIConfig(
            model_name=tool_llm,
            api_key=api_key,
            api_base=api_base,
            temperature=0.0,
            max_tokens=8192
        ),
        max_conversation_turns=20,
        conversation_timeout=300,
        sandbox_config=SandboxConfig(**sandbox_kwargs)
    )
    
    return config


# ============================================================================
# Concurrent Execution Utilities
# ============================================================================

def execute_tasks_with_concurrency(
    tasks: List[Tuple[str, Dict[str, Any], int]], 
    task_executor: Callable,
    max_concurrency: int = 1,
    debug: bool = False,
) -> List[Dict[str, Any]]:
    """Execute tasks with configurable concurrency using ThreadPoolExecutor.
    
    This function provides unified handling for both single-threaded and 
    multi-threaded execution scenarios.
    
    Args:
        tasks: List of task tuples (mode, data, trial_id)
        task_executor: Callable that executes a single task, 
                      signature: (mode, data, trial_id) -> Dict[str, Any]
        max_concurrency: Maximum number of concurrent workers
        debug: Enable debug logging
    
    Returns:
        List of execution results
    """
    results = []
    
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        # Submit all tasks
        future_to_task = {}
        
        for mode, data, trial in tasks:
            future = executor.submit(task_executor, mode, data, trial)
            future_to_task[future] = (mode, data, trial)
        
        # Process completed tasks
        for i, future in enumerate(as_completed(future_to_task), 1):
            mode, data, trial = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                
                # Show progress for single-threaded execution
                if max_concurrency == 1:
                    query = data.get('query', '')[:45]
                    print(f"🔄 [{i}/{len(tasks)}] Processing {mode.upper()}: {query}...")
                
                log_task_result(result, i, len(tasks), debug)
                    
            except Exception as e:
                error_msg = str(e)
                if debug:
                    print(f"❌ Task failed with traceback:")
                    traceback.print_exc()
                    error_msg = traceback.format_exc()
                else:
                    print(f"❌ Task failed: {e}")
                    
                results.append({
                    "mode": mode,
                    "data": data,
                    "trial_id": trial, 
                    "success": False,
                    "error": error_msg
                })
    
    return results


# ============================================================================
# Tool Utilities
# ============================================================================

def get_available_tools_for_openai() -> List[Dict[str, Any]]:
    """Get all registered tools in OpenAI function calling format.
    
    Returns tools filtered by the TOOL_NAMES whitelist defined in tool_list.py.
    
    Returns:
        List of tool specifications formatted for OpenAI function calling API
    """
    from ..core.tools import sandbox_tool_registry
    from ..tools.tool_list import TOOL_NAMES
    
    all_tools = sandbox_tool_registry.to_openai_format()
    return [
        tool for tool in all_tools 
        if tool.get("function", {}).get("name") in TOOL_NAMES
    ]


# ============================================================================
# Result Summary Utilities
# ============================================================================

def create_results_summary(
    results: List[Dict[str, Any]],
    datas: List[Dict[str, Any]],
    mode: str,
    config_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Create a summary of conversation results.
    
    Args:
        results: List of conversation results
        datas: Original input data
        mode: Conversation mode ("single_turn" or "multi_turn")
        config_info: Configuration information to include
        
    Returns:
        Summary dictionary with statistics
    """
    successful_results = [r for r in results if r.get("success", False)]
    
    summary = {
        "total_datas": len(datas),
        "mode": mode,
        "total_trials": len(results),
        "successful_conversations": len(successful_results),
        "success_rate": len(successful_results) / len(results) if results else 0,
        "average_turns": (
            sum(r.get("turns", 0) for r in successful_results) / len(successful_results) 
            if successful_results else 0
        ),
        "average_duration": (
            sum(r.get("duration", 0) for r in successful_results) / len(successful_results) 
            if successful_results else 0
        ),
        "average_steps": (
            sum(r.get("total_steps", 0) for r in successful_results) / len(successful_results) 
            if successful_results else 0
        ),
        "total_tool_calls": sum(r.get("tool_calls_count", 0) for r in successful_results),
        "config": config_info
    }
    
    return summary


def print_results_summary(summary: Dict[str, Any], output_file: str) -> None:
    """Print formatted results summary to console.
    
    Args:
        summary: Summary dictionary from create_results_summary
        output_file: Path where results were saved
    """
    mode = summary.get("mode", "unknown")
    
    print(f"\n📊 Results Summary:")
    print(f"   Mode: {mode}")
    print(f"   Total datas: {summary['total_datas']}")
    print(f"   Total trials: {summary['total_trials']}")
    print(f"   Success rate: {summary['success_rate']:.1%} "
          f"({summary['successful_conversations']}/{summary['total_trials']})")
    print(f"   Average turns: {summary['average_turns']:.1f}")
    print(f"   Average duration: {summary['average_duration']:.1f}s")
    
    if mode == "single_turn":
        print(f"   Average steps: {summary['average_steps']:.1f}")
        
    print(f"   Total tool calls: {summary['total_tool_calls']}")
    print(f"   Results saved to: {output_file}")

