"""
Main entry point for the TravelBench framework.

This module provides the CLI interface for running travel benchmark 
conversations in both single-turn and multi-turn modes.

Usage:
    python -m travelbench run --query "Plan a trip to Paris" --mode single_turn
    python -m travelbench tools
    python -m travelbench status
"""

import argparse
import json
import sys
import os
import traceback
from typing import List, Dict, Any, Optional

from .core.config import BenchmarkConfig, DEFAULT_CACHE_DIR
from .core.sandbox_manager import SandboxManager
from .agents.assistant import TravelAssistant
from .simulators.user_simulator import TravelUserSimulator
from .core.conversation import ConversationRunner
from .utils.util import (
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TURNS,
    load_datas_from_file,
    create_default_config,
    execute_tasks_with_concurrency,
    create_results_summary,
    print_results_summary,
)

from .tools import *
from .tools.tool_list import TOOL_NAMES

def run_multi_turn_conversation(
    config: BenchmarkConfig, 
    data: Dict, 
    trial_id: int,
    sandbox_manager: Optional[SandboxManager],
    max_turns: int = 10,
    debug: bool = False
) -> Dict[str, Any]:
    """Run a multi-turn conversation with a specific query.
    
    Args:
        config: Benchmark configuration
        data: Data dictionary containing query, user_profile, and context
        trial_id: Trial identifier
        sandbox_manager: Pre-initialized sandbox manager (optional)
        max_turns: Maximum conversation turns allowed
        debug: Enable debug logging
    """
    
    # Initialize sandbox manager if not provided
    if sandbox_manager is None and config.sandbox_config.enabled:
        sandbox_manager = SandboxManager(config)
        sandbox_manager.initialize_sandbox()
    
    # Extract data fields
    user_profile = data.get("user_profile", "")
    query = data.get("query", "")
    time = data.get("time", "")
    context = data.get("context", "")
    decomposed_query = data.get("intent", "")

    # Create assistant and user simulator
    assistant = TravelAssistant(
        config=config.assistant_config,
        time=time,
        context=context,
        multi_turn=True
    )
    user_simulator = TravelUserSimulator(
        config=config.user_simulator_config,
        user_profile=user_profile,
        query=query,
        time=time,
        context=context,
        decomposed_query=decomposed_query
    )
    
    # Create new ConversationRunner for each conversation
    runner = ConversationRunner(config)
    
    try:
        result = runner.run_multi_turn_conversation(assistant, user_simulator,max_turns)
        
        return {
            "trace_id": data.get("trace_id", ""),
            "mode": "multi_turn",
            "trial_id": trial_id,
            "query": query,
            "user_profile": user_profile,
            "context": context,
            "success": True,
            "conversation_id": result.conversation_id,
            "turns": result.total_turns,
            "duration": result.duration,
            "completion_reason": result.completion_reason,
            "tool_calls_count": result.tool_calls_count,
            "total_steps": result.total_steps,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "tool_errors": result.tool_errors,
            "messages": [msg.model_dump() for msg in result.messages]
        }
    except Exception as e:
        error_msg = str(e)
        if debug:
            print(f"❌ Multi-turn conversation failed with traceback:")
            traceback.print_exc()
            error_msg = traceback.format_exc()
        return {
            "trace_id": data.get("trace_id", ""),
            "mode": "multi_turn",
            "trial_id": trial_id,
            "query": query,
            "success": False,
            "error": error_msg
        }

def run_single_turn_conversation(
    config: BenchmarkConfig, 
    data: Dict, 
    trial_id: int,
    sandbox_manager: Optional[SandboxManager],
    max_steps: int = 10,
    debug: bool = False,
    unsolved: bool = False
) -> Dict[str, Any]:
    """Run a single-turn conversation with multi-step processing.
    
    Args:
        config: Benchmark configuration
        data: Data dictionary containing query and context
        trial_id: Trial identifier
        sandbox_manager: Pre-initialized sandbox manager (optional)
        max_steps: Maximum processing steps allowed
        debug: Enable debug logging
        unsolved: Whether this is an unsolved query
    """
    # Initialize sandbox manager if not provided
    if sandbox_manager is None and config.sandbox_config.enabled:
        sandbox_manager = SandboxManager(config)
        sandbox_manager.initialize_sandbox()
    
    # Extract data fields
    query = data.get("query", "")
    time = data.get("time", "")
    context = data.get("context", "")

    # Create assistant and user simulator
    assistant = TravelAssistant(
        config=config.assistant_config,
        time=time,
        context=context,
        multi_turn=False,
        unsolved=unsolved
    )
    
    # Create new ConversationRunner for each conversation
    runner = ConversationRunner(config)
    
    try:
        result = runner.run_single_turn_conversation(
            assistant=assistant,
            user_message=query,
            max_steps=max_steps
        )
        
        
        return {
            "trace_id": data.get("trace_id", ""),
            "mode": "single_turn",
            "trial_id": trial_id,
            "query": query,
            "context": context,
            "success": True,
            "conversation_id": result.conversation_id,
            "turns": result.total_turns,
            "duration": result.duration,
            "completion_reason": result.completion_reason,
            "tool_calls_count": result.tool_calls_count,
            "total_steps": result.total_steps,
            "cache_hits": result.cache_hits,
            "cache_misses": result.cache_misses,
            "tool_errors": result.tool_errors,
            "messages": [msg.model_dump() for msg in result.messages]
        }
    except Exception as e:
        error_msg = str(e)
        if debug:
            print(f"❌ Single-turn conversation failed with traceback:")
            traceback.print_exc()
            error_msg = traceback.format_exc()
        return {
            "trace_id": data.get("trace_id", ""),
            "mode": "single_turn",
            "trial_id": trial_id,
            "query": query,
            "success": False,
            "error": error_msg
        }

def _create_task_executor(
    config: BenchmarkConfig,
    sandbox_manager: Optional[SandboxManager],
    debug: bool = False
):
    """Create a task executor function for use with execute_tasks_with_concurrency.
    
    Args:
        config: Benchmark configuration
        sandbox_manager: Initialized sandbox manager (can be None if sandbox disabled)
        debug: Enable debug logging
        
    Returns:
        Callable that executes a single task
    """
    def executor(mode: str, data: Dict[str, Any], trial: int) -> Dict[str, Any]:
        if mode == "multi_turn":
            return run_multi_turn_conversation(
                config, data, trial, sandbox_manager, DEFAULT_MAX_TURNS, debug
            )
        elif mode == "single_turn":
            return run_single_turn_conversation(
                config, data, trial, sandbox_manager, DEFAULT_MAX_STEPS, debug
            )
        else:  # unsolved
            return run_single_turn_conversation(
                config, data, trial, sandbox_manager, DEFAULT_MAX_STEPS, debug, unsolved=True
            )
    
    return executor



def run_conversations(
    datas: List[Dict[str, Any]],
    mode: str = "multi_turn",
    agent_llm: str = "gpt-4",
    user_llm: str = "gpt-3.5-turbo",
    tool_llm: str = "gpt-4",
    sandbox_mode: str = "isolated",
    max_concurrency: int = 1,
    num_trials: int = 1,
    agent_llm_args: Dict[str, Any] = None,
    user_llm_args: Dict[str, Any] = None,
    tool_llm_args: Dict[str, Any] = None,
    output_file: str = "results.json",
    debug: bool = False
) -> List[Dict[str, Any]]:
    """Unified entry point for running conversations in both single-turn and multi-turn modes.
    
    Args:
        datas: List of query data dictionaries
        mode: Conversation mode ("multi_turn", "single_turn", or "unsolved")
        agent_llm: Agent/assistant model name
        user_llm: User simulator model name
        tool_llm: Tool simulation model name
        sandbox_mode: Sandbox execution mode
        max_concurrency: Maximum concurrent conversations
        num_trials: Number of trials per query
        agent_llm_args: Additional arguments for agent model
        user_llm_args: Additional arguments for user model
        tool_llm_args: Additional arguments for tool model
        output_file: Output file path for results
        debug: Enable debug mode
        
    Returns:
        List of conversation results
    """
    
    # Validate input
    if not datas:
        raise ValueError("datas list cannot be empty")
    
    if mode not in ["multi_turn", "single_turn", "unsolved"]:
        raise ValueError(f"Invalid mode: {mode}. Must be 'multi_turn', 'single_turn', or 'unsolved'")
    
    # Setup debug logging
    if debug:
        import logging
        logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
        print("🔍 Debug mode enabled - detailed logging activated")
    
    # Create configuration
    config = create_default_config(
        agent_llm=agent_llm,
        user_llm=user_llm,
        tool_llm=tool_llm,
        sandbox_mode=sandbox_mode,
        cache_dir=DEFAULT_CACHE_DIR
    )
    
    # Update model parameters in config
    if agent_llm_args:
        for key, value in agent_llm_args.items():
            setattr(config.assistant_config, key, value)
    
    if user_llm_args:
        for key, value in user_llm_args.items():
            setattr(config.user_simulator_config, key, value)

    if tool_llm_args:
        for key, value in tool_llm_args.items():
            setattr(config.tool_simulator_config, key, value)
    
    # Initialize shared components
    print("🔧 Initializing shared components...")
    
    # Initialize sandbox manager if enabled
    sandbox_manager = None
    if config.sandbox_config.enabled:
        sandbox_manager = SandboxManager(config)
        sandbox_manager.initialize_sandbox()
        print("✅ Sandbox initialized")
    
    print("✅ Configuration created")
    
    # Log execution configuration
    print(f"🚀 Running {len(datas)} datas in {mode} mode")
    print(f"📊 Config: agent={agent_llm}, user={user_llm}, sandbox={sandbox_mode}")
    print(f"🔄 Concurrency: {max_concurrency}")
    print(f"💾 Cache: {DEFAULT_CACHE_DIR}")
    
    # Build task list
    tasks = []
    for data in datas:
        for trial in range(num_trials):
            tasks.append((mode, data, trial))
    
    print(f"🚀 Running {len(tasks)} {mode} conversations ({len(datas)} datas × {num_trials} trials)")
    
    # Execute tasks using modular executor
    task_executor = _create_task_executor(config, sandbox_manager, debug)
    results = execute_tasks_with_concurrency(tasks, task_executor, max_concurrency, debug)
    
    # Create configuration info for summary
    config_info = {
        "agent_llm": agent_llm,
        "user_llm": user_llm,
        "tool_llm": tool_llm,
        "sandbox_mode": sandbox_mode,
        "cache_dir": DEFAULT_CACHE_DIR,
        "max_concurrency": max_concurrency,
        "num_trials": num_trials,
        "mode": mode
    }
    
    # Generate and save results
    summary = create_results_summary(results, datas, mode, config_info)
    
    output_data = {
        "summary": summary,
        "results": results
    }
    
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    # Print summary
    print_results_summary(summary, output_file)
    
    return results

def get_available_tools_for_openai():
    """Get all tools in OpenAI function calling format, filtered by TOOL_NAMES.
    
    Returns:
        List[Dict]: Tools formatted for OpenAI function calling
    """
    # Tools are already imported via direct imports
    # Get all registered tools in OpenAI format, but filter by TOOL_NAMES
    from .core.tools import sandbox_tool_registry
    # Get all registered tools and filter by TOOL_NAMES
    all_tools = sandbox_tool_registry.to_openai_format()
    return [
        tool for tool in all_tools 
        if tool.get("function", {}).get("name") in TOOL_NAMES
    ]


def main():
    # Tools are already imported via direct imports
    print(f"🚀 Travel Benchmark Framework")
    print(f"📦 Available tools: {', '.join(TOOL_NAMES)}")

    from .core.tools import sandbox_tool_registry
    all_tools = sandbox_tool_registry.get_tools()
    # Filter by TOOL_NAMES
    filtered_tools = [tool for tool in all_tools if tool.name in TOOL_NAMES]
    print(f"🛠️  Available tools: {len(filtered_tools)} (filtered from {len(all_tools)} total)")
    
    parser = argparse.ArgumentParser(
        description="Travel Benchmark Framework - Multi-turn Conversation Testing",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Main run command
    run_parser = subparsers.add_parser("run", help="Run conversations")
    
    # Query input options - simplified
    query_group = run_parser.add_mutually_exclusive_group(required=True)
    query_group.add_argument("--query", type=str, help="Single query to test")
    query_group.add_argument("--file", type=str, help="File containing queries (supports .txt, .json, .jsonl formats)")
    
    # Mode selection
    run_parser.add_argument("--mode", choices=["multi_turn", "single_turn", "unsolved"], default="multi_turn", 
                          help="Conversation mode (default: multi-turn)")
    # Debug mode options
    run_parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed logging")
    
    # Model configuration
    run_parser.add_argument("--agent-llm", default="gpt-51-1113-global", help="Agent/assistant model (default: gpt-4)")
    run_parser.add_argument("--user-llm", default="gpt-51-1113-global", help="User simulator model (default: gpt-3.5-turbo)")  
    run_parser.add_argument("--tool-llm", default="gpt-51-1113-global", help="Tool simulation model (default: gpt-4)")
    run_parser.add_argument("--agent-llm-args", type=str, help='Agent model args as JSON, e.g. \'{"max_tokens": 2048, "temperature": 0.0}\'')
    run_parser.add_argument("--user-llm-args", type=str, help='User model args as JSON, e.g. \'{"max_tokens": 1024, "temperature": 0.7}\'')
    run_parser.add_argument("--tool-llm-args", type=str, help='Tool model args as JSON, e.g. \'{"max_tokens": 1024, "temperature": 0.7}\'')
    
    # sandbox
    run_parser.add_argument("--sandbox-mode", default="isolated", help="Sandbox mode (default: isolated)")
    
    # Execution options
    run_parser.add_argument("--num-trials", type=int, default=1, help="Number of trials per query (default: 1)")
    run_parser.add_argument("--max-concurrency", type=int, default=1, help="Maximum concurrent conversations (default: 1)")
    run_parser.add_argument("--output", default="results.json", help="Output file path (default: results.json)")
    
    # Utility commands
    tools_parser = subparsers.add_parser("tools", help="List available tools")
    status_parser = subparsers.add_parser("status", help="Show cache status")
    
    args = parser.parse_args()
    
    if args.command == "run":
        # Load datas from input sources
        datas = []
        
        if args.query:
            datas = [{"query": args.query}]
        elif args.file:
            datas = load_datas_from_file(args.file)
            if not datas:
                print(f"❌ No datas loaded from file: {args.file}")
                sys.exit(1)
        else:
            print("❌ Either --query or --file must be provided")
            sys.exit(1)
        
        # Parse LLM arguments
        agent_llm_args = None
        user_llm_args = None
        
        if args.agent_llm_args:
            try:
                agent_llm_args = json.loads(args.agent_llm_args)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid agent-llm-args JSON: {e}")
                sys.exit(1)
        
        if args.user_llm_args:
            try:
                user_llm_args = json.loads(args.user_llm_args)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid user-llm-args JSON: {e}")
                sys.exit(1)

        if args.tool_llm_args:
            try:
                tool_llm_args = json.loads(args.tool_llm_args)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid tool-llm-args JSON: {e}")
                sys.exit(1)
        
        # Run conversations with unified interface
        try:
            run_conversations(
                datas=datas,
                mode=args.mode,
                agent_llm=args.agent_llm,
                user_llm=args.user_llm,
                tool_llm=args.tool_llm,
                sandbox_mode=args.sandbox_mode,
                max_concurrency=args.max_concurrency,
                num_trials=args.num_trials,
                agent_llm_args=agent_llm_args,
                user_llm_args=user_llm_args,
                tool_llm_args=tool_llm_args,
                output_file=args.output,
                debug=args.debug
            )
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
            sys.exit(1)
        except Exception as e:
            if args.debug:
                print(f"❌ Error with traceback:")
                traceback.print_exc()
            else:
                print(f"❌ Error: {e}")
            sys.exit(1)
            
    elif args.command == "tools":
        # Get tools in OpenAI format for display, filtered by TOOL_NAMES
        from .core.tools import sandbox_tool_registry
        all_tools_openai = sandbox_tool_registry.to_openai_format()
        tools_openai_format = [
            tool for tool in all_tools_openai
            if tool.get("function", {}).get("name") in TOOL_NAMES
        ]
        
        print("🔧 Available tools:")
        print(f"📦 Tool modules: {', '.join(TOOL_NAMES)}")
        print(f"🛠️  Total tools: {len(tools_openai_format)}")
        
        for tool_spec in tools_openai_format:
            tool_func = tool_spec["function"]
            print(f"  - {tool_func['name']}: {tool_func['description']}")
        
        # Show OpenAI API format example
        print("\n📋 OpenAI Function Calling Format:")
        print("   Tools are ready for use with OpenAI-style function calling")
        if tools_openai_format:
            print(f"   Example tool spec: {json.dumps(tools_openai_format[0], indent=4, ensure_ascii=False)}")
    elif args.command == "status":
        # Show cache status
        cache_dir = DEFAULT_CACHE_DIR
        if not os.path.exists(cache_dir):
            print(f"📂 Cache directory {cache_dir} does not exist")
            return
            
        try:
            from .core.tools import get_cache_stats
            stats = get_cache_stats(cache_dir)
            
            if not stats:
                print(f"📭 No cache data found in {cache_dir}")
                return
                
            print(f"📊 Cache Status ({cache_dir}):")
            for tool_name, stat in stats.items():
                print(f"  - {tool_name}: {stat.get('cached_calls', 0)} cached, {stat.get('missed_calls', 0)} missed")
                
        except Exception as e:
            print(f"❌ Error reading cache: {e}")
            traceback.print_exc()
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
