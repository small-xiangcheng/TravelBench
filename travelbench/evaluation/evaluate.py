#!/usr/bin/env python3
"""
Trajectory evaluation script V2.
Supports multi-turn and single-turn evaluation modes.
- multi-turn: 4 dimensions (reasoning_planning, summarization_extraction, presentation, user_interaction)
- single-turn: 3 dimensions (reasoning_planning, summarization_extraction, presentation)
"""

import json
import argparse
import time
import os
from typing import List, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from openai import OpenAI

from .prompt import (
    SYSTEM_PROMPT,
    PROMPT_SINGLE_TURN_REASONING,
    PROMPT_MULTI_TURN_REASONING,
    TOOLS_SCHEMAS,
    META_JUDGE_SYSTEM_PROMPT,
    META_JUDGE_PROMPT
)
from ..utils.eval_util import (
    PromptBuilder,
    ResultParser,
    DataLoader,
    StatisticsCalculator,
)

# ==================== Configuration ====================
class EvaluationConfig:
    """Evaluation configuration."""
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str = "gemini-3-flash-preview",
        temperature: float = 0,
        max_tokens: int = 16384,
        max_retries: int = 5,
        mode: str = "single-turn",  # single-turn or multi-turn
        enable_meta_judge: bool = False,  # whether to enable meta-judge
        meta_judge_model: Optional[str] = None,  # model for meta-judge, uses main model if None
        meta_judge_api_key: Optional[str] = None,  # api_key for meta-judge, uses main api_key if None
        meta_judge_base_url: Optional[str] = None  # base_url for meta-judge, uses main base_url if None
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.mode = mode
        self.enable_meta_judge = enable_meta_judge
        # Use main model if meta_judge_model is not specified
        self.meta_judge_model = meta_judge_model if meta_judge_model else model
        # Use main api_key if meta_judge_api_key is not specified
        self.meta_judge_api_key = meta_judge_api_key if meta_judge_api_key else api_key
        # Use main base_url if meta_judge_base_url is not specified
        self.meta_judge_base_url = meta_judge_base_url if meta_judge_base_url else base_url
        # Determine dimensions based on mode
        if mode == "multi-turn":
            self.dimensions = ["reasoning_planning", "summarization_extraction", "presentation", "user_interaction"]
        elif mode == "single-turn":
            self.dimensions = ["reasoning_planning", "summarization_extraction", "presentation"]
        else:
            raise ValueError(f"Unsupported mode: {mode}. Only 'single-turn' or 'multi-turn' are supported.")

# ==================== API Client ====================
class LLMEvaluator:
    """LLM-based evaluator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.client = OpenAI(api_key=config.api_key, base_url=config.base_url)
        self.lock = Lock()
    
    def custom_chat_completion(
        self, 
        messages: List[Dict], 
        temperature: float = 0,
        max_tokens: int = 8192, 
        top_p: float = 0.95,
        max_retries: int = 10, 
        system_prompt: Optional[str] = None
    ) -> Optional[str]:
        """Custom chat completion function."""
        attempt = 0
        if system_prompt:
            final_messages = [{"role": "system", "content": system_prompt}] + messages
        else:
            final_messages = messages
            
        while attempt < max_retries:
            try:
                response = self.client.chat.completions.create(
                    messages=final_messages,
                    model=self.config.model,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                )
                content = response.choices[0].message.content
                if content is not None:
                    return content
                else:
                    with self.lock:
                        print(f"[Warning] Attempt {attempt + 1}: Response content is None")
                    
            except Exception as e:
                error_name = type(e).__name__
                with self.lock:
                    print(f"[Warning] Attempt {attempt + 1} failed: {error_name}: {e}")
                
                # APIConnectionError cannot be resolved by retrying, raise directly
                if error_name == "APIConnectionError":
                    with self.lock:
                        print("[Error] APIConnectionError detected. Connection error cannot be resolved by retrying.")
                    raise
            
            attempt += 1
            time.sleep(1)
            
        with self.lock:
            print("[Error] Maximum retry attempts reached. No content retrieved.")
        return None

# ==================== Evaluation Prompt Builder ====================
class EvalPromptBuilder(PromptBuilder):
    """Prompt builder with evaluation-specific templates."""
    
    TEMPLATES = {
        "single-turn": PROMPT_SINGLE_TURN_REASONING,
        "multi-turn": PROMPT_MULTI_TURN_REASONING
    }
    
    def get_prompt_template(self, mode: str) -> str:
        """Get prompt template based on mode."""
        if mode not in self.TEMPLATES:
            raise ValueError(f"Unsupported mode: {mode}")
        return self.TEMPLATES[mode]
    
    def build_evaluation_prompt(self, trajectory: Dict, prompt_template: str) -> str:
        """Build evaluation prompt with TOOLS_SCHEMAS."""
        # Convert TOOLS_SCHEMAS to JSON string for template formatting
        tools_schemas_str = json.dumps(TOOLS_SCHEMAS, ensure_ascii=False, indent=2)
        return super().build_evaluation_prompt(trajectory, prompt_template, tools_schemas_str)


# ==================== Evaluation Executor ====================
class TrajectoryEvaluator:
    """Trajectory evaluator."""
    
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.llm_evaluator = LLMEvaluator(config)
        self.prompt_builder = EvalPromptBuilder()
        self.result_parser = ResultParser()
        # Create meta-judge client at initialization (if enabled)
        if config.enable_meta_judge:
            self.meta_judge_client = OpenAI(
                api_key=config.meta_judge_api_key, 
                base_url=config.meta_judge_base_url
            )
        else:
            self.meta_judge_client = None
    
    def meta_judge_evaluation(
        self,
        trajectory: Dict,
        first_evaluation: Dict
    ) -> Optional[Dict]:
        """Execute meta-judge evaluation."""
        # Build meta-judge prompt
        context = trajectory.get("context", "")
        query = trajectory.get("query", "")
        messages = trajectory.get("messages", [])
        conversation_history = self.prompt_builder.format_conversation_history(messages)
        
        # Format first evaluation scores and reasoning
        scores = first_evaluation.get("scores", {})
        parsed_result = first_evaluation.get("parsed_result", {})
        
        # Build score information
        score_lines = []
        for dimension in self.config.dimensions:
            score_key = f"{dimension}_score"
            score = scores.get(score_key, "N/A")
            score_lines.append(f"- {dimension}: {score}")
        score_lines.append(f"- average: {scores.get('average_score', 'N/A')}")
        evaluation_scores = "\n".join(score_lines)
        
        # Build reasoning information
        reasoning_lines = []
        for dimension in self.config.dimensions:
            dim_result = parsed_result.get(dimension, {})
            reasoning = dim_result.get("reasoning", "No reasoning content")
            rating = dim_result.get("rating", "No rating")
            reasoning_lines.append(f"### {dimension}")
            reasoning_lines.append(f"**Rating**: {rating}")
            reasoning_lines.append(f"**Reasoning**: {reasoning}")
            reasoning_lines.append("")
        evaluation_reasoning = "\n".join(reasoning_lines)
        
        # Build complete meta-judge prompt
        meta_prompt = META_JUDGE_PROMPT.format(
            CONTEXT_INFO=context,
            QUESTION_CONTENT=query,
            INTENDED_TOOL=TOOLS_SCHEMAS,
            CONVERSATION_HISTORY=conversation_history,
            EVALUATION_SCORES=evaluation_scores,
            EVALUATION_REASONING=evaluation_reasoning
        )
        
        # Call LLM for meta-judge evaluation, retry on XML parse failure (max 10 times)
        messages_list = [{"role": "user", "content": meta_prompt}]
        
        parsed_meta = None
        meta_score = None
        response = None
        max_parse_retries = 10
        
        for attempt in range(max_parse_retries):
            # Call LLM with meta_judge_model
            try:
                final_messages = [{"role": "system", "content": META_JUDGE_SYSTEM_PROMPT}] + messages_list
                meta_response = self.meta_judge_client.chat.completions.create(
                    messages=final_messages,
                    model=self.config.meta_judge_model,
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    top_p=0.95,
                )
                response = meta_response.choices[0].message.content
            except Exception as e:
                error_name = type(e).__name__
                if error_name == "APIConnectionError":
                    return None
                time.sleep(1)
                continue
            
            if not response:
                continue
            
            # Try to parse XML
            parsed_meta = self.result_parser.parse_meta_judge_response(response)
            
            if parsed_meta:
                # Parse successful, convert rating to score
                rating = parsed_meta.get("rating", "")
                meta_score = self.result_parser.rating_to_score(rating)
                
                if meta_score is not None:
                    break
            
            # Wait 1 second before retry if not last attempt
            if attempt < max_parse_retries - 1:
                time.sleep(1)
        
        # Return None if all attempts failed
        if not parsed_meta or meta_score is None:
            return None
        
        return {
            "raw_response": response,
            "parsed_result": parsed_meta,
            "meta_score": meta_score,
            "meta_rating": rating
        }
    
    def evaluate_single_trajectory(
        self, 
        trajectory: Dict, 
        index: int
    ) -> Optional[Dict]:
        """Evaluate single trajectory."""
        # Get corresponding prompt template
        prompt_template = self.prompt_builder.get_prompt_template(
            self.config.mode
        )
        
        # Build evaluation prompt
        evaluation_prompt = self.prompt_builder.build_evaluation_prompt(
            trajectory, 
            prompt_template
        )
        
        # Call LLM for evaluation, retry on XML parse failure (max 10 times)
        start_time = time.time()
        messages = [{"role": "user", "content": evaluation_prompt}]
        
        parsed_result = None
        scores = None
        response = None
        max_parse_retries = 10
        
        for attempt in range(max_parse_retries):
            try:
                # Call LLM
                if attempt > 6:
                    response = self.llm_evaluator.custom_chat_completion(
                        messages=messages,
                        temperature=0.7,
                        max_tokens=self.config.max_tokens,
                        system_prompt=SYSTEM_PROMPT
                    )   
                else: 
                    response = self.llm_evaluator.custom_chat_completion(
                        messages=messages,
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        system_prompt=SYSTEM_PROMPT
                    )
                
                if not response:
                    continue
            except Exception as e:
                error_name = type(e).__name__
                if error_name == "APIConnectionError":
                    return None
                continue
            
            # Try to parse XML
            parsed_result = self.result_parser.parse_xml_response(response, self.config.dimensions)
            
            if parsed_result:
                # Parse successful, extract scores
                scores = self.result_parser.extract_scores(parsed_result, self.config.dimensions)
                if scores and scores.get("average_score") is not None:
                    break
            
            # Wait 1 second before retry if not last attempt
            if attempt < max_parse_retries - 1:
                time.sleep(1)
        
        elapsed_time = time.time() - start_time
        
        # Return None if all attempts failed
        if not parsed_result or not scores or scores.get("average_score") is None:
            with self.llm_evaluator.lock:
                print(f"[{index}] Evaluation failed: unable to get valid score after {max_parse_retries} attempts")
            return None
        
        # Calculate tool error rate
        tool_calls_count = trajectory.get("tool_calls_count", 0)
        tool_errors = trajectory.get("tool_errors", 0)
        tool_error_rate = tool_errors / tool_calls_count if tool_calls_count > 0 else 0
        
        # Normalize scores to 0-100 range per paper formula: S = (avg - 1) / 4 * 100
        normalized_scores = {}
        for dimension in self.config.dimensions:
            score_key = f"{dimension}_score"
            if scores.get(score_key) is not None:
                normalized_scores[score_key] = (scores[score_key] - 1) / 4 * 100
            else:
                normalized_scores[score_key] = None
        normalized_scores["average_score"] = (scores["average_score"] - 1) / 4 * 100
        
        # Apply tool penalty: w_tool = 1 - error_rate
        # penalized_scores are in 0-100 range
        penalty_factor = 1 - tool_error_rate
        penalized_scores = {}
        for dimension in self.config.dimensions:
            score_key = f"{dimension}_score"
            if normalized_scores.get(score_key) is not None:
                penalized_scores[score_key] = normalized_scores[score_key] * penalty_factor
            else:
                penalized_scores[score_key] = None
        penalized_scores["average_score"] = normalized_scores["average_score"] * penalty_factor
        
        # Execute meta-judge evaluation (if enabled)
        meta_judge_result = None
        final_scores = penalized_scores.copy()
        meta_score = None
        
        if self.config.enable_meta_judge and self.meta_judge_client is not None:
            meta_judge_result = self.meta_judge_evaluation(trajectory, {
                "scores": scores,
                "parsed_result": parsed_result
            })
            
            if meta_judge_result and meta_judge_result.get("meta_score") is not None:
                meta_score = meta_judge_result["meta_score"]
                meta_weight = meta_score / 5.0  # w_meta = s / 5
                
                # Apply meta-judge weight: S_pen = S * w_tool * w_meta
                final_scores = {}
                for dimension in self.config.dimensions:
                    score_key = f"{dimension}_score"
                    if penalized_scores.get(score_key) is not None:
                        final_scores[score_key] = penalized_scores[score_key] * meta_weight
                    else:
                        final_scores[score_key] = None
                final_scores["average_score"] = penalized_scores["average_score"] * meta_weight
        
        # Build complete result
        result = {
            "index": index,
            "conversation_id": trajectory.get("conversation_id"),
            "query": trajectory.get("query"),
            "mode": self.config.mode,
            "original_trajectory": trajectory,
            "evaluation": {
                "raw_response": response,
                "parsed_result": parsed_result,
                "scores": scores,
                "normalized_scores": normalized_scores,
                "tool_calls_count": tool_calls_count,
                "tool_errors": tool_errors,
                "tool_error_rate": tool_error_rate,
                "penalized_scores": penalized_scores,
                "meta_judge_result": meta_judge_result,
                "final_scores": final_scores,
                "evaluation_time": elapsed_time
            }
        }
        
        # Concise log output (use lock to prevent interleaved output in multi-threading)
        with self.llm_evaluator.lock:
            if self.config.enable_meta_judge and meta_score is not None:
                print(f"[{index}] Evaluation complete, raw avg: {scores['average_score']:.2f}, tool error rate: {tool_error_rate:.2f}, meta-judge: {meta_score}, final avg (0-100): {final_scores['average_score']:.2f}")
            else:
                print(f"[{index}] Evaluation complete, raw avg: {scores['average_score']:.2f}, tool error rate: {tool_error_rate:.2f}, final avg (0-100): {final_scores['average_score']:.2f}")
        
        return result
    
    def evaluate_batch(
        self,
        trajectories: List[Dict],
        num_workers: int = 4
    ) -> List[Dict]:
        """Batch evaluation."""
        results = []
        
        if num_workers == 1:
            for i, trajectory in enumerate(trajectories):
                result = self.evaluate_single_trajectory(trajectory, i)
                if result is not None:
                    results.append(result)
        else:
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                future_to_index = {}
                
                # Submit tasks in batches to avoid API QPS throttling
                # First batch: submit 30 tasks
                batch_size = 30
                first_batch = trajectories[:batch_size]
                for i, traj in enumerate(first_batch):
                    future = executor.submit(self.evaluate_single_trajectory, traj, i)
                    future_to_index[future] = i
                
                time.sleep(10)
                
                # Second batch: submit another 30 tasks
                second_batch = trajectories[batch_size:batch_size*2]
                for i, traj in enumerate(second_batch, start=batch_size):
                    future = executor.submit(self.evaluate_single_trajectory, traj, i)
                    future_to_index[future] = i

                time.sleep(10)

                # Remaining tasks: submit with normal concurrency
                remaining_batch = trajectories[batch_size*2:]
                for i, traj in enumerate(remaining_batch, start=batch_size*2):
                    future = executor.submit(self.evaluate_single_trajectory, traj, i)
                    future_to_index[future] = i
                
                # Collect all results
                for future in as_completed(future_to_index):
                    try:
                        result = future.result()
                        if result is not None:
                            results.append(result)
                    except Exception as e:
                        index = future_to_index[future]
                        print(f"[Error] Error evaluating trajectory {index}: {type(e).__name__}: {e}")
                        import traceback
                        traceback.print_exc()
        
        results.sort(key=lambda x: x["index"])
        return results

# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description="Trajectory Evaluation Tool - Supports multi-turn and single-turn modes")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    parser.add_argument("--api_key", help="OpenAI API Key")
    parser.add_argument("--base_url", default="your-openai-api-base-here", help="API Base URL")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model for evaluation")
    parser.add_argument("--mode", "-m", choices=["single-turn", "multi-turn"], default="single-turn", 
                        help="Evaluation mode: single-turn (3 dimensions) or multi-turn (4 dimensions)")
    parser.add_argument("--max-concurrency", type=int, default=4, help="Maximum concurrency")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature parameter")
    parser.add_argument("--max_tokens", type=int, default=16384, help="Maximum tokens")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum evaluation samples")
    parser.add_argument("--enable_meta_judge", action="store_true", help="Enable meta-judge evaluation")
    parser.add_argument("--meta_judge_model", default=None, help="Model for meta-judge, uses main model if not specified")
    parser.add_argument("--meta_judge_api_key", default=None, help="API Key for meta-judge, uses main API Key if not specified")
    parser.add_argument("--meta_judge_base_url", default=None, help="API Base URL for meta-judge, uses main Base URL if not specified")
    
    args = parser.parse_args()
    
    # Command line args take priority, otherwise get API config from environment variables
    if not args.api_key:
        args.api_key = os.getenv("OPENAI_API_KEY")
    # base_url has default value, check if it's the default
    if args.base_url == "your-openai-api-base-here":
        env_base_url = os.getenv("OPENAI_API_BASE")
        if env_base_url:
            args.base_url = env_base_url
    
    # Validate required API configuration
    if not args.api_key:
        print("[Error] API Key not configured. Set via --api_key parameter or OPENAI_API_KEY environment variable")
        return
    
    config = EvaluationConfig(
        api_key=args.api_key,
        base_url=args.base_url,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        mode=args.mode,
        enable_meta_judge=args.enable_meta_judge,
        meta_judge_model=args.meta_judge_model,
        meta_judge_api_key=args.meta_judge_api_key,
        meta_judge_base_url=args.meta_judge_base_url
    )
    
    print("=" * 60)
    print("Trajectory Evaluation Tool")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print(f"Model: {args.model}")
    print(f"Evaluation mode: {args.mode}")
    print(f"Evaluation dimensions: {', '.join(config.dimensions)}")
    print(f"Concurrency: {args.max_concurrency}")
    print(f"Meta-judge: {'enabled' if args.enable_meta_judge else 'disabled'}")
    if args.enable_meta_judge:
        print(f"Meta-judge model: {config.meta_judge_model}")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    trajectories = DataLoader.load_trajectories(args.input)
    
    if args.max_samples:
        trajectories = trajectories[:args.max_samples]
        print(f"Limited to {args.max_samples} samples")
    
    # Evaluate
    print(f"\n[2/4] Starting evaluation of {len(trajectories)} trajectories...")
    evaluator = TrajectoryEvaluator(config)
    start_time = time.time()
    results = evaluator.evaluate_batch(trajectories, num_workers=args.max_concurrency)
    total_time = time.time() - start_time
    
    print(f"\nEvaluation complete, total time: {total_time:.2f}s")
    print(f"Average per trajectory: {total_time/len(trajectories):.2f}s")
    
    # Calculate statistics
    print("\n[3/4] Calculating statistics...")
    statistics = StatisticsCalculator.calculate_statistics(results, config.dimensions)
    
    print("\nStatistics:")
    print(f"  Evaluation mode: {statistics['mode']}")
    print(f"  Evaluation dimensions: {', '.join(statistics['dimensions'])}")
    print(f"  Total trajectories: {statistics['total_trajectories']}")
    print(f"  Valid evaluations: {statistics['valid_evaluations']}")
    
    print(f"\nRaw average scores (1-5):")
    for dimension, score in statistics['average_scores'].items():
        print(f"  {dimension}: {score:.2f}")
    
    print(f"\nRaw average scores (0-100):")
    for dimension, score in statistics['average_scores_100'].items():
        print(f"  {dimension}: {score:.2f}")
    
    print(f"\nTool error statistics:")
    tool_stats = statistics['tool_error_statistics']
    print(f"  Average tool error rate: {tool_stats['average_tool_error_rate']:.2%}")
    print(f"  Samples with errors: {tool_stats['total_samples_with_errors']}")
    
    print(f"\nPenalized average scores (0-100, tool penalty only):")
    for dimension, score in statistics['penalized_average_scores_100'].items():
        print(f"  {dimension}: {score:.2f}")
    
    if statistics['meta_judge_statistics'].get('enabled'):
        print(f"\nMeta-judge statistics:")
        meta_stats = statistics['meta_judge_statistics']
        print(f"  Total evaluations: {meta_stats['total_meta_evaluations']}")
        print(f"  Average meta score: {meta_stats['average_meta_score']:.2f}")
    
    print(f"\nFinal average scores (0-100):")
    for dimension, score in statistics['final_average_scores_100'].items():
        print(f"  {dimension}: {score:.2f}")
    
    # Save results
    print("\n[4/4] Saving results...")
    DataLoader.save_results(results, args.output)
    
    # Save statistics summary
    summary_file = args.output.replace('.json', '_summary.json')
    if summary_file == args.output:
        summary_file = args.output + '_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"Statistics summary saved to: {summary_file}")
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
