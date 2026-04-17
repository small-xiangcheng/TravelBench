"""
Utility functions and classes for trajectory evaluation.

This module provides reusable components for the evaluation pipeline:
- XML parsing and fixing utilities
- Prompt building utilities
- Result parsing utilities
- Data loading utilities
- Statistics calculation utilities
"""

import json
import re
from typing import List, Dict, Optional
import xml.etree.ElementTree as ET

__all__ = [
    'fix_xml_tags',
    'PromptBuilder',
    'ResultParser',
    'DataLoader',
    'StatisticsCalculator',
]


# ==================== XML Utilities ====================

def fix_xml_tags(xml_content: str, dimensions: List[str]) -> str:
    """
    Attempt to repair common XML tag issues using stack-based nested structure processing.
    
    Args:
        xml_content: XML content string to fix.
        dimensions: List of dimension names based on mode.
        
    Returns:
        A best-effort, repaired XML content string. The result may still be invalid for severely
    malformed input.
    """
    # Extract all tags
    tag_pattern = r'</?[^>]+>'
    matches = list(re.finditer(tag_pattern, xml_content))
    
    if not matches:
        return xml_content
    
    # Use stack to track open tags
    tag_stack = []
    operations = []  # Record operations: ('insert', pos, tag) or ('delete', start, end)
    
    # Define expected child tag order within each dimension
    dimension_children = ["<reasoning>", "</reasoning>", "<rating>", "</rating>"]
    
    for i, match in enumerate(matches):
        tag = match.group()
        start_pos = match.start()
        end_pos = match.end()
        
        # Determine if it's an opening or closing tag
        if tag.startswith('</'):
            # Closing tag
            tag_name = tag[2:-1]
            
            # Check if stack top has corresponding opening tag
            if tag_stack and tag_stack[-1]['name'] == tag_name:
                # Match, pop from stack
                open_tag_info = tag_stack.pop()
                
                # If it's a dimension tag, check if its content is complete
                if tag_name in dimensions:
                    # Check if reasoning and rating exist within this dimension
                    # Find all tags between dimension opening and closing tags
                    inner_tags = []
                    for j in range(open_tag_info['match_index'] + 1, i):
                        inner_tag = matches[j].group()
                        inner_tags.append(inner_tag)
                    
                    # Check if </reasoning> is missing
                    if '<reasoning>' in inner_tags and '</reasoning>' not in inner_tags:
                        # Insert </reasoning> before current closing tag
                        operations.append(('insert', start_pos, '</reasoning>'))
                    
                    # Check if <rating> and </rating> are missing
                    if '</reasoning>' in inner_tags or '<reasoning>' in inner_tags:
                        if '<rating>' not in inner_tags:
                            operations.append(('insert', start_pos, '<rating>'))
                        if '</rating>' not in inner_tags:
                            operations.append(('insert', start_pos, '</rating>'))
            else:
                # No match, possibly missing opening tag or wrong tag order
                # Try to find matching opening tag in stack
                found = False
                for idx in range(len(tag_stack) - 1, -1, -1):
                    if tag_stack[idx]['name'] == tag_name:
                        # Found, need to close intermediate tags
                        # Close all intermediate tags
                        while len(tag_stack) > idx:
                            unclosed = tag_stack.pop()
                            close_tag = f"</{unclosed['name']}>"
                            operations.append(('insert', start_pos, close_tag))
                        found = True
                        break
        else:
            # Opening tag
            tag_name = tag[1:-1]
            tag_stack.append({
                'name': tag_name,
                'match_index': i,
                'start_pos': start_pos
            })
    
    # Check for unclosed tags
    if tag_stack:
        # Close all unclosed tags at document end
        for unclosed in reversed(tag_stack):
            close_tag = f"</{unclosed['name']}>"
            operations.append(('insert', len(xml_content), close_tag))
    
    # Execute operations from back to front to avoid position offset
    for operation in reversed(operations):
        if operation[0] == 'insert':
            _, pos, new_tag = operation
            xml_content = xml_content[:pos] + new_tag + xml_content[pos:]
        elif operation[0] == 'delete':
            _, start, end = operation
            xml_content = xml_content[:start] + xml_content[end:]
    
    return xml_content


# ==================== Prompt Builder ====================

class PromptBuilder:
    """Build evaluation prompts."""

    @staticmethod
    def format_conversation_history(messages: List[Dict]) -> str:
        """Format conversation history."""
        history = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls", [])
            
            if role == "system":
                continue
            elif role == "user":
                history.append(f"{role}: {content}")
            elif role == "assistant":
                if tool_calls:
                    for tool_call in tool_calls:
                        tool_name = tool_call.get("name", "unknown")
                        arguments = tool_call.get("arguments", {})
                        history.append(f"assistant tool_calls: {tool_name} ({json.dumps(arguments, ensure_ascii=False, indent=2)})")
                if content:
                    history.append(f"assistant: {content}")
            elif role == "tool":
                tool_name = msg.get("name", "unknown")
                history.append(f"tool_response: {tool_name} {content}")
        
        return "\n".join(history)
    
    @staticmethod
    def get_prompt_template(mode: str, templates: Optional[Dict[str, str]] = None) -> str:
        """Get prompt template based on mode.
        
        This method is designed to be overridden or used with templates parameter.
        When templates is None, it returns an empty string (subclass should override).
        
        Args:
            mode: Evaluation mode ('single-turn' or 'multi-turn')
            templates: Optional dictionary mapping mode to template string
            
        Returns:
            Prompt template string
        """
        if templates is None:
            # Return empty string when no templates provided
            # This allows subclasses to override this method
            raise NotImplementedError(
                "get_prompt_template requires templates parameter or subclass override"
            )
        if mode not in templates:
            raise ValueError(f"Unsupported mode: {mode}")
        return templates[mode]
    
    @staticmethod
    def build_evaluation_prompt(
        trajectory: Dict,
        prompt_template: str,
        tools_schemas: Optional[str] = None
    ) -> str:
        """Build evaluation prompt.
        
        Args:
            trajectory: Trajectory data containing messages, query, context
            prompt_template: Template string with placeholders
            tools_schemas: Optional tool schemas string (for compatibility)
            
        Returns:
            Formatted evaluation prompt
        """
        messages = trajectory.get("messages", [])
        query = trajectory.get("query", "")
        context = trajectory.get("context", "")
        
        conversation_history = PromptBuilder.format_conversation_history(messages)
        
        # Build format kwargs
        format_kwargs = {
            "CONTEXT_INFO": context,
            "QUESTION_CONTENT": query,
            "CONVERSATION_HISTORY": conversation_history
        }
        
        # Add tools_schemas if provided
        if tools_schemas is not None:
            format_kwargs["INTENDED_TOOL"] = tools_schemas
        
        prompt = prompt_template.format(**format_kwargs)
        return prompt


# ==================== Result Parser ====================

class ResultParser:
    """Parse evaluation results."""
    
    RATING_MAP = {
        "极差": 1,
        "较差": 2,
        "一般": 3,
        "较好": 4,
        "优秀": 5
    }
    
    @staticmethod
    def parse_xml_response(xml_content: str, dimensions: List[str]) -> Optional[Dict]:
        """Parse XML format evaluation results with dynamic dimensions."""
        try:
            xml_content = xml_content.strip()
            if xml_content.startswith("```xml"):
                xml_content = xml_content[6:]
            if xml_content.endswith("```"):
                xml_content = xml_content[:-3]
            xml_content = xml_content.strip()
            
            # Extract content within <response> tag
            response_match = re.search(r'<response>(.*?)</response>', xml_content, re.DOTALL)
            if response_match:
                xml_content = f"<response>{response_match.group(1)}</response>"
            
            orig_xml_content = xml_content
            # Try direct parsing
            try:
                root = ET.fromstring(xml_content)
            except ET.ParseError:
                # Parse failed, try to fix tags
                xml_content = fix_xml_tags(xml_content, dimensions)
                root = ET.fromstring(xml_content)
            
            result = {}
            for dimension in dimensions:
                result[dimension] = {
                    "reasoning": "",
                    "rating": ""
                }
            
            for dimension in dimensions:
                element = root.find(dimension)
                if element is not None:
                    reasoning_elem = element.find("reasoning")
                    rating_elem = element.find("rating")
                    
                    if reasoning_elem is not None:
                        result[dimension]["reasoning"] = reasoning_elem.text or ""
                    if rating_elem is not None:
                        result[dimension]["rating"] = (rating_elem.text or "").strip()
            
            return result
            
        except ET.ParseError as e:
            print(f"[Error] XML parsing failed: {type(e).__name__}: {e} Retrying...")
            return None
        except Exception as e:
            print(f"[Error] XML parsing error: {type(e).__name__}: {e} Retrying..." )
            return None
    
    @staticmethod
    def rating_to_score(rating: str) -> Optional[int]:
        """Convert text rating to numeric score."""
        return ResultParser.RATING_MAP.get(rating, None)
    
    @staticmethod
    def extract_scores(parsed_result: Optional[Dict], dimensions: List[str]) -> Dict:
        """Extract numeric scores from parsed results."""
        if not parsed_result:
            scores = {f"{dim}_score": None for dim in dimensions}
            scores["average_score"] = None
            return scores
        
        scores = {}
        score_values = []
        
        for dimension in dimensions:
            rating = parsed_result.get(dimension, {}).get("rating", "")
            score = ResultParser.rating_to_score(rating)
            scores[f"{dimension}_score"] = score
            if score is not None:
                score_values.append(score)
        
        if score_values:
            scores["average_score"] = sum(score_values) / len(score_values)
        else:
            scores["average_score"] = None
        
        return scores
    
    @staticmethod
    def parse_meta_judge_response(xml_content: str) -> Optional[Dict]:
        """Parse meta-judge XML response."""
        try:
            xml_content = xml_content.strip()
            if xml_content.startswith("```xml"):
                xml_content = xml_content[6:]
            if xml_content.endswith("```"):
                xml_content = xml_content[:-3]
            xml_content = xml_content.strip()
            
            # Extract content within <meta_evaluation> tag
            response_match = re.search(r'<meta_evaluation>(.*?)</meta_evaluation>', xml_content, re.DOTALL)
            if response_match:
                xml_content = f"<meta_evaluation>{response_match.group(1)}</meta_evaluation>"
            
            root = ET.fromstring(xml_content)
            
            result = {
                "reasoning": "",
                "rating": ""
            }
            
            reasoning_elem = root.find("reasoning")
            rating_elem = root.find("rating")
            
            if reasoning_elem is not None:
                result["reasoning"] = reasoning_elem.text or ""
            if rating_elem is not None:
                result["rating"] = (rating_elem.text or "").strip()
            
            return result
            
        except ET.ParseError as e:
            print(f"[Error] Meta-judge XML parsing failed: {type(e).__name__}: {e} Rertying...")
            return None
        except Exception as e:
            print(f"[Error] Meta-judge parsing error: {type(e).__name__}: {e} Retrying..." )
            import traceback
            traceback.print_exc()
            return None


# ==================== Data Loader ====================

class DataLoader:
    """Load and process data."""
    
    @staticmethod
    def load_trajectories(file_path: str) -> List[Dict]:
        """Load trajectory file."""
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if isinstance(data, list):
                trajectories = data
            elif isinstance(data, dict):
                if "results" in data:
                    trajectories = data["results"]
                else:
                    trajectories = [data]
            else:
                print(f"[Error] Unsupported JSON format")
                return []
                
        except json.JSONDecodeError as e:
            print(f"[Error] JSON file parsing failed: {type(e).__name__}: {e}")
            return []
        except Exception as e:
            print(f"[Error] File loading failed: {type(e).__name__}: {e}")
            return []
        
        print(f"Successfully loaded {len(trajectories)} trajectories")
        return trajectories
    
    @staticmethod
    def save_results(results: List[Dict], output_file: str):
        """Save evaluation results."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")


# ==================== Statistics Calculator ====================

class StatisticsCalculator:
    """Calculate statistics."""
    
    @staticmethod
    def calculate_statistics(results: List[Dict], dimensions: List[str]) -> Dict:
        """Calculate overall statistics."""
        total = len(results)
        valid_results = [r for r in results if r["evaluation"]["scores"] and r["evaluation"]["scores"].get("average_score") is not None]
        
        if not valid_results:
            return {
                "total_trajectories": total,
                "valid_evaluations": 0,
                "mode": results[0]["mode"] if results else "unknown",
                "dimensions": dimensions,
                "average_scores": {},
                "average_scores_100": {},
                "penalized_average_scores_100": {},
                "final_average_scores_100": {},
                "rating_distribution": {},
                "tool_error_statistics": {},
                "cache_statistics": {},
                "meta_judge_statistics": {}
            }
        
        # Collect dimension scores
        dimension_scores = {dim: [] for dim in dimensions}
        normalized_dimension_scores = {dim: [] for dim in dimensions}
        penalized_dimension_scores = {dim: [] for dim in dimensions}
        final_dimension_scores = {dim: [] for dim in dimensions}
        average_scores = []
        normalized_average_scores = []
        penalized_average_scores = []
        final_average_scores = []
        tool_error_rates = []
        meta_judge_scores = []
        
        # Collect cache hit rate data
        cache_hits_list = []
        cache_misses_list = []
        
        for result in valid_results:
            scores = result["evaluation"]["scores"]
            normalized_scores = result["evaluation"].get("normalized_scores")
            penalized_scores = result["evaluation"].get("penalized_scores")
            final_scores = result["evaluation"].get("final_scores")
            tool_error_rate = result["evaluation"].get("tool_error_rate", 0)
            meta_judge_result = result["evaluation"].get("meta_judge_result")
            
            # Raw scores (1-5 scale)
            for dimension in dimensions:
                score_key = f"{dimension}_score"
                if scores.get(score_key) is not None:
                    dimension_scores[dimension].append(scores[score_key])
            
            if scores.get("average_score") is not None:
                average_scores.append(scores["average_score"])
            
            # Normalized scores (0-100 scale, after normalization only)
            if normalized_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if normalized_scores.get(score_key) is not None:
                        normalized_dimension_scores[dimension].append(normalized_scores[score_key])
                
                if normalized_scores.get("average_score") is not None:
                    normalized_average_scores.append(normalized_scores["average_score"])
            
            # Penalized scores (0-100 scale, after normalization and tool penalty only)
            if penalized_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if penalized_scores.get(score_key) is not None:
                        penalized_dimension_scores[dimension].append(penalized_scores[score_key])
                
                if penalized_scores.get("average_score") is not None:
                    penalized_average_scores.append(penalized_scores["average_score"])
            
            # Final scores (0-100 scale, after normalization and all penalties including meta-judge)
            if final_scores:
                for dimension in dimensions:
                    score_key = f"{dimension}_score"
                    if final_scores.get(score_key) is not None:
                        final_dimension_scores[dimension].append(final_scores[score_key])
                
                if final_scores.get("average_score") is not None:
                    final_average_scores.append(final_scores["average_score"])
            
            tool_error_rates.append(tool_error_rate)
            
            # Collect cache hit rate data
            trajectory = result.get("original_trajectory", {})
            cache_hits = trajectory.get("cache_hits", 0)
            cache_misses = trajectory.get("cache_misses", 0)
            cache_hits_list.append(cache_hits)
            cache_misses_list.append(cache_misses)
            
            # Collect meta-judge scores
            if meta_judge_result and meta_judge_result.get("meta_score") is not None:
                meta_judge_scores.append(meta_judge_result["meta_score"])
        
        # Calculate raw average scores (1-5 scale)
        avg_scores = {}
        for dimension in dimensions:
            if dimension_scores[dimension]:
                avg = sum(dimension_scores[dimension]) / len(dimension_scores[dimension])
                avg_scores[dimension] = avg
            else:
                avg_scores[dimension] = 0
        
        avg_overall = sum(average_scores) / len(average_scores) if average_scores else 0
        avg_scores["overall_average"] = avg_overall
        
        # Calculate normalized average scores (0-100 scale)
        avg_scores_100 = {}
        for dimension in dimensions:
            if normalized_dimension_scores[dimension]:
                avg_scores_100[dimension] = sum(normalized_dimension_scores[dimension]) / len(normalized_dimension_scores[dimension])
            else:
                avg_scores_100[dimension] = 0
        
        normalized_avg_overall = sum(normalized_average_scores) / len(normalized_average_scores) if normalized_average_scores else 0
        avg_scores_100["overall_average"] = normalized_avg_overall
        
        # Calculate penalized average scores (0-100 scale, after normalization and tool penalty only)
        penalized_avg_scores_100 = {}
        for dimension in dimensions:
            if penalized_dimension_scores[dimension]:
                penalized_avg_scores_100[dimension] = sum(penalized_dimension_scores[dimension]) / len(penalized_dimension_scores[dimension])
            else:
                penalized_avg_scores_100[dimension] = 0
        
        penalized_avg_overall = sum(penalized_average_scores) / len(penalized_average_scores) if penalized_average_scores else 0
        penalized_avg_scores_100["overall_average"] = penalized_avg_overall
        
        # Calculate final average scores (0-100 scale, after normalization and all penalties including meta-judge)
        final_avg_scores_100 = {}
        for dimension in dimensions:
            if final_dimension_scores[dimension]:
                final_avg_scores_100[dimension] = sum(final_dimension_scores[dimension]) / len(final_dimension_scores[dimension])
            else:
                final_avg_scores_100[dimension] = 0
        
        final_avg_overall = sum(final_average_scores) / len(final_average_scores) if final_average_scores else 0
        final_dimension_100_scores = [final_avg_scores_100[dim] for dim in dimensions if dim in final_avg_scores_100]
        final_avg_scores_100["overall_average"] = sum(final_dimension_100_scores) / len(final_dimension_100_scores) if final_dimension_100_scores else 0
        
        # Tool error rate statistics
        avg_tool_error_rate = sum(tool_error_rates) / len(tool_error_rates) if tool_error_rates else 0
        
        # Cache hit rate statistics
        total_cache_hits = sum(cache_hits_list)
        total_cache_misses = sum(cache_misses_list)
        total_cache_requests = total_cache_hits + total_cache_misses
        cache_hit_rate = (total_cache_hits / total_cache_requests * 100) if total_cache_requests > 0 else 0
        
        cache_statistics = {
            "total_cache_hits": total_cache_hits,
            "total_cache_misses": total_cache_misses,
            "total_cache_requests": total_cache_requests,
            "cache_hit_rate_percentage": cache_hit_rate,
            "average_cache_hits_per_sample": total_cache_hits / len(valid_results) if valid_results else 0,
            "average_cache_misses_per_sample": total_cache_misses / len(valid_results) if valid_results else 0
        }
        
        # Meta-judge statistics
        meta_judge_statistics = {}
        if meta_judge_scores:
            meta_judge_statistics = {
                "enabled": True,
                "total_meta_evaluations": len(meta_judge_scores),
                "average_meta_score": sum(meta_judge_scores) / len(meta_judge_scores),
                "meta_score_distribution": {i: meta_judge_scores.count(i) for i in range(1, 6)}
            }
        else:
            meta_judge_statistics = {
                "enabled": False
            }
        
        # Rating distribution statistics
        rating_distribution = {}
        for dimension in dimensions:
            rating_distribution[dimension] = {i: dimension_scores[dimension].count(i) for i in range(1, 6)}
        
        statistics = {
            "total_trajectories": total,
            "valid_evaluations": len(valid_results),
            "mode": results[0]["mode"] if results else "unknown",
            "dimensions": dimensions,
            "average_scores": avg_scores,
            "average_scores_100": avg_scores_100,
            "penalized_average_scores_100": penalized_avg_scores_100,
            "final_average_scores_100": final_avg_scores_100,
            "rating_distribution": rating_distribution,
            "tool_error_statistics": {
                "average_tool_error_rate": avg_tool_error_rate,
                "total_samples_with_errors": sum(1 for rate in tool_error_rates if rate > 0),
                "error_rate_distribution": {
                    "0%": sum(1 for rate in tool_error_rates if rate == 0),
                    "0-25%": sum(1 for rate in tool_error_rates if 0 < rate <= 0.25),
                    "25-50%": sum(1 for rate in tool_error_rates if 0.25 < rate <= 0.5),
                    "50-75%": sum(1 for rate in tool_error_rates if 0.5 < rate <= 0.75),
                    "75-100%": sum(1 for rate in tool_error_rates if 0.75 < rate <= 1.0)
                }
            },
            "cache_statistics": cache_statistics,
            "meta_judge_statistics": meta_judge_statistics
        }
        
        return statistics
