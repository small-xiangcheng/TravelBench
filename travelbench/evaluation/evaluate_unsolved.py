#!/usr/bin/env python3
"""
Unsolved detection evaluation script.
Evaluates model's ability to recognize boundary cases and correctly output [Unsolved] marker.
"""

import json
import argparse
from typing import List, Dict

# ==================== Data Loader ====================
class DataLoader:
    """Load and process data."""
    
    @staticmethod
    def load_trajectories(file_path: str) -> List[Dict]:
        """
        Load trajectory file (JSON format).
        
        Supports:
        1. Direct array: [{"conversation_id": "...", ...}, ...]
        2. With results field: {"summary": {...}, "results": [...]}
        3. Single object: {"conversation_id": "...", ...}
        """
        trajectories = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine JSON structure
            if isinstance(data, list):
                # Direct array format
                trajectories = data
            elif isinstance(data, dict):
                # May contain results field
                if "results" in data:
                    trajectories = data["results"]
                else:
                    # Single trajectory object
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
        """Save evaluation results as JSON format."""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Results saved to: {output_file}")

# ==================== Unsolved Detector ====================
class UnsolvedDetector:
    """Detect whether assistant response contains [Unsolved] marker."""
    
    @staticmethod
    def check_unsolved(trajectory: Dict) -> Dict:
        """
        Check if a single trajectory contains [Unsolved].
        
        Args:
            trajectory: Trajectory data.
            
        Returns:
            Dict containing detection results.
        """
        messages = trajectory.get("messages", [])
        
        # Find the first assistant reply
        first_assistant_reply = None
        for msg in messages:
            if msg.get("role") == "assistant":
                first_assistant_reply = msg.get("content", "")
                break
        
        # Check if contains [Unsolved]
        has_unsolved = False
        if first_assistant_reply:
            has_unsolved = "[Unsolved]" in first_assistant_reply
        
        return {
            "conversation_id": trajectory.get("conversation_id"),
            "query": trajectory.get("query"),
            "first_assistant_reply": first_assistant_reply,
            "has_unsolved": has_unsolved,
            "unsolved_score": 1 if has_unsolved else 0
        }
    
    @staticmethod
    def evaluate_batch(trajectories: List[Dict]) -> List[Dict]:
        """Batch evaluation."""
        results = []
        
        for i, trajectory in enumerate(trajectories):
            print(f"[{i}] Checking trajectory: {trajectory.get('conversation_id', 'unknown')}")
            result = UnsolvedDetector.check_unsolved(trajectory)
            result["index"] = i
            results.append(result)
            
            status = "contains [Unsolved]" if result["has_unsolved"] else "does not contain [Unsolved]"
            print(f"[{i}] Check completed: {status}")
        
        return results

# ==================== Statistics Module ====================
class StatisticsCalculator:
    """Calculate statistics."""
    
    @staticmethod
    def calculate_statistics(results: List[Dict]) -> Dict:
        """Calculate overall statistics."""
        total = len(results)
        
        if total == 0:
            return {
                "total_trajectories": 0,
                "unsolved_count": 0,
                "non_unsolved_count": 0,
                "unsolved_accuracy": 0.0
            }
        
        # Count trajectories containing [Unsolved]
        unsolved_count = sum(1 for r in results if r["has_unsolved"])
        non_unsolved_count = total - unsolved_count
        
        # Calculate accuracy (assuming all samples should return [Unsolved])
        unsolved_accuracy = unsolved_count / total if total > 0 else 0
        
        statistics = {
            "total_trajectories": total,
            "unsolved_count": unsolved_count,
            "non_unsolved_count": non_unsolved_count,
            "unsolved_accuracy": unsolved_accuracy,
            "unsolved_percentage": unsolved_accuracy * 100
        }
        
        return statistics

# ==================== Main Function ====================
def main():
    parser = argparse.ArgumentParser(description="Unsolved Detection Evaluation Tool")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file path")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file path")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Unsolved Detection Evaluation Tool")
    print("=" * 60)
    print(f"Input file: {args.input}")
    print(f"Output file: {args.output}")
    print("=" * 60)
    
    # Load data
    print("\n[1/3] Loading data...")
    trajectories = DataLoader.load_trajectories(args.input)
    
    if not trajectories:
        print("[Error] No trajectory data loaded")
        return
    
    # Evaluate
    print(f"\n[2/3] Starting detection for {len(trajectories)} trajectories...")
    results = UnsolvedDetector.evaluate_batch(trajectories)
    
    # Calculate statistics
    print("\n[3/3] Calculating statistics...")
    statistics = StatisticsCalculator.calculate_statistics(results)
    
    print("\nStatistics:")
    print(f"  Total trajectories: {statistics['total_trajectories']}")
    print(f"  Contains [Unsolved]: {statistics['unsolved_count']}")
    print(f"  Does not contain [Unsolved]: {statistics['non_unsolved_count']}")
    print(f"  [Unsolved] accuracy: {statistics['unsolved_accuracy']:.2%}")
    
    # Save results
    print("\nSaving results...")
    
    # Save detailed results
    DataLoader.save_results(results, args.output)
    
    # Save statistics summary
    summary_file = args.output.replace('.json', '_summary.json')
    if summary_file == args.output:
        # If no .json suffix, add _summary
        summary_file = args.output + '_summary.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(statistics, f, ensure_ascii=False, indent=2)
    print(f"Statistics summary saved to: {summary_file}")
    
    print("\n" + "=" * 60)
    print("Detection completed!")
    print("=" * 60)

if __name__ == "__main__":
    main()
