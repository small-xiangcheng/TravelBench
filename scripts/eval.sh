#!/bin/bash
# ==============================================================================
# Trajectory Evaluation Script
# Description: Automatically evaluate multi-turn, single-turn, and unsolved files
# Usage: bash scripts/eval.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
INPUT_DIR="<Your Input Directory Here>"
BASE_OUTPUT_DIR="./eval_output"
JUDGE_MODEL="gemini-3-flash-preview"  
JUDGE_API_KEY="$OPENAI_API_KEY"
JUDGE_API_BASE="$OPENAI_API_BASE"
META_JUDGE_MODEL=""
META_JUDGE_API_KEY=""
# ------------------------------------------------------------------------------
# Validation
# ------------------------------------------------------------------------------
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Extract the last directory name from the input path
DIR_NAME=$(basename "$INPUT_DIR")

# Create output directory
OUTPUT_DIR="$BASE_OUTPUT_DIR/$DIR_NAME"
mkdir -p "$OUTPUT_DIR"

# ------------------------------------------------------------------------------
# Input Files
# ------------------------------------------------------------------------------
MULTI_TURN_FILE="$INPUT_DIR/multi_turn.json"
SINGLE_TURN_FILE="$INPUT_DIR/single_turn.json"
UNSOLVED_FILE="$INPUT_DIR/unsolve.json"

for file in "$MULTI_TURN_FILE" "$SINGLE_TURN_FILE" "$UNSOLVED_FILE"; do
    if [ ! -f "$file" ]; then
        echo "Error: File does not exist: $file"
        exit 1
    fi
done

echo "=========================================="
echo "Starting Evaluation"
echo "=========================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_DIR"
echo "Judge model: $JUDGE_MODEL"
echo ""

# ------------------------------------------------------------------------------
# Run Evaluation
# ------------------------------------------------------------------------------
declare -A SCORES

# Task 1: Evaluate multi-turn
echo "----------------------------------------"
echo "Task1: Evaluating multi-turn..."
echo "----------------------------------------"
python -u -m travelbench.evaluation.evaluate \
    --input "$MULTI_TURN_FILE" \
    --output "$OUTPUT_DIR/multi-turn_eval.json" \
    --mode multi-turn \
    --model "$JUDGE_MODEL" \
    --base_url "$JUDGE_API_BASE" \
    --api_key "$JUDGE_API_KEY" \
    --max-concurrency 100 \
    --enable_meta_judge \
    --meta_judge_model "$META_JUDGE_MODEL" \
    --meta_judge_api_key "$META_JUDGE_API_KEY"

MULTI_SUMMARY="$OUTPUT_DIR/multi-turn_eval_summary.json"
if [ -f "$MULTI_SUMMARY" ]; then
    MULTI_SCORE=$(python3 -c "import json; f=open('$MULTI_SUMMARY'); d=json.load(f); print(f\"{d['final_average_scores_100']['overall_average']:.2f}\")")
    SCORES[multi_turn]=$MULTI_SCORE
    echo "✓ multi-turn evaluation completed, trial-averaged final score: $MULTI_SCORE"
fi
echo ""

# Task 2: Evaluate single-turn
echo "----------------------------------------"
echo "Task2: Evaluating single-turn..."
echo "----------------------------------------"
python -u -m travelbench.evaluation.evaluate \
    --input "$SINGLE_TURN_FILE" \
    --output "$OUTPUT_DIR/single-turn_eval.json" \
    --mode single-turn \
    --model "$JUDGE_MODEL" \
    --base_url "$JUDGE_API_BASE" \
    --api_key "$JUDGE_API_KEY" \
    --max-concurrency 100 \
    --enable_meta_judge \
    --meta_judge_model "$META_JUDGE_MODEL" \
    --meta_judge_api_key "$META_JUDGE_API_KEY"

SINGLE_SUMMARY="$OUTPUT_DIR/single-turn_eval_summary.json"
if [ -f "$SINGLE_SUMMARY" ]; then
    SINGLE_SCORE=$(python3 -c "import json; f=open('$SINGLE_SUMMARY'); d=json.load(f); print(f\"{d['final_average_scores_100']['overall_average']:.2f}\")")
    SCORES[single_turn]=$SINGLE_SCORE
    echo "✓ single-turn evaluation completed, trial-averaged final score: $SINGLE_SCORE"
fi
echo ""

# Task 3: Evaluate unsolved
echo "----------------------------------------"
echo "Task3: Evaluating unsolved..."
echo "----------------------------------------"
python -u -m travelbench.evaluation.evaluate_unsolved \
    --input "$UNSOLVED_FILE" \
    --output "$OUTPUT_DIR/unsolved_eval.json"

UNSOLVED_SUMMARY="$OUTPUT_DIR/unsolved_eval_summary.json"
if [ -f "$UNSOLVED_SUMMARY" ]; then
    UNSOLVED_SCORE=$(python3 -c "import json; f=open('$UNSOLVED_SUMMARY'); d=json.load(f); print(f\"{d['unsolved_accuracy']*100:.2f}\")")
    SCORES[unsolved]=$UNSOLVED_SCORE
    echo "✓ unsolved evaluation completed, detection rate: $UNSOLVED_SCORE"
fi
echo ""

# ------------------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------------------
echo "=========================================="
echo "Evaluation Completed - Final Statistics"
echo "=========================================="
echo "Task Scores (0-100):"
echo "  Multi-turn:   ${SCORES[multi_turn]}"
echo "  Single-turn:  ${SCORES[single_turn]}"
echo "  Unsolved:     ${SCORES[unsolved]}"
echo ""

# Calculate average score
if [ ${#SCORES[@]} -eq 3 ]; then
    AVG_SCORE=$(python3 -c "scores=[${SCORES[multi_turn]}, ${SCORES[single_turn]}, ${SCORES[unsolved]}]; print(f'{sum(scores)/len(scores):.2f}')")
    echo "Overall Average Score: $AVG_SCORE"
    echo ""
fi

echo "Detailed results saved to: $OUTPUT_DIR"
echo "=========================================="
