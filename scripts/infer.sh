#!/bin/bash
# ==============================================================================
# Inference Script
# Description: Run inference for multi-turn, single-turn, and unsolved tasks
# Usage: bash scripts/infer.sh
# ==============================================================================

set -e

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------
BASE_OUTPUT_PATH="./output"

USER_LLM="gpt-41-0414-global"
TOOL_LLM="gpt-41-0414-global"
AGENT_LLM="${MODEL_NAME}"
USE_CUSTOM_ENDPOINT=true  # Whether to use custom model endpoint (for self-deployed models)

# LLM parameters
if [ "$USE_CUSTOM_ENDPOINT" = "true" ]; then
  AGENT_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.7, \"api_base\": \"${MODEL_SERVICE_URL}\"}"
else
  AGENT_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.7}"
fi
USER_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.0}"
TOOL_LLM_ARGS="{\"max_tokens\": 8192, \"temperature\": 0.0}"

# ------------------------------------------------------------------------------
# Run Inference
# ------------------------------------------------------------------------------

# Task 1: unsolved
INPUT_FILE="./datas/unsolve.jsonl"
INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
MODE=unsolved
OUTPUT_DIR="${BASE_OUTPUT_PATH}/agent-${AGENT_LLM}_user-${USER_LLM}"
OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_BASENAME}.json"
echo "=====Starting Task: $MODE====="
mkdir -p "$OUTPUT_DIR"
echo "📁 Output path: $OUTPUT_FILE"

python -u -m travelbench run \
  --file "$INPUT_FILE" \
  --mode "$MODE" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$USER_LLM_ARGS" \
  --tool-llm "$TOOL_LLM" \
  --tool-llm-args "$TOOL_LLM_ARGS" \
  --num-trials 3 \
  --max-concurrency 300 \
  --output "$OUTPUT_FILE"

# Task 2: single_turn
INPUT_FILE="./datas/single_turn.jsonl"
INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
MODE=single_turn
OUTPUT_DIR="${BASE_OUTPUT_PATH}/agent-${AGENT_LLM}_user-${USER_LLM}"
OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_BASENAME}.json"
echo "=====Starting Task: $MODE====="
mkdir -p "$OUTPUT_DIR"
echo "📁 Output path: $OUTPUT_FILE"

python -u -m travelbench run \
  --file "$INPUT_FILE" \
  --mode "$MODE" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$USER_LLM_ARGS" \
  --tool-llm "$TOOL_LLM" \
  --tool-llm-args "$TOOL_LLM_ARGS" \
  --num-trials 3 \
  --max-concurrency 100 \
  --output "$OUTPUT_FILE"

# Task 3: multi_turn
INPUT_FILE="./datas/multi_turn.jsonl"
INPUT_BASENAME=$(basename "$INPUT_FILE" .jsonl)
MODE=multi_turn
OUTPUT_DIR="${BASE_OUTPUT_PATH}/agent-${AGENT_LLM}_user-${USER_LLM}"
OUTPUT_FILE="${OUTPUT_DIR}/${INPUT_BASENAME}.json"
echo "=====Starting Task: $MODE====="
mkdir -p "$OUTPUT_DIR"
echo "📁 Output path: $OUTPUT_FILE"

python -u -m travelbench run \
  --file "$INPUT_FILE" \
  --mode "$MODE" \
  --agent-llm "$AGENT_LLM" \
  --agent-llm-args "$AGENT_LLM_ARGS" \
  --user-llm "$USER_LLM" \
  --user-llm-args "$USER_LLM_ARGS" \
  --tool-llm "$TOOL_LLM" \
  --tool-llm-args "$TOOL_LLM_ARGS" \
  --num-trials 3 \
  --max-concurrency 100 \
  --output "$OUTPUT_FILE"
