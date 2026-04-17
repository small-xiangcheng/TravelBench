#!/bin/bash
# ==============================================================================
# Base Configuration File
# Description: Central configuration for all TravelBench scripts
# Usage: source scripts/base_config.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Main Model Configuration
# ------------------------------------------------------------------------------
export MODEL_PATH="<Your Model Path Here>"
export MODEL_NAME="<Your Model Name Here>"
export MODEL_PORT=10000
export MODEL_MAX_LEN=131072
export MODEL_GPU_LIST="0,1,2,3"           # GPUs to use, e.g., "0,1,2,3"
export MODEL_TP_SIZE=1                     # Tensor Parallel size
export MODEL_DP_SIZE=4                     # Data Parallel size (DP * TP must equal GPU count)
export MODEL_INFERENCE_MODE="no-think"     # think or no-think
export MODEL_TOOL_CALL_PARSER=""           # Tool call parser, e.g., "llama3_json", "hermes". Default: hermes if empty
export MODEL_CHAT_TEMPLATE=""              # Chat template path, e.g., "chat_template/tool_chat_template_llama3.1_json.jinja". Optional
export MODEL_REASONING_PARSER="" # Reasoning parser for think mode, e.g., "deepseek_r1". Default: deepseek_r1

# ------------------------------------------------------------------------------
# Embedding Model Configuration
# ------------------------------------------------------------------------------
export EMBEDDING_MODEL_PATH="<Your Embedding Model Path Here>"
export EMBEDDING_MODEL_NAME="Qwen3-Embedding-8B"
export EMBEDDING_PORT=40001
export EMBEDDING_MAX_LEN=32768
export EMBEDDING_GPU_LIST="4,5,6,7"        # GPUs to use, e.g., "4,5,6,7"
export EMBEDDING_TP_SIZE=2                 # Tensor Parallel size
export EMBEDDING_DP_SIZE=2                 # Data Parallel size (DP * TP must equal GPU count)

# ------------------------------------------------------------------------------
# Common Configuration
# ------------------------------------------------------------------------------
export LOG_DIR="logs"
export SANDBOX_CACHE_DIR="./sandbox_cache"

# ------------------------------------------------------------------------------
# API Configuration
# ------------------------------------------------------------------------------
export OPENAI_API_KEY="<Your OpenAI API Key Here>"
export OPENAI_API_BASE="<Your OpenAI API Base URL Here>"
