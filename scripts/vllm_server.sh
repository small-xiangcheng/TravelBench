#!/bin/bash
# ==============================================================================
# vLLM Main Model Deployment Script
# Description: Deploy the main LLM model using vLLM server
# Usage: source scripts/vllm_server.sh
# ==============================================================================

# ------------------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------------------
GPU_COUNT=$(echo "$MODEL_GPU_LIST" | awk -F',' '{print NF}')
mkdir -p "$LOG_DIR"

# Validate DP * TP = GPU count
EXPECTED_GPU_COUNT=$((MODEL_DP_SIZE * MODEL_TP_SIZE))
if [ "$EXPECTED_GPU_COUNT" -ne "$GPU_COUNT" ]; then
    echo "❌ Error: MODEL_DP_SIZE($MODEL_DP_SIZE) * MODEL_TP_SIZE($MODEL_TP_SIZE) = $EXPECTED_GPU_COUNT, but GPU_COUNT = $GPU_COUNT"
    echo "   Please ensure DP * TP equals the number of GPUs in MODEL_GPU_LIST"
    return 1 2>/dev/null || exit 1
fi

net0_ip=$(ifconfig net0 2>/dev/null | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -n 1)
[ -z "$net0_ip" ] && net0_ip=$(hostname -I | awk '{print $1}')
log_file="${LOG_DIR}/${MODEL_NAME}_$(echo $net0_ip | tr '.' '_').log"

# Set default tool call parser if not specified
TOOL_CALL_PARSER="${MODEL_TOOL_CALL_PARSER:-hermes}"

echo "🚀 Deploying model: ${MODEL_NAME}"
echo "   GPU(s): ${MODEL_GPU_LIST} (${GPU_COUNT} cards)"
echo "   TP: ${MODEL_TP_SIZE}, DP: ${MODEL_DP_SIZE}"
echo "   Port: ${MODEL_PORT}"
echo "   Mode: ${MODEL_INFERENCE_MODE}"
echo "   Tool Parser: ${TOOL_CALL_PARSER}"
[ -n "$MODEL_CHAT_TEMPLATE" ] && echo "   Chat Template: ${MODEL_CHAT_TEMPLATE}"

# ------------------------------------------------------------------------------
# Start Server
# ------------------------------------------------------------------------------
VLLM_CMD="vllm serve ${MODEL_PATH} \
    --served-model-name ${MODEL_NAME} \
    --max-model-len ${MODEL_MAX_LEN} \
    --tensor-parallel-size ${MODEL_TP_SIZE} \
    --data-parallel-size ${MODEL_DP_SIZE} \
    --gpu-memory-utilization 0.9 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser ${TOOL_CALL_PARSER} \
    --host 0.0.0.0 \
    --port ${MODEL_PORT}" 


[ -n "$MODEL_CHAT_TEMPLATE" ] && VLLM_CMD="$VLLM_CMD --chat-template ${MODEL_CHAT_TEMPLATE}"
[ "$MODEL_INFERENCE_MODE" = "think" ] && VLLM_CMD="$VLLM_CMD --reasoning-parser ${MODEL_REASONING_PARSER:-deepseek_r1}"

nohup bash -c "export CUDA_VISIBLE_DEVICES=${MODEL_GPU_LIST}; $VLLM_CMD" > "$log_file" 2>&1 &

echo "⏳ Waiting for server to start..."
while true; do
    if grep -q "Started server process" "$log_file" 2>/dev/null; then
        echo "✅ Server started successfully!"
        break
    fi
    sleep 5
done

# ------------------------------------------------------------------------------
# Output Results
# ------------------------------------------------------------------------------
echo -e "\n===== Deployment Complete ====="
export MODEL_SERVICE_URL="http://$net0_ip:$MODEL_PORT/v1"
export MODEL_NAME="${MODEL_NAME}"
echo "📍 Endpoint: $MODEL_SERVICE_URL"
echo "📝 Log: $log_file"
echo ""
echo "Environment variables:"
echo "  MODEL_SERVICE_URL=$MODEL_SERVICE_URL"
echo "  MODEL_NAME=$MODEL_NAME"