#!/usr/bin/env bash
# Start vLLM server for Qwen3-4B-Instruct.
# Usage: bash scripts/start_vllm.sh [model_name]
# NOTE: Requires CUDA sm_70+. GTX 1080 Ti (sm_61) is NOT compatible.
#       Use --model-url local:///<model_id> for older GPUs (transformers backend).
set -e

MODEL="${1:-Qwen/Qwen3-4B-Instruct}"
PORT="${VLLM_PORT:-8000}"

echo "Starting vLLM: model=$MODEL port=$PORT"

python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --served-model-name "qwen3-4b" \
    --port "$PORT" \
    --dtype float16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code
