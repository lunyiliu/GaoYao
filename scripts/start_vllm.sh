#!/usr/bin/env bash
# Start a vLLM OpenAI-compatible server in the background.
# Log: $VLLM_LOG (default /tmp/vllm.log). Auto-applies Blackwell patch on sm_120.
# Usage:
#   bash scripts/start_vllm.sh <hf_model_id> [served_name]
#   MODEL_ID=<id> SERVED_NAME=<name> bash scripts/start_vllm.sh
set -e

MODEL_ID="${1:-${MODEL_ID:-}}"
if [ -z "$MODEL_ID" ]; then
    echo "usage: $0 <hf_model_id> [served_name]" >&2
    exit 2
fi
SERVED_NAME="${2:-${SERVED_NAME:-$(basename "$MODEL_ID" | tr '[:upper:]' '[:lower:]')}}"
PORT="${VLLM_PORT:-8000}"
LOG="${VLLM_LOG:-/tmp/vllm.log}"

CAP=0
if command -v nvidia-smi >/dev/null 2>&1; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
fi

echo "[start_vllm] model=$MODEL_ID served=$SERVED_NAME port=$PORT cap=$CAP log=$LOG"

EXTRA=()
if [ "$CAP" -ge 120 ]; then
    python3 "$(dirname "$0")/patch_vllm_blackwell.py" || true
    EXTRA=(--dtype float16 --enforce-eager)
else
    EXTRA=(--dtype auto)
fi

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_ID" \
    --served-model-name "$SERVED_NAME" \
    --port "$PORT" \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --trust-remote-code \
    "${EXTRA[@]}" > "$LOG" 2>&1 &

PID=$!
echo "[start_vllm] pid=$PID — wait with: bash scripts/wait_vllm.sh $LOG"
