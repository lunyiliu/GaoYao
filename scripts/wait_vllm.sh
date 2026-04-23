#!/usr/bin/env bash
# Wait for vLLM server to become ready. Exit 0 on success, 1 on timeout.
# Usage: bash scripts/wait_vllm.sh [log_file] [timeout_sec]
set -u

LOG="${1:-/tmp/vllm.log}"
TIMEOUT="${2:-600}"

start=$(date +%s)
while :; do
    if grep -q 'Application startup complete' "$LOG" 2>/dev/null; then
        echo "[wait_vllm] ready"
        exit 0
    fi
    now=$(date +%s)
    if [ $((now - start)) -ge "$TIMEOUT" ]; then
        echo "[wait_vllm] timeout after ${TIMEOUT}s"
        tail -80 "$LOG" 2>/dev/null
        exit 1
    fi
    sleep 5
done
