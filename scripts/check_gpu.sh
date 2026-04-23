#!/usr/bin/env bash
# Step 1 capability gate. Exits 0 if GPU usable by vLLM, 1 otherwise.
set -u

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "FAIL: nvidia-smi not found"
    exit 1
fi

nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')

if [ -z "$CAP" ] || [ "$CAP" -lt 70 ]; then
    echo "FAIL: compute_cap=$CAP < 7.0 (vLLM requires sm_70+)"
    exit 1
fi

echo "OK: compute_cap=$CAP"
