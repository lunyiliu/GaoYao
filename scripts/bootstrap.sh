#!/usr/bin/env bash
# Idempotent environment bootstrap for GaoYao.
# Checks Python, GPU, project reqs, vLLM, transformers pin, COMET (model + pkg),
# applies Blackwell patch if sm_120, creates ~/.gaoyao_env template if missing.
# Usage: bash scripts/bootstrap.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_DIR"

log() { echo "[bootstrap] $*"; }
pymod() { python3 -c "import $1" >/dev/null 2>&1; }

command -v python3 >/dev/null 2>&1 || { log "ERROR: python3 not found"; exit 1; }
log "python3 $(python3 --version 2>&1 | awk '{print $2}')"

CAP=0
if command -v nvidia-smi >/dev/null 2>&1; then
    CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
    log "GPU compute_cap=$CAP"
    if [ "$CAP" -lt 70 ]; then
        log "ERROR: compute_cap < 7.0, vLLM unsupported"
        exit 1
    fi
else
    log "WARN: nvidia-smi not found; skipping GPU check"
fi

if [ -f requirements.txt ]; then
    log "installing requirements.txt"
    pip install -r requirements.txt -q
fi

if ! pymod vllm; then
    log "installing vllm"
    pip install vllm -q
fi
log "vllm $(python3 -c 'import vllm; print(vllm.__version__)' 2>/dev/null)"

if ! python3 -c "import transformers, re; v=transformers.__version__; assert tuple(int(x) for x in re.match(r'(\d+)\.(\d+)', v).groups()) >= (4,56), v" 2>/dev/null; then
    log "pinning transformers>=4.56,<4.57"
    pip install 'transformers>=4.56.0,<4.57.0' -q
fi

if ! pymod comet; then
    log "installing unbabel-comet"
    pip install unbabel-comet -q
fi

log "ensuring COMET model wmt22-comet-da (downloads on first run, then cached)"
python3 -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')" >/dev/null

if [ "$CAP" -ge 120 ]; then
    log "sm_120 detected — applying Blackwell vLLM patch"
    python3 scripts/patch_vllm_blackwell.py || log "WARN: patch reported no change"
fi

if [ ! -f "$HOME/.gaoyao_env" ]; then
    cat > "$HOME/.gaoyao_env" << 'EOF'
# GaoYao credentials — fill in before running evaluation.

# Judge model (required for subjective scoring)
export GAOYAO_JUDGE_API_KEY=
export GAOYAO_JUDGE_MODEL=
export GAOYAO_JUDGE_BASE_URL=

# Target (tested) model — only set the key if GAOYAO_LLM_URL is a hosted API.
# Local vLLM endpoints do not need it.
# export GAOYAO_LLM_API_KEY=
EOF
    chmod 600 "$HOME/.gaoyao_env"
    log "created ~/.gaoyao_env template — edit it with your credentials"
else
    log "~/.gaoyao_env already exists (not overwritten)"
fi

log "bootstrap complete"
