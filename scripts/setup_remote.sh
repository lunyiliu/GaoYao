#!/usr/bin/env bash
# Setup GaoYao evaluation environment on remote GPU machine.
# Usage: bash scripts/setup_remote.sh
set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-4B-Instruct}"
HF_CACHE="${HF_CACHE:-/root/.cache/huggingface}"

echo "=== Install Python deps ==="
pip install -r "$REPO_DIR/requirements.txt" -q
pip install vllm -q

echo "=== Install COMET model ==="
python3 -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"

echo "=== Download dataset from GaoYaoEvalDataset ==="
DATA_DIR="$REPO_DIR/data/original"
if [ ! -d "$DATA_DIR/Belebele" ]; then
    git clone --filter=blob:none --sparse https://github.com/zhaocorey/GaoYaoEvalDataset.git /tmp/gaoyao_dataset
    cd /tmp/gaoyao_dataset && git sparse-checkout set data
    # Copy each dataset folder
    for ds in mgsm MMMLU belebele include flores101 cross_cultural mono-cultural superblend alpaca_eval mt_bench; do
        if [ -d "data/$ds" ]; then
            # Normalize directory names to match pipeline registry
            case $ds in
                mgsm)          dst="MGSM" ;;
                MMMLU)         dst="MMMLU" ;;
                belebele)      dst="Belebele" ;;
                include)       dst="Include" ;;
                flores101)     dst="Flores101" ;;
                cross_cultural)dst="CrossCultural" ;;
                mono-cultural) dst="MonoCultural" ;;
                superblend)    dst="SuperBLEnD" ;;
                alpaca_eval)   dst="S-AlpacaEval-placeholder" ;;  # replaced by data2.zip
                mt_bench)      dst="S-MT-Bench-placeholder" ;;
            esac
            cp -r "data/$ds" "$DATA_DIR/$dst"
            echo "  Copied $ds -> $dst"
        fi
    done
fi

echo ""
echo "=== Setup complete ==="
echo "Next steps:"
echo "  1. Set env vars: export MADE_API_KEY=<key>  GAOYAO_LLM_URL=http://localhost:8000/v1/chat/completions"
echo "  2. Start vLLM: bash scripts/start_vllm.sh"
echo "  3. Run 1% test: python run_eval.py --sample-pct 1"
