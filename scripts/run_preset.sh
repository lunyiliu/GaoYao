#!/usr/bin/env bash
# Named evaluation presets — weak-model one-shot invocation.
# Usage: bash scripts/run_preset.sh <preset> <model_name> [extra run_eval.py args]
#
# Presets:
#   smoke         1% random sample, 4 MCQ datasets (fast sanity check)
#   lang10        10% per-language stratified, all datasets (recommended default)
#   full          100%, all datasets
#   stage1-infer  full inference only; judge calls fail silently (offline GPU machine)
#   stage2-score  scoring on cached inference (on API-reachable machine)
set -e

PRESET="${1:?usage: $0 <preset> <model_name> [extra args]}"
MODEL_NAME="${2:?usage: $0 <preset> <model_name> [extra args]}"
shift 2

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
MODEL_URL="${GAOYAO_MODEL_URL:-http://localhost:8000/v1/chat/completions}"
WORKERS="${GAOYAO_WORKERS:-4}"

cd "$REPO_DIR"
BASE=(python3 run_eval.py --model-url "$MODEL_URL" --model-name "$MODEL_NAME" --workers "$WORKERS")

case "$PRESET" in
    smoke)
        "${BASE[@]}" --datasets mgsm,mmmlu,belebele,include --sample-pct 1 "$@"
        ;;
    lang10)
        "${BASE[@]}" --datasets all --sample-lang-pct 10 "$@"
        ;;
    full)
        "${BASE[@]}" --datasets all "$@"
        ;;
    stage1-infer)
        "${BASE[@]}" --datasets all "$@"
        ;;
    stage2-score)
        "${BASE[@]}" --datasets all --skip-inference "$@"
        ;;
    *)
        echo "unknown preset: $PRESET" >&2
        echo "valid: smoke | lang10 | full | stage1-infer | stage2-score" >&2
        exit 2
        ;;
esac
