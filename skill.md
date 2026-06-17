# GaoYao — Agent Skill

End-to-end deployment and evaluation of the GaoYao multilingual benchmark on a remote GPU machine.

> **GaoYao is accepted by ACL 2026 main.**

## Overview

This skill is compiled for weak-model compatibility:

- Every step is a single copy-paste command — no parameter reasoning required
- Environment setup is one idempotent script (`scripts/bootstrap.sh`)
- Evaluation is invoked through named presets (`scripts/run_preset.sh <preset>`)
- Each step has a pass / fail gate before the next step runs

## When to use

When a user asks to evaluate a model on GaoYao, benchmark multilingual performance, or reproduce GaoYao paper results.

## Inputs — collect these before step 1

| Variable | Meaning | How to obtain |
|----------|---------|---------------|
| `MODEL_ID` | HuggingFace repo id or local path of the target model | e.g. `organization/model-name` |
| `MODEL_NAME` | short handle for this run's output directory | any string, e.g. `my-model-v1` |
| `GAOYAO_JUDGE_API_KEY` | judge API key | OpenAI-compatible provider (legacy `GAOYAO_API_KEY` still works) |
| `GAOYAO_JUDGE_MODEL` | judge model name | judge provider's model id |
| `GAOYAO_JUDGE_BASE_URL` | judge API base URL | provider's `/v1` endpoint |
| `GAOYAO_LLM_API_KEY` *(optional)* | Bearer key for the target model | only when `GAOYAO_LLM_URL` is a hosted API; local vLLM does not need it |

Export `MODEL_ID` and `MODEL_NAME` once at the start of the session; subsequent commands reference them as env vars so you never hand-substitute placeholders.

```bash
export MODEL_ID=<hf_model_id>
export MODEL_NAME=<short_handle>
```

## One-shot (weak-model happy path)

If every step should just work, run this pipeline and only branch into the detailed steps if something exits non-zero:

```bash
git clone https://github.com/lunyiliu/GaoYao.git && cd GaoYao \
  && bash scripts/check_gpu.sh \
  && bash scripts/bootstrap.sh \
  && $EDITOR ~/.gaoyao_env && source ~/.gaoyao_env \
  && bash scripts/start_vllm.sh "$MODEL_ID" "$MODEL_NAME" \
  && bash scripts/wait_vllm.sh \
  && bash scripts/run_preset.sh smoke "$MODEL_NAME" \
  && bash scripts/run_preset.sh lang10 "$MODEL_NAME"
```

## Steps

### Step 1 — Verify GPU

**precondition:** SSH session on target GPU machine.
**action:**
```bash
bash scripts/check_gpu.sh
```
**pass:** stdout ends with `OK: compute_cap=XX` where `XX ≥ 70`.
**fail:** compute_cap < 70 → GPU cannot run vLLM. Stop and report to user.

### Step 2 — Clone repo and bootstrap environment

**precondition:** step 1 passed.
**action:**
```bash
git clone https://github.com/lunyiliu/GaoYao.git
cd GaoYao
bash scripts/bootstrap.sh
```
**pass:** bootstrap prints `bootstrap complete`; `~/.gaoyao_env` exists.
**fail:** read the last log line for the failed package; run `pip install <pkg>` manually, then re-run `bash scripts/bootstrap.sh`.

The bootstrap is idempotent — safe to re-run. It installs `requirements.txt`, vLLM, `unbabel-comet`, pins `transformers>=4.56,<4.57` if conflicting, pre-downloads the COMET checkpoint, applies the Blackwell patch on sm_120, and seeds `~/.gaoyao_env`.

### Step 3 — Configure judge credentials

**action:** edit `~/.gaoyao_env` to fill the three `GAOYAO_*` values, then source it.
```bash
$EDITOR ~/.gaoyao_env
source ~/.gaoyao_env
```
**verify:**
```bash
[ -n "${GAOYAO_JUDGE_API_KEY:-$GAOYAO_API_KEY}" ] && [ -n "$GAOYAO_JUDGE_MODEL" ] && [ -n "$GAOYAO_JUDGE_BASE_URL" ] && echo OK
```
expected stdout: `OK`.
**fail:** re-edit `~/.gaoyao_env`, re-source.

### Step 4 — Start vLLM server

**precondition:** steps 2–3 passed; `MODEL_ID` and `MODEL_NAME` exported.
**action:**
```bash
bash scripts/start_vllm.sh "$MODEL_ID" "$MODEL_NAME"
bash scripts/wait_vllm.sh
```
The start script auto-detects sm_120 and applies the Blackwell patch and `--enforce-eager` when needed.
**verify:**
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"$MODEL_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":5}" \
  | grep -q '"content"' && echo OK
```
expected stdout: `OK`.
**fail:** `tail -80 /tmp/vllm.log` — cross-reference the troubleshooting table below.

### Step 5 — Smoke test

**action:**
```bash
bash scripts/run_preset.sh smoke "$MODEL_NAME"
```
Runs 4 MCQ datasets (mgsm, mmmlu, belebele, include) at 1% sampling — completes in minutes, exercises the full pipeline (inference + judge + scoring).
**verify:** `results/$MODEL_NAME/evaluation_result/mgsm/metrics.json` exists and is non-empty.
**fail:** judge 401/timeout → re-check step 3 credentials. GPU OOM → lower `--gpu-memory-utilization` in `scripts/start_vllm.sh` and restart vLLM.

### Step 6 — Full evaluation (pick one preset)

| Preset | When | Command |
|--------|------|---------|
| `lang10` | recommended default — 10% per-language stratified, guarantees every language is sampled | `bash scripts/run_preset.sh lang10 "$MODEL_NAME"` |
| `full` | 100% all datasets; GPU machine has judge API access | `bash scripts/run_preset.sh full "$MODEL_NAME"` |
| `stage1-infer` | GPU machine cannot reach judge API — inference only, results cached | `bash scripts/run_preset.sh stage1-infer "$MODEL_NAME"` |
| `stage2-score` | on an API-reachable machine after `scp`-ing `results/$MODEL_NAME` over | `bash scripts/run_preset.sh stage2-score "$MODEL_NAME"` |

**verify:** `results/$MODEL_NAME/evaluation_result/` contains one subdir per dataset, each with `metrics.json`.

To override the judge per run, append CLI args — they flow through to `run_eval.py`:
```bash
bash scripts/run_preset.sh lang10 "$MODEL_NAME" \
  --judge-url https://<judge_api>/v1/chat/completions \
  --judge-name <judge_model_name>
```

### Step 7 — Read results

Console output format:

```
========================================================
Dataset                   Score  Ref(paper)     Delta
--------------------------------------------------------
MGSM                     0.7985      0.7985   +0.0000
MMMLU                    0.6203      0.6203   +0.0000
...
========================================================
```

With the `smoke` preset, `|Delta| ≤ 0.15` is normal; for SAGE and CultureScope the 1% slice has only 8–50 items per dataset, so ±0.3 is within noise.

## Two-stage evaluation (offline GPU machine)

When the GPU machine cannot reach the judge API:

**Stage 1 — on the GPU machine:**
```bash
bash scripts/run_preset.sh stage1-infer "$MODEL_NAME"
```
Judge calls fail silently; inference results are cached at `results/$MODEL_NAME/inference_result/`.

**Stage 2 — on a machine that can reach the judge API:**
```bash
scp -P <port> -r <user>@<host>:GaoYao/results/$MODEL_NAME ./results/
bash scripts/run_preset.sh stage2-score "$MODEL_NAME"
```

Flores-101 needs `unbabel-comet` on the scoring machine. If it is not available there, run Flores on the GPU machine instead:
```bash
python3 run_eval.py --datasets flores --skip-inference \
  --model-url http://localhost:8000/v1/chat/completions \
  --model-name "$MODEL_NAME"
```

## Troubleshooting

Match the error substring on the left, apply the fix on the right.

| Error | Fix |
|-------|-----|
| `no kernel image for device` | GPU compute_cap < 7.0 — unsupported, step 1 should have caught this |
| `PTX compiled with unsupported toolchain` | sm_120 patch missing — `python3 scripts/patch_vllm_blackwell.py` then restart step 4 |
| `Free memory < desired utilization` | leftover GPU procs — `pkill -9 -f vllm` then retry step 4 |
| `tensorflow_text` import error | `pip install 'transformers>=4.56.0,<4.57.0'` |
| `401 Unauthorized` from judge | re-check `GAOYAO_JUDGE_API_KEY` (or legacy `GAOYAO_API_KEY`), re-source `~/.gaoyao_env` |
| `401 Unauthorized` from target model | set `GAOYAO_LLM_API_KEY` if `GAOYAO_LLM_URL` is a hosted API |
| Judge timeout / `Connection refused` | switch to two-stage flow (`stage1-infer` then `stage2-score`) |
| `ImportError: comet` / `unbabel-comet` | `pip install unbabel-comet` on the scoring machine |

## Capability requirements

This skill is compiled against the following capability floor. Models at or above these levels can execute it reliably.

| Capability | Level | Why this level suffices |
|------------|-------|-------------------------|
| `tool.exec` | L1 | all commands are literal copy-paste; no flag composition required |
| `follow.procedure` | L2 | 7 sequential steps with verify gates and one branch (Blackwell auto-applied in bootstrap) |
| `gen.code.shell` | not required | all shell logic is bundled in `scripts/` |
| `reason.arithmetic` | not required | sampling percentages are fixed inside presets |
| `gen.code.python` | not required | the only Python entry point (`patch_vllm_blackwell.py`) runs with no arguments |

For models below L2 on `follow.procedure`, invoke only the one-shot pipeline at the top of this file; skip the per-step narrative.
