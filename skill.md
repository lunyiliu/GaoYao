# GaoYao Multilingual Evaluation — Agent Skill

Deploy and run the GaoYao multilingual benchmark on a remote GPU machine end-to-end.

> **GaoYao is accepted by ACL 2026 main.**

## When to use

When a user asks to evaluate a model on GaoYao, benchmark multilingual performance, or reproduce GaoYao paper results.

## Prerequisites

- Remote GPU machine with SSH access (CUDA sm_70+: V100, RTX 20xx/30xx/40xx/50xx series)
- `GAOYAO_API_KEY` environment variable — required for judge calls and MMMLU fallback
- Judge model configurable via `GAOYAO_JUDGE_MODEL` env var or `--judge-name` CLI arg
- The model to evaluate must be accessible from the GPU machine (HuggingFace Hub or local path)

## Steps

### 1. Connect and verify GPU

```bash
ssh <user>@<host> -p <port>
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

**Minimum**: compute capability ≥ 7.0. If < 7.0, stop — vLLM won't work.

### 2. Clone repo and install dependencies

```bash
git clone https://github.com/lunyiliu/GaoYao.git
cd GaoYao
pip install -r requirements.txt -q
pip install vllm -q
```

If `transformers` version conflict (common on pre-configured machines):
```bash
pip install 'transformers>=4.56.0,<4.57.0' -q
```

### 3. Configure environment

```bash
export GAOYAO_API_KEY=<your_api_key>
export GAOYAO_JUDGE_MODEL=<judge_model_name>
export GAOYAO_JUDGE_BASE_URL=<judge_api_base_url>
```

Optionally persist to a local file:
```bash
cat > ~/.gaoyao_env << 'EOF'
export GAOYAO_API_KEY=<your_api_key>
export GAOYAO_JUDGE_MODEL=<judge_model_name>
export GAOYAO_JUDGE_BASE_URL=<judge_api_base_url>
EOF
chmod 600 ~/.gaoyao_env && source ~/.gaoyao_env
```

### 4. Download COMET model (for Flores-101)

```bash
python3 -c "from comet import download_model; download_model('Unbabel/wmt22-comet-da')"
```

### 5. Start vLLM server

**Standard GPU (sm_70 – sm_90):**
```bash
nohup bash scripts/start_vllm.sh <hf_model_id> > /tmp/vllm.log 2>&1 &
```

**Blackwell GPU (sm_120, RTX 50xx):**
```bash
# Patch cuda.py to use TRITON_ATTN backend for sm_120
python3 - << 'EOF'
import glob, os
candidates = glob.glob('/usr/local/lib/python3.*/dist-packages/vllm/platforms/cuda.py')
path = candidates[0] if candidates else None
if not path:
    print('cuda.py not found'); exit(1)
with open(path) as f: src = f.read()
old = '''        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]'''
new = '''        elif device_capability.major == 12:
            return [
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
                AttentionBackendEnum.FLASHINFER,
            ]
        else:
            return [
                AttentionBackendEnum.FLASH_ATTN,
                AttentionBackendEnum.FLASHINFER,
                AttentionBackendEnum.TRITON_ATTN,
                AttentionBackendEnum.FLEX_ATTENTION,
            ]'''
if old in src:
    open(path,'w').write(src.replace(old, new, 1)); print('PATCHED')
else:
    print('NOT FOUND — may already be patched or vllm version differs')
EOF

nohup python3 -m vllm.entrypoints.openai.api_server \
    --model <hf_model_id> --served-model-name <model_name> \
    --port 8000 --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --trust-remote-code --enforce-eager \
    > /tmp/vllm.log 2>&1 &
```

Wait for server ready:
```bash
until grep -q 'Application startup complete' /tmp/vllm.log 2>/dev/null; do sleep 5; done
echo "Server ready"
```

Verify:
```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"<model_name>","messages":[{"role":"user","content":"hi"}],"max_tokens":5}'
```

### 6. Run evaluation

**Quick 1% smoke test (MCQ datasets only):**
```bash
source ~/.gaoyao_env
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets mgsm,mmmlu,belebele,include \
    --sample-pct 1 --workers 4
```

**Per-language stratified sampling (recommended — ensures all languages represented):**
```bash
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --sample-lang-pct 10 --workers 4
```

**Full evaluation — if GPU machine has API access:**
```bash
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --workers 4
```

**Custom judge model:**
```bash
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --judge-url https://<judge_api>/v1/chat/completions \
    --judge-name <judge_model_name> \
    --datasets all --workers 4
```

**Full evaluation — if GPU machine has NO API access:**

Stage 1 on GPU machine — inference only (judge calls will fail silently; inference results are cached):
```bash
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --workers 4
```

Stage 2 on API-accessible machine — scoring only:
```bash
# Copy inference results from GPU machine first
scp -P <port> -r <user>@<host>:<gaoyao_dir>/results/<model_name> ./results/

# Run evaluation with cached inference
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --skip-inference
```

Note: Flores-101 COMET scoring requires `unbabel-comet` installed. If not available on the API machine, run Flores eval on the GPU machine with `--skip-inference` after copying the inference results there.

### 7. Interpret results

Expected output format:
```
========================================================
Dataset                   Score  Ref(paper)     Delta
--------------------------------------------------------
MGSM                     0.7985      0.7985   +0.0000
MMMLU                    0.6203      0.6203   +0.0000
...
========================================================
```

With 1% sampling, deltas of ±0.05–0.15 are normal due to small sample size.
Larger deviations (>0.2) on SAGE/CultureScope are expected — these datasets have few 1% samples (8–50 items).

## Troubleshooting

| Error | Fix |
|-------|-----|
| `no kernel image for device` (sm_61) | GPU too old, needs sm_70+ |
| `PTX compiled with unsupported toolchain` (sm_120) | Apply Blackwell patch in Step 5 |
| `Free memory < desired utilization` | Kill old GPU processes: `pkill -9 -f vllm` then retry |
| `tensorflow_text` import error | `pip install 'transformers>=4.56.0,<4.57.0'` |
| Judge timeout / API unreachable | Use two-stage approach (Step 6) |
| COMET not found | `pip install unbabel-comet` or run Flores on GPU machine |
