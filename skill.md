# GaoYao Multilingual Evaluation — Agent Skill

Deploy and run the GaoYao multilingual benchmark on a remote GPU machine end-to-end.

## When to use

When a user asks to evaluate a model on GaoYao, benchmark multilingual performance, or reproduce GaoYao paper results.

## Prerequisites

- Remote GPU machine with SSH access (CUDA sm_70+: V100, RTX 20xx/30xx/40xx/50xx series)
- `MADE_API_KEY` environment variable — required for judge calls (DeepSeek-V3.1) and MMMLU fallback
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
git clone https://github.com/lunyiliu/GaoYao.git /root/GaoYao
cd /root/GaoYao
pip install -r requirements.txt -q
pip install vllm -q
```

If `transformers` version conflict (common on pre-configured machines):
```bash
pip install 'transformers>=4.56.0,<4.57.0' -q
```

### 3. Configure environment

```bash
echo "export MADE_API_KEY=<key>" > /root/.gaoyao_env && chmod 600 /root/.gaoyao_env
source /root/.gaoyao_env
```

Also copy `/root/MADE/llm_client.py` from the MADE project to the remote machine if not present.

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
path = '/usr/local/lib/python3.12/dist-packages/vllm/platforms/cuda.py'
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
cd /root/GaoYao && source /root/.gaoyao_env
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets mgsm,mmmlu,belebele,include \
    --sample-pct 1 --workers 4
```

**Full evaluation — if GPU machine has API access:**
```bash
python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --workers 4
```

**Full evaluation — if GPU machine has NO API access (common for overseas machines):**

Stage 1 on GPU machine — inference only:
```bash
# Run all datasets; inference is cached, judge calls will fail but that's OK
# Instead, use the inference-only script:
python3 - << 'PYEOF'
import sys, os; sys.path.insert(0, '/root/GaoYao'); os.chdir('/root/GaoYao')
from src.tools.file_operations import read_jsonl
from src.inference.infer import run_inference
REGISTRY = {
    'mgsm':         ('data/original/MGSM',        'mgsm'),
    'mmmlu':        ('data/original/MMMLU',        'mmmlu'),
    'belebele':     ('data/original/Belebele',     'belebele'),
    'include':      ('data/original/Include',      'include'),
    'flores':       ('data/original/Flores101',    'flores'),
    's_alpaca_eval':('data/original/S-AlpacaEval', 's_alpaca_eval'),
    's_mt_bench':   ('data/original/S-MT-Bench',   's_mt_bench'),
    'sage':         ('data/original/SAGE',         'sage'),
    'culture_scope':('data/original/CultureScope', 'culture_scope'),
    'superblend':   ('data/original/SuperBLEnD',   'superblend'),
}
for ds, (path, name) in REGISTRY.items():
    out = f'results/<model_name>/inference_result/{ds}/infer.jsonl'
    if os.path.exists(out): print(f'{ds}: cached'); continue
    os.makedirs(os.path.dirname(out), exist_ok=True)
    data = read_jsonl(path)
    run_inference(data, out, 'http://localhost:8000/v1/chat/completions',
                  '<model_name>', sample_pct=1, workers=4, dataset_name=name)
    print(f'{ds}: done')
PYEOF
```

Stage 2 on API-accessible machine — evaluation only:
```bash
# Copy results/<model_name>/inference_result/ from GPU machine first
scp -P <port> -r <user>@<host>:/root/GaoYao/results/<model_name> ./results/

python3 run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all --skip-inference
```

Note: Flores-101 COMET scoring requires `unbabel-comet` installed locally. If not available, run Flores eval on the GPU machine with `--skip-inference` after copying the inference results there.

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
| Judge timeout from overseas machine | Use two-stage approach (Step 6) |
| COMET not found | `pip install unbabel-comet` or run Flores on GPU machine |
