# ⚖️ GaoYao — Multilingual LLM Evaluation Benchmark

<p align="right">
  <b>English</b> &nbsp;|&nbsp; <a href="./README.md">简体中文</a>
</p>

![Status](https://img.shields.io/badge/Status-Active-success)
![Task](https://img.shields.io/badge/Task-Multilingual_Evaluation-blue)
![Language](https://img.shields.io/badge/Language-python-orange)
![ACL 2026](https://img.shields.io/badge/ACL-2026-red)

> **🌐 A one-stop multilingual · multicultural · multi-format LLM capability evaluation framework**
>
> GaoYao aims to build a fair and comprehensive evaluation system supporting **multiple-choice**, **subjective**, and **translation** tasks, covering the full capability spectrum from basic understanding to cross-cultural adaptation.

---

## 🎉 GaoYao is accepted by ACL 2026 main!

---

<p align="center">
  <img src="./assets/gaoyao_coverage_map.png" width="85%" alt="GaoYao Language & Culture Coverage — 51 nations/areas, 26 languages" />
</p>

<p align="center"><em>GaoYao covers 26 languages across 51 nations/areas with balanced resource distribution</em></p>

---

## 🔗 Resources

- 📄 **Paper**: [arXiv:2604.20225](https://arxiv.org/abs/2604.20225)
- 🤖 **Agentic Skill (one-click deploy)**: [skill.md](./skill.md) — AI agent end-to-end deployment guide, from GPU verification to full evaluation

---

![](./assets/cover_iamge.png)

---

## 🤖 Agentic Skill Design

GaoYao is packaged as a **standardized Agent Skill**, supporting end-to-end one-click deployment and evaluation by AI agents.

- **[skill.md](./skill.md)** provides a structured 7-step guide: GPU verification → dependency install → environment config → vLLM launch → evaluation run → result interpretation
- Supports both **standard GPUs (sm_70+)** and **Blackwell GPUs (sm_120, RTX 50 series)** with an auto-applied compatibility patch
- Supports **two-stage evaluation**: inference and scoring are decoupled, suitable for GPU machines without judge-API access
- Works with any Agent framework that supports Claude Code / OpenAI Function Calling

---

## Project Overview

GaoYao is a systematic multilingual and multicultural evaluation benchmark, organized as a three-dimensional nine-subtask framework covering general multilingual capability, cross-cultural capability, and mono-cultural capability. It extends instruction-following and multi-turn dialogue test sets to 19 languages via language-expert human translation, expanding language coverage by 111% over prior work. For cultural capability, an expert-in-the-loop data synthesis method covers 34 cultures, improving cultural representativeness by 88%. The benchmark integrates high-quality human-verified data to avoid quality deficiencies of pure machine translation. We conduct stratified evaluation on 15+ mainstream open-source and commercial LLMs, providing a comprehensive and reliable measurement of multilingual and multicultural capability.

---

## 📁 Code Layout

```
GaoYaoEval/
├── data/                          # Data layer
│   └── original/                  # Raw evaluation datasets
│       ├── MGSM/
│       ├── MMMLU/
│       ├── Belebele/
│       ├── Include/
│       ├── Flores101/
│       ├── SuperBLEnD/
│       ├── S-AlpacaEval/
│       ├── S-MT-Bench/
│       ├── SAGE/
│       └── CultureScope/
│
├── results/                       # Output directory (created at runtime)
│   └── <model-name>/
│       ├── inference_result/      # Inference cache
│       │   └── <dataset>/infer.jsonl
│       └── evaluation_result/     # Analysis results
│           └── <dataset>/
│               ├── metrics.json   # Metric summary
│               ├── bad_cases.jsonl
│               └── not_pass.jsonl
│
├── src/                           # Core code
│   ├── evaluation/                # Evaluation engines
│   │   ├── base_eval.py           # Base class
│   │   ├── config.py              # Evaluation config
│   │   └── {dataset}_eval.py      # Per-dataset evaluator (10 files)
│   ├── inference/
│   │   └── infer.py               # Inference (vLLM API + local transformers)
│   ├── pipeline/
│   │   └── pipeline.py            # Evaluation pipeline controller
│   ├── tools/                     # Common utilities
│   │   ├── file_operations.py
│   │   ├── judger_algorithm.py
│   │   ├── llm_request.py         # Inference / judge / COMET request wrappers
│   │   ├── metrics_and_report_operator.py
│   │   ├── prompt_templates.py
│   │   └── text_processing.py
│   └── log/
│       └── logging_config.py
│
├── scripts/                       # One-click deployment scripts (used by skill.md)
│   ├── bootstrap.sh               # Idempotent environment setup (deps / models / patch all at once)
│   ├── check_gpu.sh               # GPU capability gate
│   ├── patch_vllm_blackwell.py    # sm_120 vLLM compatibility patch
│   ├── start_vllm.sh              # vLLM launcher (auto sm_120 patch)
│   ├── wait_vllm.sh               # Wait for vLLM readiness
│   └── run_preset.sh              # Evaluation preset wrapper (smoke/lang10/full/stage1/stage2)
│
├── run_eval.py                    # Main entry point
├── skill.md                       # Agent one-click deployment guide
├── requirements.txt
└── LICENSE
```

---

## 📊 Data Schema

### 📋 Inference fields (`inference_result/<dataset>/infer.jsonl`)

| # | Field      | Definition                | Required | Example         |
|---|------------|---------------------------|----------|-----------------|
| 1 | `uuid`     | Unique identifier         | ✅       | `"uuid-001"`    |
| 2 | `prompt`   | Question / prompt         | ✅       | `"Capital of France?"` |
| 3 | `response` | Model inference output    | ✅       | `"Paris"`       |
| 4 | `gt`       | Reference answer          | ✅       | `"Paris"`       |
| 5 | `language` | Language code             | ❌       | `"fr"`          |
| 6 | `country`  | Region / country          | ❌       | `"France"`      |

### 📈 Evaluation extension fields (`evaluation_result/<dataset>/`)

| #   | Field         | Definition                              | Task type    | Required     |
|-----|---------------|-----------------------------------------|--------------|--------------|
| +1  | `judge_score` | Evaluation score                        | Objective    | Objective ✅ |
| +2  | `prediction`  | Normalized extracted answer             | Objective    | Objective ✅ |
| +3  | `winner`      | Pairwise result (`win` / `lose` / `tie`) | Subjective   | Subjective ✅ |
| +4  | `bad_case`    | Raw abnormal response                   | All          | ❌           |
| +5  | `not_pass`    | Raw response of failing cases           | All          | ❌           |

> Evaluation results automatically inherit all inference fields and append the above extension fields.

---

## ⚙️ Quickstart

### Environment setup

```bash
git clone https://github.com/lunyiliu/GaoYao.git
cd GaoYao
pip install -r requirements.txt
pip install vllm          # requires CUDA sm_70+ (V100 / RTX 20 series or newer)
```

### Configure environment variables

```bash
export GAOYAO_API_KEY=<your_api_key>          # required for judge calls
export GAOYAO_JUDGE_MODEL=<judge_model_name>  # judge model name
export GAOYAO_JUDGE_BASE_URL=<judge_api_url>  # judge API endpoint
```

### Launch the inference server

```bash
# Standard GPU (sm_70+)
bash scripts/start_vllm.sh <hf_model_id>

# Blackwell GPU (sm_120, RTX 50 series) — patch vllm first, then launch
python3 scripts/patch_vllm_blackwell.py
nohup python3 -m vllm.entrypoints.openai.api_server \
    --model <hf_model_id> --served-model-name <model_name> \
    --port 8000 --dtype float16 --max-model-len 8192 \
    --gpu-memory-utilization 0.85 --trust-remote-code --enforce-eager \
    > /tmp/vllm.log 2>&1 &
```

### Run evaluation

```bash
# 1% smoke test (random sampling)
python run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --sample-pct 1

# 10% per-language stratified sampling (each language sampled at 10%,
# avoiding low-resource languages being dropped)
python run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --sample-lang-pct 10

# Full run
python run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets all

# Selected datasets
python run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --datasets mgsm,mmmlu,belebele,include
```

### Custom judge model

The judge model is used for subjective scoring (S-AlpacaEval, S-MT-Bench) and as an MMMLU answer-extraction fallback. Configure via environment variables, or override per-run via CLI:

```bash
python run_eval.py \
    --model-url http://localhost:8000/v1/chat/completions \
    --model-name <model_name> \
    --judge-url https://<your_judge_api>/v1/chat/completions \
    --judge-name <judge_model_name>
```

Or via environment:

```bash
export GAOYAO_JUDGE_BASE_URL=https://<your_judge_api>/v1
export GAOYAO_JUDGE_MODEL=<judge_model_name>
export GAOYAO_API_KEY=<your_api_key>
```

### CLI arguments

| Argument           | Default                                        | Description                                        |
|--------------------|------------------------------------------------|----------------------------------------------------|
| `--model-url`      | `http://localhost:8000/v1/chat/completions`    | Inference endpoint                                 |
| `--model-name`     | —                                              | Model name (required)                              |
| `--judge-url`      | same as `--model-url`                          | Judge LLM endpoint                                 |
| `--judge-name`     | same as `--model-name`                         | Judge model name                                   |
| `--data-dir`       | `data/original`                                | Dataset directory                                  |
| `--output-dir`     | `results`                                      | Output directory                                   |
| `--datasets`       | `all`                                          | Comma-separated dataset list or `all`              |
| `--sample-pct`     | `100`                                          | Random sampling percentage (1–100)                 |
| `--sample-lang-pct`| —                                              | Per-language stratified sampling percentage (takes precedence over `--sample-pct`) |
| `--workers`        | `4`                                            | Concurrent inference workers                       |
| `--skip-inference` | off                                            | Skip inference (use cached results)                |
| `--no-ref`         | off                                            | Hide reference (paper) scores in output            |

### Custom evaluator

```python
from src.evaluation.base_eval import BaseEval

class MyDatasetEval(BaseEval):
    def evaluate(self, data_list: list) -> dict:
        # implement evaluation logic, return a metrics dict
        pass
```

---

## 📊 Dataset Overview

| ID | Dataset | Type | Core Capability | Dimension |
|:--:|:--------|:----:|:----------------|:----------|
| **01** | `belebele`      | 🧩 Objective | Multilingual reading comprehension | **Reading Comprehension** |
| **02** | `mgsm`          | 🧩 Objective | Multilingual math reasoning        | **Math**                  |
| **03** | `mmmlu`         | 🧩 Objective | Multi-discipline knowledge         | **Reasoning**             |
| **04** | `superblend`    | 🧩 Objective | Mixed-domain general ability       | **Cross-Culture**         |
| **05** | `include`       | 🧩 Objective | Cultural inclusiveness             | **Knowledge**             |
| **06** | `culture_scope` | ⚖️ Hybrid    | Mono-cultural in-depth evaluation  | **Mono-Culture**          |
| **07** | `sage`          | ⚖️ Hybrid    | Cross-cultural understanding       | **Cross-Culture**         |
| **08** | `s_alpaca_eval` | 🖋️ Subjective | Complex instruction following     | **Instruction Follow**    |
| **09** | `s_mt_bench`    | 🖋️ Subjective | Multi-turn dialogue quality        | **Dialogue**              |
| **10** | `flores`        | 🔄 Translation | High-quality machine translation | **Translation**           |

---

## 💡 Extensibility

- **Pluggable evaluators**: inherit `BaseEval` to add a new dataset in minutes
- **Unified data contract**: standardized input/output fields lower integration cost
- **Replaceable judge**: swap any OpenAI-compatible judge model via `--judge-url` / `--judge-name` or environment variables
- **Stratified sampling**: `--sample-lang-pct` ensures low-resource languages are not dropped — ideal for fast validation

---

## 📚 Citation

If this work is useful for your research, please cite our paper:

```bibtex
@misc{liu2026gaoyaobenchmarkcomprehensiveframework,
      title={The GaoYao Benchmark: A Comprehensive Framework for Evaluating Multilingual and Multicultural Abilities of Large Language Models}, 
      author={Yilun Liu and Chunguang Zhao and Mengyao Piao and Lingqi Miao and Shimin Tao and Minggui He and Chenxin Liu and Li Zhang and Hongxia Ma and Jiaxin Guo and Chen Liu and Liqun Deng and Jiansheng Wei and Xiaojun Meng and Fanyi Du and Daimeng Wei and Yanghua Xiao},
      year={2026},
      eprint={2604.20225},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2604.20225}, 
}
```
