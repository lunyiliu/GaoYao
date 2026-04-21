"""
Inference module for GaoYao evaluation.

Two backends:
  - vllm / OpenAI-compatible API  (--model-url http://...)
  - local transformers pipeline   (--model-url local:///<hf_model_id>)
    Falls back automatically when model_url starts with 'local://'.
"""
import os
import json
import random
import concurrent.futures
from src.tools.llm_request import send_inference
from src.tools.file_operations import read_jsonl, write_jsonl
from src.log.logging_config import logger

SYSTEM_PROMPT = "You are a helpful multilingual assistant."

# ── Local transformers backend ─────────────────────────────────────────────────

import threading
_local_pipeline = None
_local_lock = threading.Lock()

def _get_local_pipeline(model_id: str):
    global _local_pipeline
    if _local_pipeline is not None:
        return _local_pipeline
    with _local_lock:
        if _local_pipeline is None:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            logger.info(f"  Loading local model: {model_id}")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            # Use CUDA if available (even sm_61), else CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"  Using device: {device}")
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
            ).to(device)
            model.eval()
            _local_pipeline = (model, tok, device)
            logger.info("  Local model loaded.")
    return _local_pipeline


def _infer_local(messages: list, model_id: str) -> str:
    model, tok, device = _get_local_pipeline(model_id)
    try:
        import torch
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )
        new_tokens = output_ids[0][input_ids.shape[-1]:]
        return tok.decode(new_tokens, skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Local inference error: {e}")
        return ""

# ── Unified single-item inference ─────────────────────────────────────────────

def _infer_single(item, model_url, model_name, params):
    prompt  = item.get("prompt", "")
    history = item.get("_history", [])

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    if model_url.startswith("local://"):
        local_id = model_url[len("local://"):].lstrip("/")
        response = _infer_local(messages, local_id)
    else:
        response = send_inference(messages, model_name=model_name,
                                  model_url=model_url, params=params)
    out = dict(item)
    out.pop("_history", None)
    out["response"] = response
    return out

# ── Public API ─────────────────────────────────────────────────────────────────

def run_inference(data_list, output_path, model_url, model_name,
                  sample_pct=100, workers=4, params=None, dataset_name=""):
    if os.path.exists(output_path):
        logger.info(f"  Inference cache found: {output_path}, loading.")
        return read_jsonl(output_path)

    if sample_pct < 100:
        k = max(1, int(len(data_list) * sample_pct / 100))
        data_list = random.sample(data_list, k)
        logger.info(f"  Sampled {k} items ({dataset_name})")

    # local backend is single-threaded (GPU memory)
    if model_url.startswith("local://"):
        workers = 1

    if dataset_name in ("s_mt_bench", "mt_bench"):
        results = _handle_mt_bench(data_list, model_url, model_name, params)
        write_jsonl(output_path, results)
        return results

    results = []
    logger.info(f"  Inference: {len(data_list)} items, {workers} workers [{dataset_name}]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_infer_single, item, model_url, model_name, params): i
            for i, item in enumerate(data_list)
        }
        done = 0
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 20 == 0:
                logger.info(f"    {done}/{len(data_list)} done")

    write_jsonl(output_path, results)
    logger.info(f"  Written: {output_path}")
    return results


def _handle_mt_bench(data_list, model_url, model_name, params):
    from collections import defaultdict
    by_src = defaultdict(dict)
    for item in data_list:
        src_id = item.get("source_id", item.get("uuid"))
        rnd    = item.get("round_number", 1)
        by_src[src_id][rnd] = item

    results = []
    for src_id, rounds in by_src.items():
        item1 = rounds.get(1)
        item2 = rounds.get(2)
        if item1:
            r1 = _infer_single(item1, model_url, model_name, params)
            results.append(r1)
            if item2:
                item2 = dict(item2)
                item2["_history"] = [
                    {"role": "user",      "content": item1.get("prompt", "")},
                    {"role": "assistant", "content": r1.get("response", "")},
                ]
                results.append(_infer_single(item2, model_url, model_name, params))
        elif item2:
            results.append(_infer_single(item2, model_url, model_name, params))
    return results
