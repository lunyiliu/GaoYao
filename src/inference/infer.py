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
            from transformers import pipeline, AutoTokenizer
            logger.info(f"  Loading local model: {model_id}")
            tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            _local_pipeline = pipeline(
                "text-generation",
                model=model_id,
                tokenizer=tok,
                device_map="auto",
                torch_dtype="float16",
                trust_remote_code=True,
                max_new_tokens=2048,
            )
            logger.info("  Local model loaded.")
    return _local_pipeline


def _infer_local(messages: list, model_id: str) -> str:
    pipe = _get_local_pipeline(model_id)
    try:
        out = pipe(messages, return_full_text=False)
        return out[0]["generated_text"]
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
