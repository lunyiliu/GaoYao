"""Inference module: runs the target model over a dataset split and writes inference_result JSONL."""
import os
import json
import random
import concurrent.futures
from src.tools.llm_request import send_inference
from src.tools.file_operations import read_jsonl, write_jsonl
from src.log.logging_config import logger


SYSTEM_PROMPT = "You are a helpful multilingual assistant."

MT_BENCH_SYSTEM = (
    "You are a helpful, respectful, and honest assistant. "
    "Always answer as helpfully as possible."
)


def _infer_single(item, model_url, model_name, params):
    prompt = item.get('prompt', '')
    history = item.get('_history', [])  # injected by MT-Bench multi-turn handling

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    response = send_inference(messages, model_name=model_name, model_url=model_url, params=params)
    out = dict(item)
    out.pop('_history', None)
    out['response'] = response
    return out


def run_inference(data_list, output_path, model_url, model_name,
                  sample_pct=100, workers=4, params=None, dataset_name=''):
    """
    Run inference on data_list and write results to output_path.
    Returns list of items with 'response' field.
    """
    if os.path.exists(output_path):
        logger.info(f"  Inference cache found: {output_path}, loading.")
        return read_jsonl(output_path)

    # Sampling
    if sample_pct < 100:
        k = max(1, int(len(data_list) * sample_pct / 100))
        data_list = random.sample(data_list, k)
        logger.info(f"  Sampled {k} items for inference ({dataset_name})")

    # MT-Bench multi-turn: group by source_id, handle round ordering
    if dataset_name in ('s_mt_bench', 'mt_bench'):
        data_list = _handle_mt_bench(data_list, model_url, model_name, params)
        write_jsonl(output_path, data_list)
        return data_list

    results = []
    logger.info(f"  Running inference: {len(data_list)} items, {workers} workers [{dataset_name}]")

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(_infer_single, item, model_url, model_name, params): i
            for i, item in enumerate(data_list)
        }
        done = 0
        for f in concurrent.futures.as_completed(futures):
            results.append(f.result())
            done += 1
            if done % 50 == 0:
                logger.info(f"    {done}/{len(data_list)} done")

    write_jsonl(output_path, results)
    logger.info(f"  Inference written: {output_path}")
    return results


def _handle_mt_bench(data_list, model_url, model_name, params):
    """
    MT-Bench has round_number 1 and 2.
    Run round 1 first, then use response as history for round 2.
    """
    from collections import defaultdict
    by_src = defaultdict(dict)
    for item in data_list:
        src_id = item.get('source_id', item.get('uuid'))
        rnd    = item.get('round_number', 1)
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
                item2['_history'] = [
                    {"role": "user",      "content": item1.get('prompt', '')},
                    {"role": "assistant", "content": r1.get('response', '')},
                ]
                results.append(_infer_single(item2, model_url, model_name, params))
        elif item2:
            results.append(_infer_single(item2, model_url, model_name, params))

    return results
