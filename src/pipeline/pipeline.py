"""
GaoYao Evaluation Pipeline
Usage: see run_eval.py
"""
import os
import json
from decimal import Decimal
from src.log.logging_config import logger
from src.tools.file_operations import read_jsonl
from src.inference.infer import run_inference

# Dataset registry: name → (eval_class, data_subdir, inference_workers)
DATASET_REGISTRY = {
    'mgsm':         ('src.evaluation.mgsm_eval',         'MGSM',          'MGSM',           4),
    'mmmlu':        ('src.evaluation.mmmlu_eval',         'MMMLU',         'MMMLU',          4),
    'belebele':     ('src.evaluation.belebele_eval',      'Belebele',      'Belebele',       4),
    'include':      ('src.evaluation.include_eval',       'Include',       'Include',        4),
    'flores':       ('src.evaluation.flores_eval',        'Flores',        'Flores101',      4),
    'superblend':   ('src.evaluation.superblend_eval',    'SuperBlend',    'SuperBLEnD',     2),
    's_alpaca_eval':('src.evaluation.s_alpaca_eval',      'AlpacaEval',    'S-AlpacaEval',   2),
    's_mt_bench':   ('src.evaluation.s_mt_bench_eval',    'MTBench',       'S-MT-Bench',     1),
    'sage':         ('src.evaluation.sage_eval',          'SAGE',          'CrossCultural',  2),
    'culture_scope':('src.evaluation.culture_scope_eval', 'CulturalScope', 'MonoCultural',   2),
}

ALL_DATASETS = list(DATASET_REGISTRY.keys())


def _import_eval_class(module_path, class_name):
    import importlib
    mod = importlib.import_module(module_path)
    return getattr(mod, class_name)


def _mean(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0
    return float(Decimal(str(sum(vals))) / Decimal(str(len(vals))))


def _dataset_mean(metrics) -> float:
    """Compute per-dataset aggregate score from eval output."""
    if isinstance(metrics, dict):
        # SuperBLEnD returns {'language': {...}, 'country': {...}}
        if 'language' in metrics and isinstance(metrics['language'], dict):
            return _mean(list(metrics['language'].values()))
        # calc_acc_by_language returns {'language': [...], 'accuracy': [...]}
        if 'accuracy' in metrics:
            return _mean(metrics['accuracy'])
        # flores / win-rate: {lang: score}
        return _mean(list(metrics.values()))
    return 0.0


def run_pipeline(
    data_dir: str,
    output_dir: str,
    model_url: str,
    model_name: str,
    datasets=None,
    sample_pct: float = 100,
    infer_workers: int = 4,
    skip_inference: bool = False,
):
    if datasets is None or datasets == ['all']:
        datasets = ALL_DATASETS

    os.makedirs(output_dir, exist_ok=True)
    summary = {}

    for ds_name in datasets:
        if ds_name not in DATASET_REGISTRY:
            logger.warning(f"Unknown dataset: {ds_name}, skipping")
            continue

        module_path, class_name, data_subdir, default_workers = DATASET_REGISTRY[ds_name]
        workers = infer_workers or default_workers

        data_path = os.path.join(data_dir, data_subdir)
        if not os.path.exists(data_path):
            logger.warning(f"Data not found: {data_path}, skipping {ds_name}")
            continue

        logger.info(f"\n{'='*50}")
        logger.info(f"Dataset: {ds_name.upper()}")

        # Load original data
        raw_data = read_jsonl(data_path)
        if not raw_data:
            logger.warning(f"No data loaded from {data_path}")
            continue
        logger.info(f"  Loaded {len(raw_data)} items")

        # Inference
        infer_path = os.path.join(output_dir, 'inference_result', ds_name, 'infer.jsonl')
        os.makedirs(os.path.dirname(infer_path), exist_ok=True)

        if skip_inference and os.path.exists(infer_path):
            logger.info(f"  Skipping inference (--skip-inference), loading {infer_path}")
            inferred = read_jsonl(infer_path)
        else:
            inferred = run_inference(
                raw_data, infer_path, model_url, model_name,
                sample_pct=sample_pct, workers=workers, dataset_name=ds_name,
            )

        if not inferred:
            logger.warning(f"  No inference results for {ds_name}")
            continue

        # Evaluation
        EvalClass = _import_eval_class(module_path, class_name)
        evaluator = EvalClass()

        eval_dir   = os.path.join(output_dir, 'evaluation_result', ds_name)
        os.makedirs(eval_dir, exist_ok=True)
        badcase_path  = os.path.join(eval_dir, 'bad_cases.jsonl')
        not_pass_path = os.path.join(eval_dir, 'not_pass.jsonl')

        try:
            metrics = evaluator.evaluate(inferred)
            evaluator.badcases and __import__('src.tools.file_operations',
                fromlist=['write_jsonl']).write_jsonl(badcase_path, evaluator.badcases)
            evaluator.not_pass and __import__('src.tools.file_operations',
                fromlist=['write_jsonl']).write_jsonl(not_pass_path, evaluator.not_pass)
        except Exception as e:
            logger.error(f"  Eval error for {ds_name}: {e}", exc_info=True)
            continue

        # Save metrics
        metrics_path = os.path.join(eval_dir, 'metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

        agg = _dataset_mean(metrics)
        summary[ds_name] = agg
        logger.info(f"  {ds_name}: {agg:.4f}")

    return summary


def print_summary(summary: dict, reference: dict = None):
    header = f"{'Dataset':<20} {'Score':>8}"
    if reference:
        header += f"  {'Ref':>8}  {'Delta':>8}"
    print('\n' + '='*50)
    print(header)
    print('-'*50)
    for ds, score in summary.items():
        row = f"{ds:<20} {score:>8.4f}"
        if reference and ds in reference:
            ref = reference[ds]
            row += f"  {ref:>8.4f}  {score-ref:>+8.4f}"
        print(row)
    print('='*50)
