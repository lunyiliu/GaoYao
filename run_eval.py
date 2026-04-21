#!/usr/bin/env python3
"""
GaoYao Multilingual Evaluation — main entry point.

Quick 1% test:
  python run_eval.py --model-url http://localhost:8000/v1/chat/completions \
                     --model-name qwen3-4b \
                     --sample-pct 1

Full run:
  python run_eval.py --model-url http://localhost:8000/v1/chat/completions \
                     --model-name Qwen/Qwen3-4B-Instruct \
                     --datasets all

Reference results (Qwen3-4B-Instruct-2507 internal):
  mgsm=0.7985  mmmlu=0.6203  belebele=0.8979
  include=0.6151  flores=0.8540  s_alpaca_eval=0.1973  s_mt_bench=0.2961
  sage=0.8998  culture_scope=0.9318  superblend=0.5459
"""
import argparse
import os
import sys

# Allow running from repo root without installing
sys.path.insert(0, os.path.dirname(__file__))

from src.pipeline.pipeline import run_pipeline, print_summary, ALL_DATASETS

REFERENCE = {
    'mgsm':          0.7985,
    'mmmlu':         0.6203,
    'belebele':      0.8979,
    'include':       0.6151,
    'flores':        0.8540,
    's_alpaca_eval': 0.1973,
    's_mt_bench':    0.2961,
    'sage':          0.8998,
    'culture_scope': 0.9318,
    'superblend':    0.5459,
}


def main():
    p = argparse.ArgumentParser(description='GaoYao Evaluation Pipeline')
    p.add_argument('--model-url',    default=os.environ.get('GAOYAO_LLM_URL',
                   'http://localhost:8000/v1/chat/completions'))
    p.add_argument('--model-name',   default=os.environ.get('GAOYAO_LLM_NAME', 'qwen3-4b'))
    p.add_argument('--judge-url',    default=None,
                   help='Judge LLM URL (defaults to --model-url)')
    p.add_argument('--judge-name',   default=None,
                   help='Judge model name (defaults to --model-name)')
    p.add_argument('--data-dir',     default='data/original')
    p.add_argument('--output-dir',   default='results')
    p.add_argument('--datasets',     default='all',
                   help=f'Comma-separated list or "all". Available: {", ".join(ALL_DATASETS)}')
    p.add_argument('--sample-pct',   type=float, default=100,
                   help='Randomly sample X%% of each dataset (0<x<=100). Default 100.')
    p.add_argument('--workers',      type=int, default=4,
                   help='Concurrent inference workers per dataset')
    p.add_argument('--skip-inference', action='store_true',
                   help='Skip inference if infer.jsonl already exists')
    p.add_argument('--no-ref',       action='store_true',
                   help='Do not show reference comparison')
    args = p.parse_args()

    # Propagate judge settings to env so llm_request picks them up
    if args.judge_url:
        os.environ['GAOYAO_JUDGE_URL']  = args.judge_url
    if args.judge_name:
        os.environ['GAOYAO_JUDGE_NAME'] = args.judge_name
    os.environ['GAOYAO_LLM_URL']  = args.model_url
    os.environ['GAOYAO_LLM_NAME'] = args.model_name

    datasets = (ALL_DATASETS if args.datasets.strip().lower() == 'all'
                else [d.strip() for d in args.datasets.split(',')])

    output_dir = os.path.join(args.output_dir, args.model_name.replace('/', '-'))

    summary = run_pipeline(
        data_dir       = args.data_dir,
        output_dir     = output_dir,
        model_url      = args.model_url,
        model_name     = args.model_name,
        datasets       = datasets,
        sample_pct     = args.sample_pct,
        infer_workers  = args.workers,
        skip_inference = args.skip_inference,
    )

    print_summary(summary, reference=(None if args.no_ref else REFERENCE))


if __name__ == '__main__':
    main()
