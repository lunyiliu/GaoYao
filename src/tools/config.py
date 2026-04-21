import os

# Override via env vars: GAOYAO_LLM_URL, GAOYAO_LLM_NAME, GAOYAO_JUDGE_URL, GAOYAO_JUDGE_NAME
LLM_URL  = os.environ.get('GAOYAO_LLM_URL',   'http://localhost:8000/v1/chat/completions')
LLM_NAME = os.environ.get('GAOYAO_LLM_NAME',  'qwen3-4b')

JUDGE_URL  = os.environ.get('GAOYAO_JUDGE_URL',   LLM_URL)
JUDGE_NAME = os.environ.get('GAOYAO_JUDGE_NAME',  LLM_NAME)

COMET_MODEL = os.environ.get('GAOYAO_COMET_MODEL', 'Unbabel/wmt22-comet-da')

# Legacy alias kept for backward compat
COMET_URL = os.environ.get('GAOYAO_COMET_URL', '')

SAMPLING_CONFIGS = {
    "thinking_default":    {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "max_tokens": 16384},
    "non_thinking_default":{"temperature": 0.7, "top_p": 0.8,  "top_k": 20, "max_tokens": 8192},
}
