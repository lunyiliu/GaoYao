"""
LLM request helpers for GaoYao evaluation.

Inference calls (target model): use vLLM local endpoint via send_inference().
Judge calls (DS-V3.1): use MADE ArchivedLLMClient via send_chat_completion().
  - API key from MADE_API_KEY env var (never hardcoded)
  - Every call archived to /root/MADE/api_archives/
"""
import os
import sys
import logging
import requests

logger = logging.getLogger('eval_logger')

# Default models (overridable via env)
_DEFAULT_JUDGE_MODEL = os.environ.get('GAOYAO_JUDGE_MODEL', 'deepseek-v3.1-terminus')
_DEFAULT_LLM_URL     = os.environ.get('GAOYAO_LLM_URL',    'http://localhost:8000/v1/chat/completions')
_DEFAULT_LLM_NAME    = os.environ.get('GAOYAO_LLM_NAME',   'qwen3-4b')
_COMET_MODEL         = os.environ.get('GAOYAO_COMET_MODEL', 'Unbabel/wmt22-comet-da')


def _get_made_client(model: str = None, caller: str = 'gaoyao_judge'):
    """Return an ArchivedLLMClient from the MADE project using MADE_API_KEY."""
    made_path = os.environ.get('MADE_ROOT', '/root/MADE')
    if made_path not in sys.path:
        sys.path.insert(0, made_path)
    from llm_client import client_from_env, LLMClientConfig, ArchivedLLMClient
    if model:
        api_key  = os.environ['MADE_API_KEY']
        base_url = os.environ.get('MADE_BASE_URL', 'https://yibuapi.com/v1')
        archive  = os.environ.get('MADE_ARCHIVE_DIR', '/root/MADE/api_archives')
        return ArchivedLLMClient(LLMClientConfig(
            api_key=api_key, base_url=base_url, model=model,
            temperature=0.1, max_tokens=1024,
            archive_dir=archive, verify_ssl=False, caller=caller,
        ))
    return client_from_env(caller=caller)


def send_chat_completion(system_prompt, user_prompt,
                         model_name=None, model_url=None, params=None):
    """
    Judge call: uses DS-V3.1 via yibuapi / ArchivedLLMClient.
    model_url is ignored (always uses MADE API path).
    model_name overrides the judge model.
    """
    judge_model = model_name or _DEFAULT_JUDGE_MODEL
    try:
        client = _get_made_client(model=judge_model, caller='gaoyao_judge')
        result = client.chat([
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ])
        if result.get('error'):
            logger.error(f"Judge error: {result['error']}")
            return ''
        return result.get('text') or ''
    except Exception as e:
        logger.error(f"Judge LLM failed: {e}")
        return ''


def send_inference(messages, model_name=None, model_url=None, params=None):
    """Inference call: local vLLM endpoint (not MADE API)."""
    url  = model_url  or _DEFAULT_LLM_URL
    name = model_name or _DEFAULT_LLM_NAME
    try:
        data = {"model": name, "messages": messages,
                "temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
        if params:
            data.update(params)
        res = requests.post(url, headers={'Content-Type': 'application/json'},
                            json=data, timeout=300)
        if res.status_code != 200:
            logger.error(f"Inference failed {res.status_code}: {res.text[:200]}")
            return ''
        return res.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Inference error: {e}")
        return ''


def compute_comet_local(src_lines, ref_lines, mt_lines, model_name=None):
    """COMET-22 scoring via unbabel-comet, returns raw score (0-1)."""
    cmodel = model_name or _COMET_MODEL
    try:
        from comet import download_model, load_from_checkpoint
        model_path = download_model(cmodel)
        model = load_from_checkpoint(model_path)
        data = [{"src": s, "mt": m, "ref": r}
                for s, m, r in zip(src_lines, mt_lines, ref_lines)]
        out = model.predict(data, batch_size=8, gpus=0)
        return float(out.system_score)
    except Exception as e:
        logger.error(f"COMET local error: {e}")
        return 0.0


def send_comet_req(request_id, src_lines, ref_lines, mt_lines):
    """COMET scoring: local compute (wmt22-comet-da, raw score 0-1)."""
    return compute_comet_local(src_lines, ref_lines, mt_lines)
