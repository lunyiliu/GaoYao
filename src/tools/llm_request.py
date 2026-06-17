"""
LLM request helpers for GaoYao evaluation.

The target (tested) model and the judge model are separately configured:

  Target model — send_inference()
    GAOYAO_LLM_URL       endpoint (default: local vLLM)
    GAOYAO_LLM_NAME      model name
    GAOYAO_LLM_API_KEY   Bearer key (optional; only needed if the endpoint
                         requires auth — e.g. a hosted OpenAI-compatible API)

  Judge model — send_chat_completion()
    GAOYAO_JUDGE_BASE_URL endpoint
    GAOYAO_JUDGE_MODEL    model name
    GAOYAO_JUDGE_API_KEY  Bearer key (falls back to GAOYAO_API_KEY for
                          back-compat; never hardcoded)
"""
import os
import sys
import logging
import requests

logger = logging.getLogger('eval_logger')

# Default models (overridable via env)
_DEFAULT_JUDGE_MODEL = os.environ.get('GAOYAO_JUDGE_MODEL', '')
_DEFAULT_LLM_URL     = os.environ.get('GAOYAO_LLM_URL',    'http://localhost:8000/v1/chat/completions')
_DEFAULT_LLM_NAME    = os.environ.get('GAOYAO_LLM_NAME',   '')
_COMET_MODEL         = os.environ.get('GAOYAO_COMET_MODEL', 'Unbabel/wmt22-comet-da')


def _get_judge_client(model: str = None, caller: str = 'gaoyao_judge'):
    """Return an ArchivedLLMClient for judge calls."""
    judge_root = os.environ.get('GAOYAO_JUDGE_ROOT', '')
    if judge_root and judge_root not in sys.path:
        sys.path.insert(0, judge_root)
    from llm_client import client_from_env, LLMClientConfig, ArchivedLLMClient
    if model:
        api_key  = os.environ.get('GAOYAO_JUDGE_API_KEY') or os.environ.get('GAOYAO_API_KEY')
        if not api_key:
            raise RuntimeError(
                "judge API key not set — export GAOYAO_JUDGE_API_KEY "
                "(or GAOYAO_API_KEY for backwards compatibility)"
            )
        base_url = os.environ.get('GAOYAO_JUDGE_BASE_URL', 'https://api.openai.com/v1')
        archive  = os.environ.get('GAOYAO_ARCHIVE_DIR', os.path.join(judge_root or '.', 'api_archives'))
        return ArchivedLLMClient(LLMClientConfig(
            api_key=api_key, base_url=base_url, model=model,
            temperature=0.1, max_tokens=1024,
            archive_dir=archive, verify_ssl=False, caller=caller,
        ))
    return client_from_env(caller=caller)


def send_chat_completion(system_prompt, user_prompt,
                         model_name=None, model_url=None, params=None):
    """
    Judge call via ArchivedLLMClient.
    model_url is ignored; set GAOYAO_JUDGE_MODEL or pass model_name to override.
    """
    judge_model = model_name or _DEFAULT_JUDGE_MODEL
    try:
        client = _get_judge_client(model=judge_model, caller='gaoyao_judge')
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
    """Inference call to the target (tested) model.

    Defaults to a local vLLM endpoint with no auth. When GAOYAO_LLM_API_KEY is
    set, it is sent as a Bearer token — use this for hosted OpenAI-compatible
    APIs. The target key is intentionally distinct from the judge key so the
    two providers can be billed and rate-limited separately.
    """
    url  = model_url  or _DEFAULT_LLM_URL
    name = model_name or _DEFAULT_LLM_NAME
    headers = {'Content-Type': 'application/json'}
    llm_api_key = os.environ.get('GAOYAO_LLM_API_KEY', '').strip()
    if llm_api_key:
        headers['Authorization'] = f'Bearer {llm_api_key}'
    try:
        data = {"model": name, "messages": messages,
                "temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
        if params:
            data.update(params)
        res = requests.post(url, headers=headers, json=data, timeout=300)
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
