"""
LLM request helpers for GaoYao evaluation.

Both the target (tested) model and the judge model are reached through a plain
OpenAI-compatible /chat/completions endpoint via ``requests`` — no extra
package is required.

  Target model — ``send_inference()``
    GAOYAO_LLM_URL       full /chat/completions URL (default: local vLLM)
    GAOYAO_LLM_NAME      model name
    GAOYAO_LLM_API_KEY   Bearer key (optional; only needed if the endpoint
                         requires auth, e.g. a hosted API)

  Judge model — ``send_chat_completion()``
    GAOYAO_JUDGE_BASE_URL provider base URL ending in ``/v1`` (default: OpenAI)
    GAOYAO_JUDGE_MODEL    model name
    GAOYAO_JUDGE_API_KEY  Bearer key (falls back to GAOYAO_API_KEY for
                          backwards compatibility; never hardcoded)
"""
import os
import logging
import requests

logger = logging.getLogger('eval_logger')

# Env vars are read at *call time* (see helpers below) so a runner that sets
# them after importing this module — e.g. ``run_eval.py`` after argparse —
# is honoured. Module-level snapshots would silently lose those overrides.

def _judge_model():    return os.environ.get('GAOYAO_JUDGE_MODEL',    '')
def _judge_url():
    """Full /chat/completions URL for the judge.

    Priority: GAOYAO_JUDGE_URL  (explicit full URL, set by ``--judge-url``)
              > GAOYAO_JUDGE_BASE_URL + '/chat/completions'
              > OpenAI default.
    """
    full = os.environ.get('GAOYAO_JUDGE_URL', '').strip()
    if full:
        return full
    base = os.environ.get('GAOYAO_JUDGE_BASE_URL', 'https://api.openai.com/v1').rstrip('/')
    return f"{base}/chat/completions"
def _llm_url():        return os.environ.get('GAOYAO_LLM_URL',  'http://localhost:8000/v1/chat/completions')
def _llm_name():       return os.environ.get('GAOYAO_LLM_NAME', '')
def _comet_model():    return os.environ.get('GAOYAO_COMET_MODEL', 'Unbabel/wmt22-comet-da')


def _post_chat(url: str, api_key: str, model: str,
               messages: list, temperature: float, max_tokens: int) -> str:
    """Single helper for any OpenAI-compatible /chat/completions endpoint."""
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    res = requests.post(url, headers=headers, json=payload, timeout=300)
    if res.status_code != 200:
        logger.error(f"chat completion failed {res.status_code}: {res.text[:200]}")
        return ''
    return res.json()["choices"][0]["message"]["content"]


def send_chat_completion(system_prompt, user_prompt,
                         model_name=None, model_url=None, params=None):
    """Judge call to an OpenAI-compatible /chat/completions endpoint.

    Caller-supplied ``model_name`` and ``model_url`` win over env vars.
    """
    judge_model = model_name or _judge_model()
    url         = model_url  or _judge_url()
    api_key     = os.environ.get('GAOYAO_JUDGE_API_KEY') or os.environ.get('GAOYAO_API_KEY')
    if not api_key:
        logger.error(
            "judge API key not set — export GAOYAO_JUDGE_API_KEY "
            "(or GAOYAO_API_KEY for backwards compatibility)"
        )
        return ''
    try:
        return _post_chat(
            url, api_key, judge_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.1, max_tokens=1024,
        )
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
    url     = model_url  or _llm_url()
    name    = model_name or _llm_name()
    api_key = os.environ.get('GAOYAO_LLM_API_KEY', '').strip()
    sampling = {"temperature": 0.7, "top_p": 0.8, "max_tokens": 4096}
    if params:
        sampling.update(params)
    headers = {'Content-Type': 'application/json'}
    if api_key:
        headers['Authorization'] = f'Bearer {api_key}'
    try:
        data = {"model": name, "messages": messages, **sampling}
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
    cmodel = model_name or _comet_model()
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
