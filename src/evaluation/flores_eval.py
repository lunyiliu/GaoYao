import uuid
from src.evaluation.base_eval import BaseEval
from src.tools.llm_request import send_comet_req, compute_comet_local
from src.tools.text_processing import clean_response
from src.log.logging_config import logger


class Flores(BaseEval):
    def evaluate(self, data_list):
        lang_map = {}
        for item in data_list:
            lang_map.setdefault(item.get('language', 'unknown'), []).append(item)

        results = {}
        for lang, items in lang_map.items():
            srcs = [x.get('src_text', x.get('prompt', '')) for x in items]
            refs = [str(x.get('gt', '')) for x in items]
            mts  = [clean_response(x.get('response', '')) for x in items]

            logger.info(f"  COMET scoring {lang} ({len(items)} items)")
            score = send_comet_req(str(uuid.uuid4()), srcs, refs, mts)
            results[lang] = score

        return results
