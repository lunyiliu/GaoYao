import re
from src.evaluation.base_eval import BaseEval
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import WIN_RATE_SYSYTEM_PROMPT
from src.tools.text_processing import clean_response
from src.tools.metrics_and_report_operator import calculate_win_rate
from src.tools.judger_algorithm import get_compare_tag


def _extract_verdict(text: str) -> str:
    m = re.search(r'\[\[([ABC])\]\]', text)
    return m.group(1) if m else ''


class AlpacaEval(BaseEval):
    def evaluate(self, data_list):
        lang_stats = {}

        for item in data_list:
            lang   = item.get('language', 'unknown')
            prompt = item.get('prompt', '')
            gt     = str(item.get('gt', ''))
            resp   = clean_response(item.get('response', ''))

            res_1 = _extract_verdict(
                send_chat_completion(WIN_RATE_SYSYTEM_PROMPT,
                                     f"[Question]{prompt}\n[Assistant A]{resp}\n[Assistant B]{gt}"))
            res_2 = _extract_verdict(
                send_chat_completion(WIN_RATE_SYSYTEM_PROMPT,
                                     f"[Question]{prompt}\n[Assistant A]{gt}\n[Assistant B]{resp}"))

            winner = get_compare_tag(res=res_1, res_swap=res_2)
            item['winner'] = winner

            if lang not in lang_stats:
                lang_stats[lang] = {"win": 0, "tie": 0, "lose": 0}
            lang_stats[lang][winner] = lang_stats[lang].get(winner, 0) + 1

        return {lang: calculate_win_rate(stats) for lang, stats in lang_stats.items()}
