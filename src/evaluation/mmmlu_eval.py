import re
from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import (
    clean_response, extract_general_choice, normalize_multilingual_choice,
    MULTILINGUAL_ANSWER_REGEXES, normalize_response,
)
from src.tools.judger_algorithm import is_valid_choice
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT


class MMMLU(BaseEval):
    MULTILINGUAL_ANSWER_PATTERN_TEMPLATE = r"(?:{})\s*([A-D])"

    def normalize_extracted_answer(self, text: str) -> str:
        return normalize_multilingual_choice(text.strip()).upper()[:1]

    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            lang = item.get('language', 'unknown')
            stats["total"][lang] = stats["total"].get(lang, 0) + 1

            gt = normalize_multilingual_choice(
                extract_general_choice(str(item.get('gt', '')))
            ).upper()

            resp = clean_response(item.get('response', ''))
            clean_resp = normalize_response(resp)

            pred = extract_general_choice(clean_resp)
            if len(pred) == 1 and pred.upper() in 'ABCD':
                pred = normalize_multilingual_choice(pred).upper()
            else:
                extracted = ''
                for regex_tmpl in MULTILINGUAL_ANSWER_REGEXES:
                    pattern = self.MULTILINGUAL_ANSWER_PATTERN_TEMPLATE.format(regex_tmpl)
                    matches = list(re.finditer(pattern, clean_resp))
                    if matches:
                        extracted = self.normalize_extracted_answer(matches[-1].group(1))
                        break
                if not extracted:
                    fallback = send_chat_completion(CHOICE_SYSTEM_PROMPT, resp)
                    extracted = extract_general_choice(fallback).upper()
                pred = extracted

            item['prediction'] = pred

            if is_valid_choice(pred) and pred == gt:
                stats["match"][lang] = stats["match"].get(lang, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        return calc_acc_by_language(stats)
