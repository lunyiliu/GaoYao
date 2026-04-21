import re
from src.evaluation.base_eval import BaseEval
from src.tools.judger_algorithm import (
    parse_judge_score, identify_question_type, compare_list,
)
from src.tools.llm_request import send_chat_completion
from src.tools.prompt_templates import MISSING_POINTS_SYSTEM_PROMPT, COMMON_SYSTEM_PROMPT
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.tools.text_processing import clean_response
from src.evaluation.config import CHOICE_TYPE, OPEN_QUESTION_TYPE, TRUE_OR_FALSE_TYPE

_TRUE_WORDS  = {'true', '正确', '对', 'verdadero', 'vrai', 'wahr', 'верно', 'صحيح'}
_FALSE_WORDS = {'false', '错误', '错', 'falso', 'faux', 'falsch', 'неверно', 'خطأ'}


def _parse_tf(text: str):
    t = text.strip().lower()
    if t in _TRUE_WORDS or re.fullmatch(r'(true|正确|对|verdadero)', t, re.I):
        return 1
    if t in _FALSE_WORDS or re.fullmatch(r'(false|错误|错|falso)', t, re.I):
        return 0
    return -1


def _extract_choice_list(text: str):
    return [s.strip() for s in re.split(r'[、,，。.\n]', text) if s.strip()]


class SAGE(BaseEval):

    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            gt   = str(item.get('gt', ''))
            lang = item.get('language', 'unknown')
            resp = clean_response(item.get('response', ''))

            stats["total"][lang] = stats["total"].get(lang, 0) + 1

            q_type = identify_question_type(gt)

            if q_type == CHOICE_TYPE:
                self._eval_choice(gt, resp, lang, item, stats)
            elif q_type == TRUE_OR_FALSE_TYPE:
                self._eval_tf(gt, resp, lang, item, stats)
            else:
                self._eval_open(gt, resp, lang, item, stats)

        return calc_acc_by_language(stats)

    def _eval_choice(self, gt, resp, lang, item, stats):
        gt_list   = _extract_choice_list(gt)
        pred_list = _extract_choice_list(resp)
        result = compare_list(gt_list, pred_list)
        item['gt_prediction']   = gt
        item['prediction'] = resp
        if result == -1:
            self.badcases.append(item)
        elif result == 1:
            stats["match"][lang] = stats["match"].get(lang, 0) + 1
        else:
            self.not_pass.append(item)

    def _eval_tf(self, gt, resp, lang, item, stats):
        gt_val   = _parse_tf(gt)
        pred_val = _parse_tf(resp)
        if pred_val == -1:
            from src.tools.text_processing import extracted_true_or_false_question_answer
            fallback = extracted_true_or_false_question_answer(resp)
            pred_val = _parse_tf(fallback)
        if pred_val == -1:
            item['prediction'] = resp
            self.badcases.append(item)
            return
        item['prediction'] = pred_val
        if gt_val == pred_val:
            stats["match"][lang] = stats["match"].get(lang, 0) + 1
        else:
            self.not_pass.append(item)

    def _eval_open(self, gt, resp, lang, item, stats):
        prompt   = item.get('prompt', '')
        user_msg = (MISSING_POINTS_SYSTEM_PROMPT
                    .replace("[question]",    prompt)
                    .replace("[reference]",   gt)
                    .replace("[model_output]", resp))
        judge_out = send_chat_completion(COMMON_SYSTEM_PROMPT, user_msg)
        score = parse_judge_score(judge_out)
        item['judge_score'] = score
        if score >= 50:
            stats["match"][lang] = stats["match"].get(lang, 0) + 1
        else:
            self.not_pass.append(item)
