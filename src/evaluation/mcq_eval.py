from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import clean_response, extract_general_choice, normalize_multilingual_choice
from src.tools.llm_request import send_chat_completion
from src.tools.judger_algorithm import is_valid_choice, is_number
from src.tools.metrics_and_report_operator import calc_acc_by_language
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT
from src.evaluation.config import NUMBER_TO_CHOICE_1_BASE


class MultiChoiceQuestion(BaseEval):
    number_to_choice = NUMBER_TO_CHOICE_1_BASE

    def evaluate(self, data_list):
        stats = {"total": {}, "match": {}}

        for item in data_list:
            lang = item.get('language', 'unknown')
            stats["total"][lang] = stats["total"].get(lang, 0) + 1

            gt_raw = str(item.get('gt', ''))
            gt = extract_general_choice(gt_raw)
            if is_number(gt):
                gt = self.number_to_choice.get(gt, gt)
            gt = normalize_multilingual_choice(gt).upper()

            resp = clean_response(item.get('response', ''))
            pred = extract_general_choice(resp)

            if not is_valid_choice(pred):
                pred = send_chat_completion(CHOICE_SYSTEM_PROMPT, resp)
                pred = extract_general_choice(pred)

            pred = normalize_multilingual_choice(pred).upper()
            item['prediction'] = pred

            if is_valid_choice(pred) and pred == gt:
                stats["match"][lang] = stats["match"].get(lang, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        return calc_acc_by_language(stats)


# Alias kept for legacy imports
MultiChoiceExam = MultiChoiceQuestion
