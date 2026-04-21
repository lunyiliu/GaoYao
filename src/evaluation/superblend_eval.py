from src.evaluation.base_eval import BaseEval
from src.tools.text_processing import clean_response, extract_general_choice
from src.tools.llm_request import send_chat_completion
from src.tools.judger_algorithm import is_valid_choice, is_number
from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT
from src.tools.metrics_and_report_operator import export_country_acc_result
from src.evaluation.config import NUMBER_TO_CHOICE_1_BASE


class SuperBlend(BaseEval):
    number_to_choice = NUMBER_TO_CHOICE_1_BASE

    def evaluate(self, data_list):
        all_dict   = {}
        match_dict = {}

        for item in data_list:
            lang    = item.get('language', 'unknown')
            country = item.get('country', 'unknown')

            all_dict.setdefault(lang, {})
            all_dict[lang][country] = all_dict[lang].get(country, 0) + 1

            gt_raw = str(item.get('gt', ''))
            gt = extract_general_choice(gt_raw)
            if is_number(gt):
                gt = self.number_to_choice.get(gt, gt)
            gt = gt.upper()

            resp = clean_response(item.get('response', ''))
            pred = extract_general_choice(resp)

            if not is_valid_choice(pred):
                pred = send_chat_completion(CHOICE_SYSTEM_PROMPT, resp)
                pred = extract_general_choice(pred)

            pred = pred.upper()
            item['prediction'] = pred

            if is_valid_choice(pred) and pred == gt:
                match_dict.setdefault(lang, {})
                match_dict[lang][country] = match_dict[lang].get(country, 0) + 1
            elif not is_valid_choice(pred):
                self.badcases.append(item)
            else:
                self.not_pass.append(item)

        return export_country_acc_result(all_dict, match_dict)
