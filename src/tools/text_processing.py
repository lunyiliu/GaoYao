import re
from src.tools.judger_algorithm import is_number
from src.evaluation.config import LANG_TO_ANSWER_PREFIX, NUMBER_TO_CHOICE_1_BASE

MULTILINGUAL_ANSWER_REGEXES = [
    r"Answer\s*:", r"Answer\s*:​+", r"答案\s*：", r"答案\s*:",
    r"Respuesta\s*:", r"Réponse\s*:", r"Antwort\s*:", r"Ответ\s*:",
    r"Jibu\s*:", r"సమాధానం\s*:", r"คำตอบ\s*:", r"উত্তর\s*:",
    r"الإجابة\s*:", r"Jawaban\s*:", r"Odgovor\s*:", r"Yanıt\s*:",
]


def clean_response(response: str) -> str:
    if not response:
        return ""
    if '</think>' in response:
        response = response.split('</think>')[-1]
    for tag in ['[unused16]', '[unused17]']:
        if tag in response:
            response = response.split(tag)[-1]
    if '[unused10]' in response:
        response = response.split('[unused10]')[0]
    return response.strip()


def extract_general_choice(text: str) -> str:
    if not text:
        return ''
    match = re.fullmatch(r'\s*\(?\s*([A-D0-4])\s*\)?\s*', text, re.IGNORECASE)
    if match:
        return match.group(1)
    matches = re.findall(r'(?:answer|答案)[:\s：]*([A-D])', text, re.IGNORECASE)
    if len(matches) == 1:
        return matches[0]
    matches2 = re.findall(r'\b([A-D])\b', text)
    if len(matches2) == 1:
        return matches2[0]
    return text


def normalize_multilingual_choice(text: str) -> str:
    return (text
            .replace("أ", "A").replace("ب", "B").replace("ج", "C").replace("د", "D")
            .replace("অ", "A").replace("ব", "B").replace("ড", "C").replace("ঢ", "D")
            .replace("Ａ", "A").replace("Ｂ", "B").replace("Ｃ", "C").replace("Ｄ", "D")
            .strip())


def normalize_response(response: str) -> str:
    return (response
            .replace("**", "").replace("$\\boxed{", "").replace("}$", "")
            .replace("\\$", "").replace("$\\text{", "").replace("$", "")
            .replace("\\mathrm{", "").replace("\\{", "").replace("\\text", "")
            .replace("\\(", "").replace("\\)", "").replace("\\mathbf{", "")
            .replace("{", "").replace("\\boxed", "").replace("}", "")
            .replace("based on the provided options", "")
            .replace("Option", "").replace("option", "")
            .replace("(", "").replace(")", "")
            .strip())


def get_prediction_numbers(prediction_prefix, response):
    if is_number(response):
        return response.strip()
    return parse_response_answer(response, prediction_prefix)


def parse_response_answer(answer: str, answer_prefix: str) -> str:
    if not answer:
        return ""
    working_prefix = answer_prefix
    if working_prefix.lower() not in answer.lower():
        for item in LANG_TO_ANSWER_PREFIX.values():
            if item.lower() in answer.lower():
                working_prefix = item
                break
        if working_prefix.lower() not in answer.lower():
            return ""
    answer_text = answer.lower().split(working_prefix.lower())[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    if not numbers:
        return ""
    pred = numbers[-1].rstrip(".")
    if "." in pred:
        pred = pred.rstrip("0").rstrip(".")
    return pred.replace(",", "")


def extracted_choice_question_answer(response_text, number_to_choice=None):
    if number_to_choice is None:
        number_to_choice = NUMBER_TO_CHOICE_1_BASE
    from src.tools.llm_request import send_chat_completion
    from src.tools.prompt_templates import CHOICE_SYSTEM_PROMPT
    extracted = send_chat_completion(CHOICE_SYSTEM_PROMPT, response_text)
    if is_number(extracted.strip()):
        extracted = number_to_choice.get(extracted.strip(), extracted.strip())
    return extracted


def extracted_true_or_false_question_answer(response_text):
    from src.tools.llm_request import send_chat_completion
    from src.tools.prompt_templates import TRUE_OR_FALSE_SYSTEM_PROMPT
    return send_chat_completion(TRUE_OR_FALSE_SYSTEM_PROMPT, response_text)
