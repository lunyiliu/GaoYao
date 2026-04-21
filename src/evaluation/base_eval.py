import os
import random
from abc import ABC, abstractmethod
from src.tools.file_operations import read_jsonl, write_jsonl
from src.log.logging_config import logger


class BaseEval(ABC):
    def __init__(self):
        self.badcases = []
        self.not_pass = []

    def run(self, input_path, output_path, badcase_path, not_pass_path, sample_pct=100):
        logger.info(f"Start Eval: {self.__class__.__name__} (sample={sample_pct}%)")
        data_list = read_jsonl(input_path)

        if sample_pct < 100:
            k = max(1, int(len(data_list) * sample_pct / 100))
            data_list = random.sample(data_list, k)
            logger.info(f"  Sampled {k}/{len(data_list)+k} items")

        metrics = self.evaluate(data_list)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if self.badcases:
            write_jsonl(badcase_path, self.badcases)
        if self.not_pass:
            write_jsonl(not_pass_path, self.not_pass)

        return metrics

    @abstractmethod
    def evaluate(self, data_list):
        pass
