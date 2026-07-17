import os
import re
from dotenv import load_dotenv
from datasets import load_dataset
from parsers.base_parser import BaseParser
from perturbation_engine import PerturbationEngine

load_dotenv()

class GSM8KParser(BaseParser):

    def __init__(self, split="train"):
        super().__init__(
            dataset_name="gsm8k",
            description="Grade school math word problems with step-by-step solutions",
            split=split
        )
        self.engine = PerturbationEngine()

    def parse(self, num_samples=100):
        print(f"\nLoading GSM8K ({self.split} split)...")
        dataset = load_dataset("openai/gsm8k", "main")[self.split]
        samples = dataset.select(range(min(num_samples, len(dataset))))

        self.register_dataset(total_samples=len(samples))

        success, failed = 0, 0

        for i, item in enumerate(samples):
            try:
                question_text = item["question"].strip()
                answer_text   = item["answer"].strip()
                answer_num    = self._extract_final_number(answer_text)

                qid = self.insert_question(
                    index          = i,
                    question_text  = question_text,
                    answer_text    = answer_text,
                    answer_numeric = answer_num,
                    question_type  = "math"
                )

                perturbations = self.engine.generate(question_text)
                self.insert_perturbations(qid, perturbations)

                success += 1
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(samples)} questions...")

            except Exception as e:
                print(f"  ERROR on sample {i}: {e}")
                failed += 1

        print(f"\nGSM8K parsing complete.")
        print(f"  Inserted : {success} questions")
        print(f"  Failed   : {failed} questions")

    def _extract_final_number(self, answer_text):
        match = re.search(r'####\s*([\d,.-]+)', answer_text)
        if match:
            num_str = match.group(1).replace(",", "")
            try:
                return float(num_str)
            except ValueError:
                return None
        return None