import os
from dotenv import load_dotenv
from datasets import load_dataset
from parsers.base_parser import BaseParser
from perturbation_engine import PerturbationEngine

load_dotenv()

class HotpotQAParser(BaseParser):

    def __init__(self, split="validation"):
        super().__init__(
            dataset_name="hotpotqa",
            description="Wikipedia-based multi-hop reasoning questions (distractor setting)",
            split=split
        )
        self.engine = PerturbationEngine()

    def parse(self, num_samples=100):
        print(f"\nLoading HotpotQA ({self.split} split)...")
        print("  (First run downloads ~750MB — this may take a while.)")
        dataset = load_dataset("hotpotqa/hotpot_qa", "distractor")[self.split]
        samples = dataset.select(range(min(num_samples, len(dataset))))

        self.register_dataset(total_samples=len(samples))

        success, failed = 0, 0

        for i, item in enumerate(samples):
            try:
                question_text = item["question"].strip()
                answer_text   = item["answer"].strip()

                qid = self.insert_question(
                    index          = i,
                    question_text  = question_text,
                    answer_text    = answer_text,
                    answer_numeric = None,
                    question_type  = "multi_hop"
                )

                perturbations = self.engine.generate(question_text)
                self.insert_perturbations(qid, perturbations)

                success += 1
                if (i + 1) % 10 == 0:
                    print(f"  Processed {i+1}/{len(samples)} questions...")

            except Exception as e:
                print(f"  ERROR on sample {i}: {e}")
                failed += 1

        print(f"\nHotpotQA parsing complete.")
        print(f"  Inserted : {success} questions")
        print(f"  Failed   : {failed} questions")