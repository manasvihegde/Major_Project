import json
import os

from parsers.base_parser import BaseParser
from perturbation_engine import PerturbationEngine


# Maps a JSON record's "domain" field to the questions.question_type
# value used by the schema ('math', 'multi_hop', 'boolean').
# Anything not recognized falls back to "commonsense" for backward
# compatibility with older curated_reasoning_dataset.json files.
DOMAIN_TO_QUESTION_TYPE = {
    "math": "math",
    "boolean": "boolean",
    "multi_hop": "multi_hop",
    "commonsense": "boolean",  # older JSON files tagged StrategyQA as "commonsense"
}


class CuratedDatasetParser(BaseParser):

    def __init__(self, json_path):
        super().__init__(
            dataset_name="curated_reasoning",
            description="Curated reasoning dataset (GSM8K + StrategyQA + HotpotQA)",
            split="custom"
        )

        self.json_path = json_path
        self.engine = PerturbationEngine()

    def parse(self, num_samples=100):

        print(f"\nLoading curated dataset from {self.json_path}")

        with open(self.json_path, "r", encoding="utf-8") as f:
            dataset = json.load(f)

        dataset = dataset[:min(num_samples, len(dataset))]

        self.register_dataset(total_samples=len(dataset))

        success = 0
        failed = 0

        for i, item in enumerate(dataset):

            try:

                question_text = item["prompt"].strip()
                answer_text = str(item["ground_truth_answer"]).strip()
                domain = item.get("domain", "commonsense")

                question_type = DOMAIN_TO_QUESTION_TYPE.get(domain, "commonsense")

                if question_type == "math":
                    try:
                        answer_numeric = float(answer_text)
                    except ValueError:
                        answer_numeric = None
                else:
                    # boolean (StrategyQA "True"/"False") and multi_hop
                    # (HotpotQA free-text spans) never have a numeric answer.
                    answer_numeric = None

                qid = self.insert_question(
                    index=i,
                    question_text=question_text,
                    answer_text=answer_text,
                    answer_numeric=answer_numeric,
                    question_type=question_type
                )

                perturbations = self.engine.generate(question_text)

                self.insert_perturbations(qid, perturbations)

                success += 1

            except Exception as e:

                print(f"Error on sample {i}: {e}")
                failed += 1

        print(f"\nCurated dataset parsing complete")
        print(f"Inserted : {success}")
        print(f"Failed   : {failed}")