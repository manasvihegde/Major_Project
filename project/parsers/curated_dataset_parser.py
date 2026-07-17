import json
import os

from parsers.base_parser import BaseParser
from perturbation_engine import PerturbationEngine


class CuratedDatasetParser(BaseParser):

    def __init__(self, json_path):
        super().__init__(
            dataset_name="curated_reasoning",
            description="Curated reasoning dataset (GSM8K + StrategyQA)",
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

                if item["domain"] == "math":
                    question_type = "math"

                    try:
                        answer_numeric = float(answer_text)
                    except ValueError:
                        answer_numeric = None

                else:
                    question_type = "commonsense"
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