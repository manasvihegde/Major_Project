import os
import re
import random
from dotenv import load_dotenv
from datasets import load_dataset
from parsers.base_parser import BaseParser

load_dotenv()

class GSM8KParser(BaseParser):

    def __init__(self, split="train"):
        super().__init__(
            dataset_name="gsm8k",
            description="Grade school math word problems with step-by-step solutions",
            split=split
        )

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

                perturbations = self._build_perturbations(question_text)
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

    def _build_perturbations(self, question_text):
        perturbations = []

        prefixes = [
            "Please solve the following: ",
            "Work through this math problem: ",
            "Answer this question carefully: ",
            "Here is a word problem: "
        ]
        perturbations.append({
            "type":          "prefix",
            "text":          random.choice(prefixes) + question_text,
            "original_text": question_text
        })

        name_map = {
            "Natalia": "Priya",   "James": "Arjun",
            "Mary":    "Divya",   "Tom":   "Ravi",
            "Sarah":   "Anjali",  "John":  "Kiran",
            "Emma":    "Meera",   "Mike":  "Suresh",
            "Lisa":    "Pooja",   "David": "Vikram"
        }
        p_name = question_text
        for orig, new in name_map.items():
            p_name = p_name.replace(orig, new)
        perturbations.append({
            "type":          "name_swap",
            "text":          p_name,
            "original_text": question_text
        })

        synonym_map = {
            "How many":      "What is the total number of",
            "altogether":    "in total",
            "In total":      "Altogether",
            "purchased":     "bought",
            "twice as many": "double the number of",
            "half as many":  "half the number of",
            "remaining":     "left over",
            "more than":     "greater than",
            "less than":     "fewer than",
            "If ":           "Given that ",
            "each day":      "per day",
            "per hour":      "every hour",
            "total cost":    "combined price",
            "How much":      "What is the amount",
            "spent":         "used"
        }
        p_syn = question_text
        for orig, syn in synonym_map.items():
            p_syn = p_syn.replace(orig, syn)
        perturbations.append({
            "type":          "synonym",
            "text":          p_syn,
            "original_text": question_text
        })

        words = question_text.split()
        if len(words) > 4:
            pos = random.sample(range(1, len(words) - 1), min(2, len(words) - 2))
            for p in sorted(pos, reverse=True):
                words.insert(p, "")
        p_ws = " ".join(words).strip()
        perturbations.append({
            "type":          "whitespace",
            "text":          p_ws,
            "original_text": question_text
        })

        return perturbations