"""
curate_dataset.py

Week 1 - Dataset curation
Pulls three reasoning domains into curated_reasoning_dataset.json:
  - math          (GSM8K)
  - boolean       (StrategyQA)
  - multi_hop     (HotpotQA, distractor setting)

Uses the `datasets` library instead of raw URL scraping for StrategyQA/HotpotQA
since the previous raw-GitHub StrategyQA URL (elazar/strategyqa) does not exist
and was silently falling back to a 3-item backup array every run.
"""

import json
import re
import pandas as pd

# pip install datasets --break-system-packages
from datasets import load_dataset


def extract_gsm8k_numeric_answer(raw_answer: str):
    """
    GSM8K's 'answer' field is the full reasoning chain, ending in '#### <number>'.
    Pulls out just the final numeric answer.
    """
    match = re.search(r"####\s*(-?[\d,]+(?:\.\d+)?)", raw_answer)
    if not match:
        return None
    try:
        return float(match.group(1).replace(",", ""))
    except ValueError:
        return None


def curate_gsm8k(curated_registry, limit=50):
    print("📦 Downloading GSM8K mathematical reasoning samples...")
    try:
        gsm8k_url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        gsm8k_df = pd.read_json(gsm8k_url, lines=True)

        for index in range(min(limit, len(gsm8k_df))):
            record = gsm8k_df.iloc[index]
            raw_answer = record["answer"]
            numeric = extract_gsm8k_numeric_answer(raw_answer)

            curated_registry.append({
                "id": f"gsm8k_{index:03d}",
                "domain": "math",
                "prompt": record["question"],
                # Store just the final numeric answer, not the full reasoning
                # chain, so ground_truth_answer is consistent across domains.
                "ground_truth_answer": str(numeric) if numeric is not None else raw_answer.split("####")[-1].strip()
            })
        print("   ✅ GSM8K loaded successfully.")

    except Exception:
        print("   ⚠️ GSM8K URL fallback triggered due to network layout. Using backup subset.")
        backup_math = [
            {"q": "Weng earns $12 an hour for babysitting. Yesterday, she babysat for 5 hours. How much money did she earn?", "a": "60"},
            {"q": "A deep-sea monster shark has 143 teeth. It loses 25 teeth in a battle. How many teeth does it have left?", "a": "118"},
            {"q": "If John has 5 apples and eats 2, how many are left?", "a": "3"}
        ]
        for index, item in enumerate(backup_math):
            curated_registry.append({
                "id": f"gsm8k_{index:03d}",
                "domain": "math",
                "prompt": item["q"],
                "ground_truth_answer": item["a"]
            })


def curate_strategyqa(curated_registry, limit=50):
    print("📦 Downloading StrategyQA commonsense/boolean reasoning samples...")
    try:
        # wics/strategy-qa: 2,290 rows, clean `question` + boolean `answer` columns.
        # (The previously used elazar/strategyqa raw-GitHub URL does not exist.)
        ds = load_dataset("wics/strategy-qa", "strategyQA", split="test")

        for index in range(min(limit, len(ds))):
            record = ds[index]
            curated_registry.append({
                "id": f"strategyqa_{index:03d}",
                "domain": "boolean",
                "prompt": record["question"],
                "ground_truth_answer": str(record["answer"])  # "True" / "False"
            })
        print("   ✅ StrategyQA loaded successfully.")

    except Exception as e:
        print(f"   ⚠️ StrategyQA load exception: {e}. Injecting backup subset.")
        backup_logic = [
            {"q": "Do fish breathe underground?", "a": "False"},
            {"q": "Could a human walk on the surface of Neptune?", "a": "False"},
            {"q": "Are lilies capable of photosynthesis?", "a": "True"}
        ]
        for index, item in enumerate(backup_logic):
            curated_registry.append({
                "id": f"strategyqa_{index:03d}",
                "domain": "boolean",
                "prompt": item["q"],
                "ground_truth_answer": item["a"]
            })


def curate_hotpotqa(curated_registry, limit=50):
    print("📦 Downloading HotpotQA multi-hop reasoning samples...")
    try:
        # Official HotpotQA, distractor setting (each question ships with
        # 10 context paragraphs, 2 of which are the real supporting facts).
        ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")

        for index in range(min(limit, len(ds))):
            record = ds[index]
            curated_registry.append({
                "id": f"hotpotqa_{index:03d}",
                "domain": "multi_hop",
                "prompt": record["question"],
                "ground_truth_answer": record["answer"]
            })
        print("   ✅ HotpotQA loaded successfully.")

    except Exception as e:
        print(f"   ⚠️ HotpotQA load exception: {e}. Injecting backup subset.")
        backup_multihop = [
            {"q": "What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?",
             "a": "Chief of Protocol"},
            {"q": "The Oberoi family is part of a hotel company that has a head office in what city?",
             "a": "Delhi"},
        ]
        for index, item in enumerate(backup_multihop):
            curated_registry.append({
                "id": f"hotpotqa_{index:03d}",
                "domain": "multi_hop",
                "prompt": item["q"],
                "ground_truth_answer": item["a"]
            })


def curate_evaluation_datasets(samples_per_domain=50):
    print("🚀 Sourcing multi-step reasoning evaluation datasets...")
    curated_registry = []

    curate_gsm8k(curated_registry, limit=samples_per_domain)
    curate_strategyqa(curated_registry, limit=samples_per_domain)
    curate_hotpotqa(curated_registry, limit=samples_per_domain)

    output_filename = "curated_reasoning_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as file_out:
        json.dump(curated_registry, file_out, indent=4)

    by_domain = {}
    for item in curated_registry:
        by_domain[item["domain"]] = by_domain.get(item["domain"], 0) + 1

    print(f"\n✨ SUCCESS! {len(curated_registry)} unified evaluation profiles compiled inside '{output_filename}'!")
    for domain, count in by_domain.items():
        print(f"   {domain:12s}: {count}")


if __name__ == "__main__":
    curate_evaluation_datasets(samples_per_domain=50)