import json
from datasets import load_dataset

def curate_evaluation_datasets():
    print("🚀 Initializing reasoning dataset download via Hugging Face...")
    
    # Sourcing standard testing splits
    gsm8k_dataset = load_dataset("gsm8k", "main", split="test")
    strategy_dataset = load_dataset("wandael/strategyqa", split="train")
    
    curated_registry = []

    # 1. Processing GSM8K Records (Math Domain)
    print("📦 Processing GSM8K mathematical reasoning samples...")
    for index in range(min(50, len(gsm8k_dataset))):
        record = gsm8k_dataset[index]
        curated_registry.append({
            "id": f"gsm8k_{index:03d}",
            "domain": "math",
            "prompt": record["question"],
            "ground_truth_answer": record["answer"]
        })

    # 2. Processing StrategyQA Records (Commonsense Domain)
    print("📦 Processing StrategyQA logic reasoning samples...")
    for index in range(min(50, len(strategy_dataset))):
        record = strategy_dataset[index]
        curated_registry.append({
            "id": f"strategyqa_{index:03d}",
            "domain": "commonsense",
            "prompt": record["question"],
            "ground_truth_answer": str(record["answer"])
        })

    # Output to standardized JSON ecosystem file
    output_filename = "curated_reasoning_dataset.json"
    with open(output_filename, "w", encoding="utf-8") as file_out:
        json.dump(curated_registry, file_out, indent=4)
        
    print(f"✅ Sourcing Complete! {len(curated_registry)} unified records exported safely to '{output_filename}'.")

if __name__ == "__main__":
    curate_evaluation_datasets()
