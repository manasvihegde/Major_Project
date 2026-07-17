import json
import re

def call_model_backend_stub(prompt):
    """
    Mock interface wrapper.
    Replace with direct execution hooks calling Janvi's vLLM/Inference API.
    """
    return "Step 1: Extrapolate variables. Step 2: Formulate solution. The final answer is 42."

def run_baseline_benchmarking(dataset_filepath):
    print(f"📊 Reading evaluation target: {dataset_filepath}")
    with open(dataset_filepath, "r", encoding="utf-8") as f_in:
        dataset = json.load(f_in)
        
    total_records = len(dataset)
    aggregate_word_count = 0
    correct_matches = 0

    print(f"⚙️ Running baseline passes across {total_records} curated evaluation prompts...")
    
    for item in dataset:
        generated_explanation = call_model_backend_stub(item["prompt"])
        
        words = generated_explanation.split()
        aggregate_word_count += len(words)
        
        normalized_truth = re.sub(r'[^a-zA-Z0-9]', '', item["ground_truth_answer"].lower())
        normalized_pred = re.sub(r'[^a-zA-Z0-9]', '', generated_explanation.lower())
        
        if normalized_truth in normalized_pred:
            correct_matches += 1

    mean_length = aggregate_word_count / total_records
    computed_accuracy = (correct_matches / total_records) * 100
    
    print("\n📈 ======= BASELINE METRIC MATRIX =======")
    print(f"Average Generation Length : {mean_length:.2f} words")
    print(f"Baseline Engine Accuracy   : {computed_accuracy:.2f}%")
    print("===========================================\n")
    
    telemetry_payload = {
        "dataset_source": dataset_filepath,
        "sample_size": total_records,
        "metrics": {
            "avg_explanation_length_words": mean_length,
            "baseline_accuracy_percentage": computed_accuracy
        }
    }
    
    with open("baseline_metrics_log.json", "w", encoding="utf-8") as log_out:
        json.dump(telemetry_payload, log_out, indent=4)
    print("💾 Telemetry logs securely written to 'baseline_metrics_log.json'.")

if __name__ == "__main__":
    run_baseline_benchmarking("curated_reasoning_dataset.json")

