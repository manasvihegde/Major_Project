import os
import pandas as pd

def build_training_dataset(report_path="data/outputs/truth_score_report.csv"):
    """
    Builds training tuples from the evaluation CSV log or database exports:
    (question_id, baseline_prompt, perturbed_prompt, ground_truth_answer, current_truth_score, current_instability_category)
    """
    if not os.path.exists(report_path):
        print(f"⚠️ Warning: {report_path} not found. Returning a fallback sample dataset.")
        return [
            {
                "question_id": 1,
                "baseline_prompt": "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:",
                "perturbed_prompt": "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:",
                "ground_truth_answer": " 180 miles.",
                "truth_score": 85.0,
                "instability_category": "LOW"
            }
        ]

    df = pd.read_csv(report_path)
    dataset = []
    
    for _, row in df.iterrows():
        dataset.append({
            "question_id": row.get("prompt_id", 1),
            "baseline_prompt": row.get("original_prompt", ""),
            "perturbed_prompt": row.get("perturbed_prompt", ""),
            "ground_truth_answer": row.get("base_output", ""),
            "truth_score": float(row.get("truth_score", 50.0)),
            "instability_category": row.get("unfaithfulness_gap", "MODERATE")
        })
        
    print(f"✅ Loaded {len(dataset)} samples for training from {report_path}")
    return dataset

if __name__ == "__main__":
    build_training_dataset()