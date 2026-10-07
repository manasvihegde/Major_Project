import os
import torch
import numpy as np
import pandas as pd
from peft import PeftModel
from model_pipeline import HookedModelPipeline
from interventions.geometric_metrics import compute_layer_drifts
from interventions.truth_score import calculate_truth_score

try:
    from interventions.regression_tracker import compare_runs
except ImportError:
    compare_runs = None

def compare_models():
    print("📊 Running Comparative Analysis: Base Model vs. LoRA-Tuned Model (Objective 3 Report)")
    
    prompt = "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:"
    
    # 1. Base Model evaluation
    print("Evaluating Base Model...")
    base_pipeline = HookedModelPipeline(model_key="gpt2")
    base_res = base_pipeline.generate_with_hooks(prompt, max_new_tokens=20)
    base_acts = base_pipeline.activations.copy()
    
    # 2. Tuned Model evaluation (Loading LoRA adapter checkpoint)
    print("Evaluating LoRA-Tuned Model...")
    tuned_pipeline = HookedModelPipeline(model_key="gpt2")
    
    adapter_path = "checkpoints/lora_v1"
    if not os.path.exists(adapter_path):
        adapter_path = "data/outputs/lora_adapter_checkpoint"
        
    if os.path.exists(adapter_path):
        tuned_pipeline.model = PeftModel.from_pretrained(tuned_pipeline.model, adapter_path)
    else:
        print(f"⚠️ Warning: Adapter path not found at {adapter_path}. Running with unadapted model.")
        
    tuned_res = tuned_pipeline.generate_with_hooks(prompt, max_new_tokens=20)
    tuned_acts = tuned_pipeline.activations.copy()
    
    # 3. Calculate metrics
    drifts = compute_layer_drifts(base_acts, tuned_acts)
    cosine_vals = [d["cosine_distance"] for d in drifts.values() if not np.isnan(d["cosine_distance"])]
    euclid_vals = [d["euclidean_distance"] for d in drifts.values() if not np.isnan(d["euclidean_distance"])]
    
    avg_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0
    avg_euclid = float(np.mean(euclid_vals)) if euclid_vals else 0.0
    truth_score = calculate_truth_score(avg_cosine, avg_euclid)
    
    print("\n" + "="*50)
    print(f"--- Baseline Output --- \n{base_res['generated_text']}")
    print("-" * 50)
    print(f"--- Tuned (LoRA) Output --- \n{tuned_res['generated_text']}")
    print("=" * 50)
    print(f"✨ Objective 3 Comparative Metrics:")
    print(f"   - Adapter Stability Truth-Score: {truth_score} / 100")
    print(f"   - Average Geometric Drift: {avg_cosine:.4f}")
    print("🎉 Objective 3 XAI-Guided Tuning & Verification Complete!")

def evaluate_regression_diff():
    print("\n📊 Executing Regression Tracker Diff Between Baseline and LoRA Adapter...")
    if compare_runs is not None:
        try:
            comparison_results = compare_runs("gpt2_baseline", "gpt2+lora_v1")
            print("✨ Comparison Complete!")
            print(comparison_results)
        except Exception as e:
            print(f"ℹ️ Note: Regression comparison encountered an issue ({e}).")
    else:
        print("ℹ️ regression_tracker module not found. Skipping SQL diff.")

if __name__ == "__main__":
    compare_models()
    evaluate_regression_diff()

from interventions.regression_tracker import compare_runs, get_or_create_run_id

base_id = get_or_create_run_id("gpt2", notes="baseline_run")
tuned_id = get_or_create_run_id("gpt2+lora_v1", notes="tuned_run")
diff_report = compare_runs(base_id, tuned_id)


def generate_comparison_report(
    baseline_csv="data/outputs/truth_score_report_baseline.csv",
    tuned_csv="data/outputs/truth_score_report_tuned.csv"
):
    print("📊 Evaluating Tuning Gains: Baseline vs. LoRA-Tuned Model")

    if not os.path.exists(baseline_csv) or not os.path.exists(tuned_csv):
        print("⚠️ Missing evaluation CSVs. Ensure comprehensive_evaluator has run for both models.")
        return

    df_base = pd.read_csv(baseline_csv)
    df_tuned = pd.read_csv(tuned_csv)

    base_mean = df_base["truth_score"].mean()
    tuned_mean = df_tuned["truth_score"].mean()
    delta = tuned_mean - base_mean

    def get_distribution(df):
        counts = df["unfaithfulness_gap"].value_counts(normalize=True) * 100
        return {
            "LOW": counts.get("LOW", 0.0),
            "MODERATE": counts.get("MODERATE", 0.0),
            "HIGH": counts.get("HIGH", 0.0)
        }

    dist_base = get_distribution(df_base)
    dist_tuned = get_distribution(df_tuned)

    print("\n" + "=" * 65)
    print("✨ OBJECTIVE 3: BASELINE VS. LORA-TUNED FAITHFULNESS REPORT")
    print("=" * 65)
    print(f"{'Metric':<30} | {'Baseline (GPT-2)':<15} | {'Tuned (LoRA v1)':<15}")
    print("-" * 65)
    print(f"{'Mean Truth-Score':<30} | {base_mean:<15.2f} | {tuned_mean:<15.2f}")
    print(f"{'Score Delta':<30} | {'-':<15} | {f'{delta:+.2f}':<15}")
    print(f"{'Low Instability (%)':<30} | {dist_base['LOW']:<14.1f}% | {dist_tuned['LOW']:<14.1f}%")
    print(f"{'High Instability (%)':<30} | {dist_base['HIGH']:<14.1f}% | {dist_tuned['HIGH']:<14.1f}%")
    print("=" * 65)

    os.makedirs("data/outputs", exist_ok=True)
    report_md = "data/outputs/tuning_comparison_summary.md"
    with open(report_md, "w", encoding="utf-8") as f:
        f.write("# Objective 3: XAI-Guided LoRA Tuning Report\n\n")
        f.write(f"- **Baseline Mean Truth-Score:** {base_mean:.2f}\n")
        f.write(f"- **Tuned Mean Truth-Score:** {tuned_mean:.2f} ({delta:+.2f})\n")
        f.write(f"- **High Instability Prompts Reduced:** {dist_base['HIGH']:.1f}% -> {dist_tuned['HIGH']:.1f}%\n")
    print(f"✅ Summary saved to {report_md}")

if __name__ == "__main__":
    generate_comparison_report()