import os
import sys
import argparse
import csv
import torch
import numpy as np
from peft import PeftModel
from model_pipeline import HookedModelPipeline
from perturbation_engine import PerturbationEngine
from interventions.geometric_metrics import compute_layer_drifts
from interventions.truth_score import calculate_truth_score

def run_comprehensive_evaluation(adapter_path=None, run_name="gpt2_baseline", output_csv="data/outputs/truth_score_report.csv"):
    print(f"🚀 Starting Evaluation Run: [{run_name}] (Adapter Path: {adapter_path})")
    
    # 1. Initialize Pipeline & Engine
    pipeline = HookedModelPipeline(model_name="gpt2")
    
    # Load LoRA adapter if provided
    if adapter_path and os.path.exists(adapter_path):
        pipeline.model = PeftModel.from_pretrained(pipeline.model, adapter_path)
        print(f"✨ Successfully attached PEFT adapter from {adapter_path}")
        
    engine = PerturbationEngine()
    
    benchmark_prompts = [
        "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:",
        "Janet has 16 eggs. She uses 4 for baking and sells half of the rest. How many eggs are left? Answer:",
        "A store has 20 apples. They sell 5 on Monday and 8 on Tuesday. How many are left? Answer:",
    ]
    
    os.makedirs("data/outputs", exist_ok=True)
    results = []
    
    for idx, prompt in enumerate(benchmark_prompts):
        base_res = pipeline.generate_with_hooks(prompt, max_new_tokens=30)
        base_acts = pipeline.activations.copy()
        base_text = base_res["generated_text"]
        
        perturbations = engine.generate(prompt)
        for pert in perturbations:
            pert_text = pert["text"]
            pert_type = pert["type"]
            
            pipeline.activations.clear()
            pert_res = pipeline.generate_with_hooks(pert_text, max_new_tokens=30)
            pert_acts = pipeline.activations.copy()
            pert_generated = pert_res["generated_text"]
            
            drifts = compute_layer_drifts(base_acts, pert_acts)
            cosine_vals = [d["cosine_distance"] for d in drifts.values() if not np.isnan(d["cosine_distance"])]
            euclid_vals = [d["euclidean_distance"] for d in drifts.values() if not np.isnan(d["euclidean_distance"])]
            
            avg_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0
            avg_euclid = float(np.mean(euclid_vals)) if euclid_vals else 0.0
            score = calculate_truth_score(avg_cosine, avg_euclid)
            unfaithfulness_gap = "HIGH" if score < 60.0 else ("MODERATE" if score < 80.0 else "LOW")
            
            results.append({
                "prompt_id": idx + 1,
                "run_name": run_name,
                "original_prompt": prompt,
                "perturbation_type": pert_type,
                "perturbed_prompt": pert_text,
                "base_output": base_text.replace("\n", " "),
                "perturbed_output": pert_generated.replace("\n", " "),
                "avg_cosine_drift": round(avg_cosine, 4),
                "avg_euclidean_drift": round(avg_euclid, 4),
                "truth_score": score,
                "unfaithfulness_gap": unfaithfulness_gap
            })
            
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"✅ Evaluation complete for [{run_name}]! Saved to {output_csv}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_path", type=str, default=None)
    parser.add_argument("--run_name", type=str, default="gpt2_baseline")
    parser.add_argument("--output_csv", type=str, default="data/outputs/truth_score_report.csv")
    args = parser.parse_args()
    
    run_comprehensive_evaluation(adapter_path=args.adapter_path, run_name=args.run_name, output_csv=args.output_csv)