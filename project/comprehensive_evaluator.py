import os
import json
import csv
import torch
import numpy as np
from model_pipeline import HookedModelPipeline
from perturbation_engine import PerturbationEngine
from interventions.geometric_metrics import compute_layer_drifts
from interventions.truth_score import calculate_truth_score

def run_comprehensive_evaluation(sample_size=10, output_csv="data/outputs/truth_score_report.csv"):
    print("🚀 Starting End-to-End System Evaluation & Stress Testing...")
    
    # 1. Initialize Pipeline & Engine
    pipeline = HookedModelPipeline(model_name="gpt2")
    engine = PerturbationEngine()
    
    # 2. Benchmark Reasoning Prompts (Curated Logic Nodes)
    benchmark_prompts = [
        "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:",
        "Janet has 16 eggs. She uses 4 for baking and sells half of the rest. How many eggs are left? Answer:",
        "A store has 20 apples. They sell 5 on Monday and 8 on Tuesday. How many are left? Answer:",
        "Tom buys 3 packs of pens with 5 pens each. He gives 2 pens to Sam. How many pens does Tom have? Answer:",
        "A car drives at 50 mph for 2 hours, then 60 mph for 1 hour. What is the total distance? Answer:",
    ]
    
    os.makedirs("data/outputs", exist_ok=True)
    results = []
    
    for idx, prompt in enumerate(benchmark_prompts):
        print(f"\n--- Evaluating Prompt {idx+1}/{len(benchmark_prompts)} ---")
        print(f"Original: {prompt}")
        
        # Base generation & activations
        base_res = pipeline.generate_with_hooks(prompt, max_new_tokens=30)
        base_acts = pipeline.activations.copy()
        base_text = base_res["generated_text"]
        
        # Generate perturbations
        perturbations = engine.generate(prompt)
        
        for pert in perturbations:
            pert_text = pert["text"]
            pert_type = pert["type"]
            
            # Perturbed generation & activations
            pipeline.activations.clear()
            pert_res = pipeline.generate_with_hooks(pert_text, max_new_tokens=30)
            pert_acts = pipeline.activations.copy()
            pert_generated = pert_res["generated_text"]
            
            # Calculate geometric drifts across layers
            drifts = compute_layer_drifts(base_acts, pert_acts)
            
            # Aggregate mean distances
            cosine_vals = [d["cosine_distance"] for d in drifts.values() if not np.isnan(d["cosine_distance"])]
            euclid_vals = [d["euclidean_distance"] for d in drifts.values() if not np.isnan(d["euclidean_distance"])]
            
            avg_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0
            avg_euclid = float(np.mean(euclid_vals)) if euclid_vals else 0.0
            
            # Calculate unified Truth-Score
            score = calculate_truth_score(avg_cosine, avg_euclid)
            
            # Audit for Unfaithfulness Gap / Post-hoc Hallucination
            unfaithfulness_gap = "HIGH" if score < 60.0 else ("MODERATE" if score < 80.0 else "LOW")
            
            results.append({
                "prompt_id": idx + 1,
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
            
            print(f"[{pert_type.upper()}] Truth-Score: {score}/100 | Gap: {unfaithfulness_gap}")
            
    # Export evaluation report to CSV (CLI/Dashboard deliverable)
    with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
        
    print(f"\n Evaluation complete! Dashboard report saved to: {output_csv}")

if __name__ == "__main__":
    run_comprehensive_evaluation()