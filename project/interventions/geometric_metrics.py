"""
Week 5 - Janvi
Mathematical module for geometric distance/misalignment.
Optimized for scale using native PyTorch vectorization.
"""

import torch
import torch.nn.functional as F

def calculate_layer_misalignment(baseline_activations: dict, intervened_activations: dict) -> dict:
    """
    Computes the geometric distance (Cosine & L2) between two sets of neural states.
    Optimized for scale: avoids Python loops over sequence lengths, strictly uses 
    vectorized operations.
    """
    misalignment_scores = {}

    for layer_name in baseline_activations.keys():
        # Only compare layers that exist in both runs
        if layer_name not in intervened_activations:
            continue

        base_tensor = baseline_activations[layer_name]
        int_tensor = intervened_activations[layer_name]

        # STRUCTURAL FAIL-SAFE: Truncate to match sequence lengths 
        # (in case the intervention caused the generation length to differ)
        min_seq = min(base_tensor.shape[1], int_tensor.shape[1])
        base_tensor = base_tensor[:, :min_seq, :]
        int_tensor = int_tensor[:, :min_seq, :]

        # --- OPTIMIZED VECTOR MATH ---
        # 1. Cosine Distance: 0.0 means identical, 2.0 means completely opposite
        # F.cosine_similarity operates directly on the last dimension (hidden_dim)
        cos_sim = F.cosine_similarity(base_tensor, int_tensor, dim=-1)
        cos_dist = 1.0 - cos_sim  

        # 2. L2 Distance: Absolute magnitude difference
        l2_dist = torch.norm(base_tensor - int_tensor, p=2, dim=-1)

        # Average the distances across all tokens in the sequence, 
        # moving to .item() only at the very end to prevent CPU/GPU bottlenecks
        misalignment_scores[layer_name] = {
            "mean_cosine_distance": cos_dist.mean().item(),
            "mean_l2_distance": l2_dist.mean().item()
        }

    return misalignment_scores


# --- QUICK TEST SCRIPT ---
if __name__ == "__main__":
    from model_pipeline import HookedModelPipeline

    print("🚀 Initializing Pipeline...")
    pipeline = HookedModelPipeline(model_name="gpt2")
    prompt = "The capital of France is"

    # 1. Baseline Run
    base_result = pipeline.generate_with_hooks(prompt, max_new_tokens=5)
    base_acts = dict(pipeline.activations)

    # 2. Intervened Run
    pipeline.clear_interventions()
    # Zero out layer 3
    lobotomy_tensor = torch.zeros_like(base_acts['layer_3'])
    pipeline.add_intervention("layer_3", lobotomy_tensor)
    int_result = pipeline.generate_with_hooks(prompt, max_new_tokens=5)
    int_acts = dict(pipeline.activations)

    # 3. Calculate Geometry at Scale
    print("\n📊 Computing Geometric Misalignment...")
    scores = calculate_layer_misalignment(base_acts, int_acts)
    
    for layer, metrics in list(scores.items())[:5]: # Print first 5 layers
        print(f"{layer}: Cosine Dist = {metrics['mean_cosine_distance']:.4f}, L2 Dist = {metrics['mean_l2_distance']:.4f}")