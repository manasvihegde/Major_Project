"""
Week 5 - Janvi
Heatmap Visualizer for Geometric Misalignment.
Generates a 2D heatmap showing Cosine Distance across Layers and Tokens.
"""

import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

def generate_layer_token_heatmap(baseline_activations: dict, intervened_activations: dict, tokens: list, filename="misalignment_heatmap.png"):
    """
    Computes per-token cosine distance between baseline and intervened states,
    then renders and saves a 2D heatmap.
    """
    layers = [layer for layer in baseline_activations.keys() if layer in intervened_activations]
    
    # We will build a 2D matrix: [Number of Layers, Number of Tokens]
    heatmap_matrix = []

    for layer in layers:
        base_tensor = baseline_activations[layer]
        int_tensor = intervened_activations[layer]

        # Fail-safe: Align sequence lengths
        min_seq = min(base_tensor.shape[1], int_tensor.shape[1], len(tokens))
        base_tensor = base_tensor[:, :min_seq, :]
        int_tensor = int_tensor[:, :min_seq, :]
        current_tokens = tokens[:min_seq]

        # Calculate Cosine Distance per token (no .mean() this time!)
        # 0.0 = identical, higher = more divergent
        cos_sim = F.cosine_similarity(base_tensor, int_tensor, dim=-1)
        cos_dist = 1.0 - cos_sim 
        
        # Squeeze out the batch dimension and convert to numpy list
        heatmap_matrix.append(cos_dist.squeeze(0).numpy())

    # Convert to a strict 2D numpy array for matplotlib
    heatmap_array = np.array(heatmap_matrix)

    # --- RENDER THE HEATMAP ---
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Use the "Reds" colormap (darker red = higher misalignment)
    cax = ax.imshow(heatmap_array, cmap="Reds", aspect="auto")

    # Configure X-axis (Tokens)
    ax.set_xticks(np.arange(len(current_tokens)))
    ax.set_xticklabels(current_tokens, rotation=45, ha="right", fontsize=9)
    
    # Configure Y-axis (Layers)
    ax.set_yticks(np.arange(len(layers)))
    ax.set_yticklabels(layers, fontsize=9)

    # Labels & Title
    plt.colorbar(cax, label="Cosine Distance (Magnitude of Shift)")
    plt.title("Neural Misalignment Heatmap (Baseline vs. Intervention)", pad=20)
    plt.xlabel("Generated Tokens")
    plt.ylabel("Transformer Layers")
    plt.tight_layout()

    # Save the output safely
    out_dir = "data/outputs/"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)
    
    plt.savefig(out_path, dpi=200)
    print(f"📊 Heatmap successfully generated and saved to: {out_path}")
    plt.close()


# --- QUICK TEST SCRIPT ---
if __name__ == "__main__":
    from model_pipeline import HookedModelPipeline

    print("🚀 Initializing Pipeline...")
    pipeline = HookedModelPipeline(model_name="gpt2")
    prompt = "The capital of France is"

    # 1. Baseline Run
    print("Running Baseline...")
    base_result = pipeline.generate_with_hooks(prompt, max_new_tokens=8)
    base_acts = dict(pipeline.activations)
    
    # Extract the tokens to label our X-axis
    # GPT-2 input_ids + output_ids concatenated for the full sequence
    all_ids = torch.cat([base_result["input_ids"], base_result["output_ids"][0]])
    tokens = [pipeline.tokenizer.decode([t]) for t in all_ids]

    # 2. Intervened Run (Lobotomize Layer 4)
    print("Running Intervention on Layer 4...")
    pipeline.clear_interventions()
    lobotomy_tensor = torch.zeros_like(base_acts['layer_4'])
    pipeline.add_intervention("layer_4", lobotomy_tensor)
    
    int_result = pipeline.generate_with_hooks(prompt, max_new_tokens=8)
    int_acts = dict(pipeline.activations)

    # 3. Generate Heatmap
    print("\n🎨 Drawing Heatmap...")
    generate_layer_token_heatmap(base_acts, int_acts, tokens)