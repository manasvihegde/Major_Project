"""
Week 5 - Janvi
Heatmap Visualizer for Geometric Misalignment.
Generates a 2D heatmap showing Cosine Distance across Layers and Tokens.
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def generate_layer_token_heatmap(baseline_activations: dict, intervened_activations: dict, tokens: list, filename="misalignment_heatmap.png"):
    """
    Computes per-token cosine distance between baseline and intervened states,
    then renders and saves a 2D heatmap.
    """
    layers = [layer for layer in baseline_activations.keys() if layer in intervened_activations]

    # Build 2D matrix: [Number of Layers, Number of Tokens]
    heatmap_matrix = []

    for layer in layers:
        base_tensor = baseline_activations[layer]
        int_tensor = intervened_activations[layer]

        # Fail-safe: Align sequence lengths
        min_seq = min(base_tensor.shape[1], int_tensor.shape[1], len(tokens))
        base_tensor = base_tensor[:, :min_seq, :]
        int_tensor = int_tensor[:, :min_seq, :]
        current_tokens = tokens[:min_seq]

        # Calculate Cosine Distance per token
        cos_sim = F.cosine_similarity(base_tensor, int_tensor, dim=-1)
        cos_dist = 1.0 - cos_sim

        # Squeeze batch dimension, detach, and convert to numpy list
        heatmap_matrix.append(cos_dist.squeeze(0).detach().cpu().numpy())

    # Convert to strict 2D numpy array for matplotlib
    heatmap_array = np.array(heatmap_matrix)

    # --- RENDER THE HEATMAP ---
    fig, ax = plt.subplots(figsize=(10, 8))

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

    # Save output
    out_dir = "data/outputs"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, filename)

    plt.savefig(out_path, dpi=200)
    print(f"🎨 Heatmap successfully generated and saved to: {out_path}")
    plt.close()


# --- TEST SCRIPT ---
if __name__ == "__main__":
    from model_pipeline import HookedModelPipeline

    print("🚀 Initializing Pipeline...")
    pipeline = HookedModelPipeline(model_name="gpt2")
    prompt = "The capital of France is"

    # 1. Baseline Run
    print("Running Baseline...")
    pipeline.register_layer_hooks()
    base_result = pipeline.generate_with_hooks(prompt, max_new_tokens=8)
    base_acts = dict(pipeline.activations)

    # Extract tokens to label X-axis
    base_text = base_result.get("generated_text", str(base_result)) if isinstance(base_result, dict) else str(base_result)
    tokens = pipeline.tokenizer.tokenize(base_text)

    # 2. Intervened Run (Lobotomize Layer 4 via direct PyTorch hook)
    print("Running Intervention on Layer 4...")

    def zero_layer_hook(module, input_act, output_act):
        if isinstance(output_act, tuple):
            h = torch.zeros_like(output_act[0])
            return (h,) + output_act[1:]
        else:
            return torch.zeros_like(output_act)

    target_layer = pipeline.model.transformer.h[4]
    intervene_handle = target_layer.register_forward_hook(zero_layer_hook)

    try:
        pipeline.register_layer_hooks()
        int_result = pipeline.generate_with_hooks(prompt, max_new_tokens=8)
        int_acts = dict(pipeline.activations)
    finally:
        intervene_handle.remove()

    # 3. Generate Heatmap
    print("\n🎨 Drawing Heatmap...")
    generate_layer_token_heatmap(base_acts, int_acts, tokens)