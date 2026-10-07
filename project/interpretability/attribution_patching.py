import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from db.connection import get_connection
from model_pipeline import HookedModelPipeline

def run_activation_patching(limit=25):
    # 1. Initialize Pipeline & Model
    pipeline = HookedModelPipeline("gpt2")
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    device = next(model.parameters()).device
    model.eval()

    print(f"[*] Running model on: {device}")

    # 2. Query Database
    conn = get_connection()
    cur = conn.cursor()

    cur.execute(f"""
    SELECT
        p.id,
        q.id,
        q.question_text,
        p.perturbed_text
    FROM perturbations p
    JOIN questions q ON p.question_id = q.id
    LIMIT {limit}
    """)
    rows = cur.fetchall()

    if not rows:
        print("No perturbation data found in database.")
        cur.close()
        conn.close()
        return

    num_layers = len(model.transformer.h)
    all_layer_patch_effects = []

    print(f"[*] Starting activation patching across {len(rows)} samples...")

    for sample_idx, (perturbation_id, question_id, clean_text, perturbed_text) in enumerate(rows):
        print(f" -> Processing sample {sample_idx + 1}/{len(rows)}...", flush=True)

        clean_inputs = tokenizer(clean_text, return_tensors="pt").to(device)
        corrupt_inputs = tokenizer(perturbed_text, return_tensors="pt").to(device)

        seq_len = min(clean_inputs.input_ids.size(1), corrupt_inputs.input_ids.size(1))
        if seq_len == 0:
            continue

        clean_ids = clean_inputs.input_ids[:, :seq_len]
        corrupt_ids = corrupt_inputs.input_ids[:, :seq_len]

        # Base forward passes
        with torch.no_grad():
            clean_out = model(input_ids=clean_ids, output_hidden_states=True)
            corrupt_out = model(input_ids=corrupt_ids, output_hidden_states=True)

        target_token_id = clean_out.logits[0, -1].argmax().item()
        clean_metric = clean_out.logits[0, -1, target_token_id].item()
        corrupt_metric = corrupt_out.logits[0, -1, target_token_id].item()

        denom = clean_metric - corrupt_metric
        if abs(denom) < 1e-8:
            continue

        sample_layer_effects = []

        # Layer-wise patching across the 12 transformer blocks
        for layer_idx in range(num_layers):
            clean_layer_act = clean_out.hidden_states[layer_idx + 1][:, :seq_len, :].detach().clone()
            target_block = model.transformer.h[layer_idx]

            def layer_patch_hook(module, input_act, output_act):
    # Patch only the decision token position (-1)
                if isinstance(output_act, tuple):
                    h = output_act[0].clone()
                    h[:, seq_len - 1, :] = clean_layer_act[:, seq_len - 1, :]
                    return (h,) + output_act[1:]
                else:
                    h = output_act.clone()
                    h[:, seq_len - 1, :] = clean_layer_act[:, seq_len - 1, :]
                    return h

            handle = target_block.register_forward_hook(layer_patch_hook)
            try:
                with torch.no_grad():
                    patched_out = model(input_ids=corrupt_ids)
                    patched_metric = patched_out.logits[0, -1, target_token_id].item()
            finally:
                handle.remove()

            layer_norm_effect = (patched_metric - corrupt_metric) / denom
            sample_layer_effects.append(layer_norm_effect)

        # Log final layer representative score to database
        if sample_layer_effects:
            cur.execute("""
            INSERT INTO xai_attribution_scores
            (run_id, question_id, perturbation_id, token_index, token_text, activation_patch_score)
            VALUES (%s, %s, %s, %s, %s, %s)
            """, (1, question_id, perturbation_id, 0, "ALL_TOKENS", float(sample_layer_effects[-1])))

        if len(sample_layer_effects) == num_layers:
            all_layer_patch_effects.append(sample_layer_effects)

        conn.commit()

    cur.close()
    conn.close()

    # 3. Save plot and raw numpy array for Wilcoxon test
    if all_layer_patch_effects:
        effects_matrix = np.array(all_layer_patch_effects)
        mean_effects = np.mean(effects_matrix, axis=0)
        std_err = np.std(effects_matrix, axis=0) / np.sqrt(len(effects_matrix))

        layers = np.arange(num_layers)
        output_dir = "data/outputs"
        os.makedirs(output_dir, exist_ok=True)
        plot_path = os.path.join(output_dir, "activation_patching_layers.png")

        plt.figure(figsize=(9, 5))
        plt.plot(layers, mean_effects, marker="o", color="#1f77b4", linewidth=2, label="Mean Recovery Effect")
        plt.fill_between(layers, mean_effects - std_err, mean_effects + std_err, color="#1f77b4", alpha=0.25, label="±1 SEM")
        plt.xlabel("Layer Index", fontsize=12)
        plt.ylabel("Normalized Patching Effect", fontsize=12)
        plt.title("Layer-Wise Activation Patching Recovery", fontsize=14)
        plt.xticks(layers)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300)
        plt.close()

        npy_path = os.path.join(output_dir, "layer_patching_effects.npy")
        np.save(npy_path, effects_matrix)
        print(f"\n[+] Layer plot successfully saved to: {plot_path}")
        print(f"[+] Raw effects matrix saved to: {npy_path}")

if __name__ == "__main__":
    run_activation_patching(limit=25)