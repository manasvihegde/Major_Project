import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

def run_wilcoxon_control_test(k=3, seed=42):
    data_path = "data/outputs/layer_patching_effects.npy"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Could not find {data_path}. Run attribution_patching.py first to generate layer effects."
        )

    # Load matrix: shape [N_samples, N_layers]
    effects_matrix = np.load(data_path)
    n_samples, n_layers = effects_matrix.shape
    print(f"Loaded patching data: {n_samples} samples across {n_layers} layers.")

    # 1. Identify Top-k Target Layers by mean recovery effect
    mean_per_layer = np.mean(effects_matrix, axis=0)
    sorted_layer_indices = np.argsort(mean_per_layer)[::-1]
    target_layers = sorted_layer_indices[:k]
    
    # 2. Select Control Layers (excluding top layers)
    np.random.seed(seed)
    available_control_layers = [l for l in range(n_layers) if l not in target_layers]
    control_layers = np.random.choice(available_control_layers, size=k, replace=False)

    print(f"Target Important Layers (top-{k}): {target_layers.tolist()}")
    print(f"Control Random Layers: {control_layers.tolist()}")

    # 3. Compute paired mean recovery per sample across selected layers
    target_sample_effects = np.mean(effects_matrix[:, target_layers], axis=1)
    control_sample_effects = np.mean(effects_matrix[:, control_layers], axis=1)

    # 4. Perform Wilcoxon Signed-Rank Test (paired, one-sided: target > control)
    test_result = wilcoxon(
        target_sample_effects,
        control_sample_effects,
        alternative="greater"
    )

    stat = test_result.statistic
    p_val = test_result.pvalue

    print("\n--- Wilcoxon Signed-Rank Test Results ---")
    print(f"Sample Size (N): {n_samples}")
    print(f"Target Mean Effect:  {np.mean(target_sample_effects):.4f} (± {np.std(target_sample_effects)/np.sqrt(n_samples):.4f})")
    print(f"Control Mean Effect: {np.mean(control_sample_effects):.4f} (± {np.std(control_sample_effects)/np.sqrt(n_samples):.4f})")
    print(f"Test Statistic (W):  {stat}")
    print(f"p-value:             {p_val:.4e}")
    print(f"Significant at p < 0.05: {'Yes' if p_val < 0.05 else 'No'}")

    # 5. Save results to CSV for Objective 3 reporting
    results_df = pd.DataFrame([{
        "sample_size": n_samples,
        "k_layers": k,
        "target_layers": str(target_layers.tolist()),
        "control_layers": str(control_layers.tolist()),
        "target_mean_effect": float(np.mean(target_sample_effects)),
        "control_mean_effect": float(np.mean(control_sample_effects)),
        "wilcoxon_statistic": float(stat),
        "p_value": float(p_val),
        "statistically_significant": bool(p_val < 0.05)
    }])

    output_csv = "data/outputs/wilcoxon_control_results.csv"
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    results_df.to_csv(output_csv, index=False)
    print(f"Saved results to: {output_csv}")

if __name__ == "__main__":
    run_wilcoxon_control_test()