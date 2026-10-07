import os
import streamlit as st
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from model_pipeline import HookedModelPipeline
from perturbation_engine import PerturbationEngine
from interventions.geometric_metrics import compute_layer_drifts
from interventions.truth_score import calculate_truth_score

# --- Define MODEL_REGISTRY ---
MODEL_REGISTRY = {
    "distilgpt2": "distilgpt2",
    "gpt-neo-125m": "EleutherAI/gpt-neo-125m",
    "gpt2": "gpt2",
    "qwen2.5-0.5b": "Qwen/Qwen2.5-0.5B",
    "tinyllama-1.1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
}

# Page Configuration
st.set_page_config(page_title="LLM Stability & Faithfulness XAI Inspector", layout="wide")

st.title("🔍 LLM Stability & Faithfulness Inspector")
st.markdown("**Mechanistic Interpretability Suite:** Causal Bottleneck Verification & Neural Misalignment Heatmaps.")

# --- PART 1.1 & PART 3: Sidebar Control Panel ---
st.sidebar.header("Control Panel")

# 3.1 & 3.2 Dynamic model list directly from shared registry
available_models = list(MODEL_REGISTRY.keys())

st.sidebar.subheader("Mode A: Single Model Inspector")
selected_model = st.sidebar.selectbox(
    "Select Model Architecture",
    available_models,
    index=0
)

max_tokens = st.sidebar.slider("Max Generation Tokens", 10, 100, 30)
run_single_btn = st.sidebar.button("Run Single Model Analysis", type="secondary")

st.sidebar.markdown("---")
st.sidebar.subheader("Mode B: Multi-Model Benchmark")
st.sidebar.caption(f"Run full mechanistic probing across all {len(available_models)} loaded models.")
run_all_btn = st.sidebar.button("🚀 Run Stress Test on ALL Models", type="primary")

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Layer Reference Guide")
st.sidebar.info("""
- **0-3 (Surface Layers):** Syntax, grammar, and low-level token embeddings.
- **4-7 (Context Layers):** Concept synthesis and intermediate phrase composition.
- **8-10 (Reasoning Bottleneck):** Logical abstraction and causal decisions.
- **11+ (Decision Layers):** Vocabulary logit projection.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Heatmap Gradient Guide")
st.sidebar.markdown("""
- ⬜ **0.0 - 0.3 (White/Pink):** Stable internal representations.
- 🟧 **0.4 - 0.7 (Orange-Red):** Moderate semantic drift.
- 🟥 **0.8 - 1.0 (Dark Burgundy):** Critical causal bottleneck fracture.
""")

# Model parameter size mapping for summary display
MODEL_SIZES = {
    "distilgpt2": "82M",
    "gpt-neo-125m": "125M",
    "gpt2": "124M",
    "qwen2.5-0.5b": "490M",
    "tinyllama-1.1b": "1.1B"
}

@st.cache_resource
def get_pipeline(model_key: str):
    pipe = HookedModelPipeline(model_name=model_key)
    pipe.register_layer_hooks()
    return pipe

engine = PerturbationEngine()

# Input Reasoning Prompt
prompt_input = st.text_area(
    "Test Reasoning Prompt:",
    "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:"
)

# Core Reusable Probing and Deviation Computation (5.1 & 5.4)
def run_model_probing(model_key: str, prompt: str, max_new_tokens: int):
    # Resolve the actual Hugging Face model path from MODEL_REGISTRY
    hf_model_id = MODEL_REGISTRY.get(model_key, model_key)
    pipe = get_pipeline(hf_model_id)
    
    # 1. Baseline
    base_res = pipe.generate_with_hooks(prompt, max_new_tokens=max_new_tokens)
    base_acts = {k: v.clone() for k, v in pipe.activations.items()}

    # 2. Perturbation
    perturbations = engine.generate(prompt)
    pert = perturbations[0]

    pipe.activations.clear()
    pert_res = pipe.generate_with_hooks(pert["text"], max_new_tokens=max_new_tokens)
    pert_acts = {k: v.clone() for k, v in pipe.activations.items()}

    # 3. Geometric Metric Calculation
    drifts = compute_layer_drifts(base_acts, pert_acts)
    cosine_vals = [d["cosine_distance"] for d in drifts.values() if not np.isnan(d["cosine_distance"])]
    euclid_vals = [d["euclidean_distance"] for d in drifts.values() if not np.isnan(d["euclidean_distance"])]

    avg_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0
    avg_euclid = float(np.mean(euclid_vals)) if euclid_vals else 0.0
    truth_score = calculate_truth_score(avg_cosine, avg_euclid)

    # 4. Token x Layer Misalignment Matrix
    layers = list(base_acts.keys())
    matrix_data = []

    for l in layers:
        if l in base_acts and l in pert_acts:
            b_tensor = base_acts[l].float()
            p_tensor = pert_acts[l].float()

            # Align sequences safely across both dimensions
            b_seq = b_tensor.shape[1] if b_tensor.dim() >= 2 else b_tensor.shape[0]
            p_seq = p_tensor.shape[1] if p_tensor.dim() >= 2 else p_tensor.shape[0]
            seq_len = min(b_seq, p_seq)

            layer_row = []
            for t in range(seq_len):
                b_vec = b_tensor[0, t, :].view(-1) if b_tensor.dim() == 3 else b_tensor[t, :].view(-1)
                p_vec = p_tensor[0, t, :].view(-1) if p_tensor.dim() == 3 else p_tensor[t, :].view(-1)

                cos_sim = torch.nn.functional.cosine_similarity(b_vec.unsqueeze(0), p_vec.unsqueeze(0)).item()
                layer_row.append(max(0.0, min(1.0, 1.0 - cos_sim)))
            matrix_data.append(layer_row)

    fracture_layer = "None (Stable)"
    fracture_idx = 999
    if matrix_data:
        threshold = 0.5
        layer_means = [np.mean(row) for row in matrix_data]
        deviation_indices = [idx for idx, d in enumerate(layer_means) if d > threshold]
        if deviation_indices:
            fracture_idx = deviation_indices[0]
            fracture_layer = layers[fracture_idx]

    # Safe extraction of text from base_res and pert_res
    base_text_str = base_res.get("generated_text", str(base_res)) if isinstance(base_res, dict) else str(base_res)
    pert_res_text_str = pert_res.get("generated_text", str(pert_res)) if isinstance(pert_res, dict) else str(pert_res)

    return {
        "model_key": model_key,
        "base_text": base_text_str,
        "pert_type": pert["type"],
        "pert_text": pert["text"],
        "pert_res_text": pert_res_text_str,
        "avg_cosine": avg_cosine,
        "avg_euclid": avg_euclid,
        "truth_score": truth_score,
        "layers": layers,
        "matrix_data": matrix_data,
        "drifts": drifts,
        "fracture_layer": fracture_layer,
        "fracture_idx": fracture_idx,
    }

# Helper to render the visual elements inside any container or page area (4.1)
def render_model_details(data: dict):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Baseline Output:**")
        st.success(data["base_text"])
    with c2:
        st.markdown(f"**Perturbed Output (`{data['pert_type']}`):**")
        st.warning(data["pert_res_text"])

    m1, m2, m3 = st.columns(3)
    m1.metric("Unified Truth-Score", f"{data['truth_score']} / 100")
    m2.metric("Average Cosine Drift", f"{data['avg_cosine']:.4f}")
    m3.metric("Average Euclidean Drift", f"{data['avg_euclid']:.4f}")

    # Deviation Analysis
    st.markdown("### 🎯 Deviation Analysis")
    if data["fracture_layer"] != "None (Stable)":
        st.error(f"⚠️ **Causal Bottleneck Fracture:** Detected at **{data['fracture_layer']}**.")
        st.write(
            f"The model's internal geometry drifted significantly starting at **{data['fracture_layer']}** "
            f"(average cosine divergence > 0.5). This confirms the `{data['pert_type']}` perturbation "
            f"disrupted the model's internal reasoning before token projection."
        )
    else:
        st.success("✅ **Stable Alignment:** Internal vector representations remained aligned across all layers.")

    # Heatmaps
    if data["matrix_data"]:
        max_len = max(len(row) for row in data["matrix_data"])
        padded_matrix = np.array([row + [0.0] * (max_len - len(row)) for row in data["matrix_data"]])

        st.markdown("### 🔥 Dynamic Causal Divergence Across Layers & Token Steps")
        fig1, ax1 = plt.subplots(figsize=(12, 4))
        sns.heatmap(padded_matrix, cmap="Reds", annot=False, cbar=True, ax=ax1, vmin=0, vmax=1)
        ax1.set_yticks(np.arange(len(data["layers"])) + 0.5)
        ax1.set_yticklabels(data["layers"], rotation=0)
        ax1.set_xlabel("Generated Token Steps")
        ax1.set_ylabel("Transformer Layers")
        ax1.set_title(f"Dynamic Divergence Heatmap ({data['model_key']})")
        st.pyplot(fig1)

        st.markdown("### 🧬 Causal Divergence Across Transformer Depth")
        dim_matrix = np.array([[data["drifts"][l]["cosine_distance"] for l in data["layers"]]])
        fig2, ax2 = plt.subplots(figsize=(12, 2))
        sns.heatmap(dim_matrix, cmap="Reds", annot=False, cbar=True, ax=ax2, vmin=0, vmax=1)
        ax2.set_yticks([0.5])
        ax2.set_yticklabels(["Mean Drift"], rotation=0)
        ax2.set_xticks(np.arange(len(data["layers"])) + 0.5)
        ax2.set_xticklabels(data["layers"], rotation=45)
        ax2.set_xlabel("Transformer Layers")
        ax2.set_title(f"Mean Layer Drift Depth ({data['model_key']})")
        st.pyplot(fig2)

# --- EXECUTION ROUTING ---

# Mode A: Single Model (2.1)
if run_single_btn:
    st.markdown(f"## 🤖 Model Evaluation: `{selected_model}`")
    with st.spinner(f"Running probing and generating heatmaps for {selected_model}..."):
        try:
            result = run_model_probing(selected_model, prompt_input, max_tokens)
            render_model_details(result)
        except Exception as e:
            st.error(f"Error evaluating model '{selected_model}': {str(e)}")

# Mode B: All Models (Part 4)
elif run_all_btn:
    st.markdown("## 🚀 Multi-Model Comparative Benchmark")

    # 4.3 Incremental progress setup
    progress_bar = st.progress(0)
    status_text = st.empty()

    summary_container = st.container()
    results_container = st.container()

    evaluated_results = []
    total_models = len(available_models)

    for i, model_key in enumerate(available_models):
        status_text.info(f"⏳ Probing model **{i+1}/{total_models}: `{model_key}`**...")
        try:
            res = run_model_probing(model_key, prompt_input, max_tokens)
            evaluated_results.append(res)

            # 4.1 & 4.2 Incremental collapsible rendering
            with results_container:
                headline = (
                    f"**{model_key}** ({MODEL_SIZES.get(model_key, 'N/A')}) – "
                    f"{'⚠️ Fracture at ' + res['fracture_layer'] if res['fracture_layer'] != 'None (Stable)' else '✅ Preserved Alignment'}"
                )
                with st.expander(headline, expanded=(i == 0)):
                    render_model_details(res)

        except Exception as e:
            # 5.2 Partial failure handling
            evaluated_results.append({
                "model_key": model_key,
                "fracture_layer": "ERROR",
                "fracture_idx": -1,
                "error": str(e)
            })
            with results_container:
                with st.expander(f"❌ **{model_key}** – FAILED", expanded=False):
                    st.error(f"Failed to probe model '{model_key}': {str(e)}")

        progress_bar.progress((i + 1) / total_models)

    status_text.success(f"✅ Finished stress-testing all {total_models} models!")

    # 4.4 Compact cross-model summary table (sorted by earlier fracture layer)
    with summary_container:
        st.markdown("### 📊 Cross-Model Stability Summary")
        st.caption("Lower-numbered fracture layers indicate that internal reasoning broke down earlier in the network depth.")

        # Sort from earliest breakdown (lower index) to latest/stable
        sorted_results = sorted(evaluated_results, key=lambda x: x["fracture_idx"])

        summary_rows = []
        for r in sorted_results:
            summary_rows.append({
                "Model Key": r["model_key"],
                "Parameter Scale": MODEL_SIZES.get(r["model_key"], "Unknown"),
                "Fracture Layer": r["fracture_layer"],
                "Truth-Score": f"{r.get('truth_score', 'N/A')} / 100" if "truth_score" in r else "N/A",
                "Status": "⚠️ Fractured" if r["fracture_layer"] not in ["None (Stable)", "ERROR"] else ("✅ Stable" if r["fracture_layer"] == "None (Stable)" else "❌ Error")
            })

        st.table(summary_rows)
        st.markdown("---")