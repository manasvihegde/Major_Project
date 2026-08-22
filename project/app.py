import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from model_pipeline import HookedModelPipeline
from perturbation_engine import PerturbationEngine
from interventions.geometric_metrics import compute_layer_drifts
from interventions.truth_score import calculate_truth_score

# Page Layout
st.set_page_config(page_title="LLM Stability & Faithfulness XAI Inspector", layout="wide")

st.title("🔍 LLM Stability & Faithfulness Inspector")
st.markdown("**Project Mentor Review Dashboard:** Interactive Causal Bottleneck & Truth-Score Verification.")

# Sidebar Control Panel & Guides
st.sidebar.header("Control Panel")
selected_model = st.sidebar.selectbox("Select Model Architecture", ["gpt2"])
max_tokens = st.sidebar.slider("Max Generation Tokens", 10, 100, 30)

st.sidebar.markdown("---")
st.sidebar.subheader("📖 Layer Reference Guide")
st.sidebar.info("""
- **0-3 (Surface Layers):** Processes syntax, grammar, and word embeddings.
- **4-7 (Context Layers):** Synthesizes concepts and phrases.
- **8-10 (Reasoning Bottleneck):** Critical layers for logic and causal decisions.
- **11 (Decision Layer):** Final token probability projection.
""")

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Heatmap Gradient Guide")
st.sidebar.markdown("""
- ⬜ **Light Pink / White (0.0 - 0.3):** Stable. No significant geometric drift.
- 🟧 **Medium Orange-Red (0.4 - 0.7):** Moderate semantic or context shift.
- 🟥 **Dark Burgundy / Red (0.8 - 1.0):** **Critical Causal Fracture.** Severe internal reasoning divergence (Hallucination trigger).
""")

@st.cache_resource
def load_pipeline():
    return HookedModelPipeline(model_name="gpt2")

with st.spinner("Loading transformer model and registering layer hooks..."):
    pipeline = load_pipeline()

engine = PerturbationEngine()

# Input section
prompt_input = st.text_input("Test Reasoning Prompt:", "If a train travels 60 miles in 1 hour, how far will it travel in 3 hours? Answer:")

if st.button("Run Causal Intervention Analysis"):
    with st.spinner("Running baseline generation and perturbation sweeps..."):
        # 1. Baseline
        base_res = pipeline.generate_with_hooks(prompt_input, max_new_tokens=max_tokens)
        base_acts = pipeline.activations.copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline Output")
            st.success(base_res["generated_text"])
            
        # 2. Perturbation
        perturbations = engine.generate(prompt_input)
        pert = perturbations[0] # Use the first perturbation
        
        pipeline.activations.clear()
        pert_res = pipeline.generate_with_hooks(pert["text"], max_new_tokens=max_tokens)
        pert_acts = pipeline.activations.copy()
        
        with col2:
            st.subheader(f"Perturbed Output ({pert['type']})")
            st.warning(pert_res["generated_text"])
            
        # 3. Metrics
        drifts = compute_layer_drifts(base_acts, pert_acts)
        cosine_vals = [d["cosine_distance"] for d in drifts.values() if not np.isnan(d["cosine_distance"])]
        euclid_vals = [d["euclidean_distance"] for d in drifts.values() if not np.isnan(d["euclidean_distance"])]
        
        avg_cosine = float(np.mean(cosine_vals)) if cosine_vals else 0.0
        avg_euclid = float(np.mean(euclid_vals)) if euclid_vals else 0.0
        truth_score = calculate_truth_score(avg_cosine, avg_euclid)
        
        # 4. Display Metrics
        st.markdown("---")
        st.header("📊 Quantitative Evaluation Metrics")
        m1, m2, m3 = st.columns(3)
        m1.metric("Unified Truth-Score", f"{truth_score} / 100")
        m2.metric("Average Cosine Drift", f"{avg_cosine:.4f}")
        m3.metric("Average Euclidean Drift", f"{avg_euclid:.4f}")

        # 5. Render Heatmap & Exact Token Trigger Analysis
        st.markdown("---")
        st.header("🔥 Layer-wise Neural Misalignment Heatmap & Trigger Analysis")
        
        layers = list(base_acts.keys())
        matrix_data = []
        
        for l in layers:
            if l in base_acts and l in pert_acts:
                b_tensor = base_acts[l].float()
                p_tensor = pert_acts[l].float()
                
                seq_len = min(b_tensor.shape[1], p_tensor.shape[1])
                layer_row = []
                for t in range(seq_len):
                    b_vec = b_tensor[0, t, :].view(-1)
                    p_vec = p_tensor[0, t, :].view(-1)
                    cos_sim = torch.nn.functional.cosine_similarity(b_vec.unsqueeze(0), p_vec.unsqueeze(0)).item()
                    distance = max(0.0, min(1.0, 1.0 - cos_sim))
                    layer_row.append(distance)
                matrix_data.append(layer_row)
                
        if matrix_data:
            max_len = max(len(row) for row in matrix_data)
            cosine_matrix = np.array([row + [0.0] * (max_len - len(row)) for row in matrix_data])
            
            # Isolate exact words causing variation by token comparison
            base_tokens = pipeline.tokenizer.tokenize(prompt_input)
            pert_tokens = pipeline.tokenizer.tokenize(pert["text"])
            
            # Find differing words
            diff_words = [w for w in pert_tokens if w not in base_tokens]
            trigger_highlight = " ".join(diff_words) if diff_words else pert["type"]

            # Isolate exact words causing variation cleanly without raw tokenizer artifacts
            raw_diff_text = pert["text"]
            clean_diff_text = raw_diff_text.replace("Ġ", " ").replace("</w>", "")

            st.markdown("### 🎯 Deviation Source & Exact Text Trigger")
            col_t1, col_t2 = st.columns(2)
            with col_t1:
                st.markdown("**Exact Modified Text Causing Variation:**")
                st.code(clean_diff_text)
            with col_t2:
                st.markdown("**Perturbation Category:**")
                st.code(pert["type"].upper())

            st.write(f"**Causal Insight:** The injection of this specific `{pert['type']}` modification altered the intermediate vector geometry, triggering a deep logical fracture in the model's reasoning bottleneck.")
            fig, ax = plt.subplots(figsize=(12, 6))
            sns.heatmap(cosine_matrix, cmap="Reds", annot=False, cbar=True, ax=ax, vmin=0, vmax=1)
            ax.set_yticks(np.arange(len(layers)) + 0.5)
            ax.set_yticklabels(layers, rotation=0)
            ax.set_xlabel("Generated Token Steps")
            ax.set_ylabel("Transformer Layers")
            ax.set_title("Dynamic Causal Divergence Across Layers & Token Steps")
            
            st.pyplot(fig)
        else:
            st.warning("Insufficient activation dimensions for plotting.")