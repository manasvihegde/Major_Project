"""
Week 3 - Diya
"Identify layers/heads with highest sensitivity variance"

There's no per-layer/per-head table in schema.sql, so this operates
in-memory directly on HookedModelPipeline's captured activations/attentions
(the same object comparative_logger.py uses) rather than reading from DB.
Compares original vs. perturbed prompts and ranks layers/heads by how much
their activations/attention shift — i.e. where the model is most sensitive
to the perturbation.

Output is saved to data/outputs/ as JSON rather than a DB table, since no
schema exists for this yet. If you want this persisted relationally, we'd
need a new table (e.g. layer_sensitivity_scores) — let me know if you want
that added to schema.sql.
"""

import json
import torch
from model_pipeline import HookedModelPipeline
from db.connection import get_dict_connection


def compute_layer_activation_variance(orig_activations: dict, pert_activations: dict) -> dict:
    """
    For each layer present in both activation dicts, computes the mean
    absolute difference between original and perturbed activations as a
    proxy for "sensitivity" to the perturbation.
    """
    layer_variance = {}
    for layer_name, orig_tensor in orig_activations.items():
        pert_tensor = pert_activations.get(layer_name)
        if pert_tensor is None:
            continue

        min_len = min(orig_tensor.shape[1], pert_tensor.shape[1])
        diff = (orig_tensor[:, :min_len, :] - pert_tensor[:, :min_len, :]).abs()
        layer_variance[layer_name] = diff.mean().item()

    return layer_variance


def compute_head_attention_variance(orig_attentions: list, pert_attentions: list) -> dict:
    """
    orig_attentions / pert_attentions: list of per-layer tensors
    [heads, seq, seq], as returned by HookedModelPipeline.generate_with_hooks.
    Returns {layer_idx: {head_idx: variance}}.
    """
    head_variance = {}
    for layer_idx, (orig_layer, pert_layer) in enumerate(zip(orig_attentions, pert_attentions)):
        num_heads = min(orig_layer.shape[0], pert_layer.shape[0])
        min_seq = min(orig_layer.shape[1], pert_layer.shape[1])

        head_scores = {}
        for h in range(num_heads):
            diff = (orig_layer[h, :min_seq, :min_seq] - pert_layer[h, :min_seq, :min_seq]).abs()
            head_scores[h] = diff.mean().item()

        head_variance[layer_idx] = head_scores

    return head_variance


def fetch_perturbation_pair(conn, question_id: int, perturbation_type: str = "prefix") -> dict:
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT q.question_text, p.perturbed_text
        FROM questions q
        JOIN perturbations p ON p.question_id = q.id
        WHERE q.id = %s AND p.perturbation_type = %s
        """,
        (question_id, perturbation_type)
    )
    row = cursor.fetchone()
    cursor.close()
    if row is None:
        raise ValueError(f"No perturbation of type '{perturbation_type}' found for question {question_id}")
    return {"original_text": row["question_text"], "perturbed_text": row["perturbed_text"]}


def analyze_sensitivity(model_name: str, question_id: int, perturbation_type: str = "prefix", top_k: int = 5):
    conn = get_dict_connection()
    pair = fetch_perturbation_pair(conn, question_id, perturbation_type)
    conn.close()

    pipeline = HookedModelPipeline(model_name=model_name)

    orig_result = pipeline.generate_with_hooks(pair["original_text"], max_new_tokens=20)
    orig_activations = dict(pipeline.activations)  # snapshot before it's cleared by the next call

    pert_result = pipeline.generate_with_hooks(pair["perturbed_text"], max_new_tokens=20)
    pert_activations = dict(pipeline.activations)

    layer_variance = compute_layer_activation_variance(orig_activations, pert_activations)
    head_variance = compute_head_attention_variance(
        orig_result["attentions"] or [], pert_result["attentions"] or []
    )

    pipeline.clear_hooks()

    ranked_layers = sorted(layer_variance.items(), key=lambda x: x[1], reverse=True)[:top_k]

    output = {
        "question_id": question_id,
        "perturbation_type": perturbation_type,
        "top_layers_by_variance": [{"layer": name, "variance": round(v, 6)} for name, v in ranked_layers],
        "head_variance_by_layer": {
            str(layer_idx): {str(h): round(v, 6) for h, v in heads.items()}
            for layer_idx, heads in head_variance.items()
        }
    }

    out_path = f"data/outputs/sensitivity_variance_q{question_id}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nTop {top_k} most sensitive layers for Q{question_id} ({perturbation_type}):")
    for layer, var in ranked_layers:
        print(f"  {layer}: {var:.6f}")
    print(f"Saved full breakdown to {out_path}")

    return output


if __name__ == "__main__":
    analyze_sensitivity(model_key="gpt2", question_id=1, perturbation_type="prefix")