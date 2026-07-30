"""
Week 4 - Manasvi Hegde
Translate text modifications into token/embedding interventions.
Reads finished perturbations from the perturbations table (written during
parsing via BaseParser.insert_perturbations(), using PerturbationEngine
from Week 2) and converts each perturbed_text into an embedding-space
intervention using the same HookedModelPipeline class comparative_logger.py uses.
"""

import torch
from db.connection import get_dict_connection
from model_pipeline import HookedModelPipeline


def fetch_perturbation(perturbation_id: int) -> dict:
    conn = get_dict_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT p.id, p.perturbation_type, p.perturbed_text, p.diff_from_original,
               q.question_text
        FROM perturbations p
        JOIN questions q ON q.id = p.question_id
        WHERE p.id = %s
        """,
        (perturbation_id,)
    )
    row = cursor.fetchone()
    cursor.close()
    conn.close()

    if row is None:
        raise ValueError(f"No perturbation found with id {perturbation_id}")

    return {
        "id": row["id"],
        "perturbation_type": row["perturbation_type"],
        "perturbed_text": row["perturbed_text"],
        "diff_from_original": row["diff_from_original"],
        "original_text": row["question_text"],
    }


def text_diff_to_token_positions(tokenizer, original_text: str, modified_text: str) -> list[int]:
    orig_ids = tokenizer(original_text, add_special_tokens=False)["input_ids"]
    mod_ids = tokenizer(modified_text, add_special_tokens=False)["input_ids"]

    diff_positions = []
    max_len = max(len(orig_ids), len(mod_ids))
    for i in range(max_len):
        o = orig_ids[i] if i < len(orig_ids) else None
        m = mod_ids[i] if i < len(mod_ids) else None
        if o != m:
            diff_positions.append(i)

    return diff_positions


def build_embedding_intervention(pipeline: HookedModelPipeline, modified_text: str, positions: list[int], scale: float = 1.0):
    inputs = pipeline.tokenizer(modified_text, return_tensors="pt", add_special_tokens=False).to(pipeline.device)
    with torch.no_grad():
        embeddings = pipeline.model.get_input_embeddings()(inputs["input_ids"])

    mask = torch.zeros(embeddings.shape[1], dtype=torch.bool)
    for p in positions:
        if p < embeddings.shape[1]:
            mask[p] = True

    return {
        "input_ids": inputs["input_ids"].cpu(),
        "embeddings": embeddings.cpu(),
        "intervene_mask": mask,
        "scale": scale
    }


def run_intervention_for_perturbation(model_name: str, perturbation_id: int):
    """
    Loads a perturbation already stored by Week 2's pipeline (perturbations table)
    and converts it into an embedding intervention, ready for the Week 4
    causal bottleneck verification work.
    """
    pipeline = HookedModelPipeline(model_name=model_name)
    perturbation = fetch_perturbation(perturbation_id)

    positions = text_diff_to_token_positions(
        pipeline.tokenizer, perturbation["original_text"], perturbation["perturbed_text"]
    )
    intervention = build_embedding_intervention(
        pipeline, perturbation["perturbed_text"], positions
    )
    pipeline.clear_hooks()

    return perturbation, intervention


if __name__ == "__main__":
    perturbation, intervention = run_intervention_for_perturbation("gpt2", perturbation_id=1)
    print("Perturbation type:", perturbation["perturbation_type"])
    print("Intervention mask:", intervention["intervene_mask"])