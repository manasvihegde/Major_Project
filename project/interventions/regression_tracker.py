"""
Week 4 - Manasvi Hegde
Automated regression tracking for final answer probability.
Uses HookedModelPipeline (same class as comparative_logger.py) to get
per-step logits, then computes the probability the model assigned to the
first token of its own generated answer vs. the ground truth token.
Writes results into model_outputs.final_answer_probability and compares
across run_logs entries.
"""

import torch
from db.connection import get_connection, get_dict_connection
from model_pipeline import HookedModelPipeline


def get_answer_probability(pipeline: HookedModelPipeline, prompt: str, answer_token: str, max_new_tokens: int = 10) -> float:
    """
    Runs the prompt through the pipeline and returns the probability assigned
    to `answer_token` at the first generation step (logits[0] from generate_with_hooks).
    """
    result = pipeline.generate_with_hooks(prompt, max_new_tokens=max_new_tokens)
    if not result["logits"]:
        raise ValueError("No logits captured — check HookedModelPipeline.generate_with_hooks")

    first_step_logits = result["logits"][0]
    probs = torch.softmax(first_step_logits, dim=-1)

    answer_id = pipeline.tokenizer.encode(answer_token, add_special_tokens=False)[0]
    return probs[answer_id].item()


def update_output_probability(model_output_id: int, probability: float):
    """Writes a computed probability back onto an existing model_outputs row."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        "UPDATE model_outputs SET final_answer_probability = %s WHERE id = %s",
        (probability, model_output_id)
    )
    conn.commit()
    cursor.close()
    conn.close()


def track_regression(pipeline: HookedModelPipeline, run_id: int):
    """
    For every model_outputs row under a given run_id, compute the probability
    the model assigned to its ground-truth answer and store it.
    """
    conn = get_dict_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT mo.id AS model_output_id, mo.input_text, q.ground_truth_answer
        FROM model_outputs mo
        JOIN questions q ON q.id = mo.question_id
        WHERE mo.run_id = %s
        """,
        (run_id,)
    )
    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    results = []
    for row in rows:
        prob = get_answer_probability(pipeline, row["input_text"], str(row["ground_truth_answer"]))
        update_output_probability(row["model_output_id"], prob)
        results.append({"model_output_id": row["model_output_id"], "probability": prob})

    return results


def compare_runs(baseline_run_id: int, current_run_id: int, threshold: float = 0.05) -> list[dict]:
    """
    Compares final_answer_probability between two run_logs (by matching question_id)
    and flags regressions where probability dropped more than `threshold`.
    """
    conn = get_dict_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        SELECT mo.question_id, mo.final_answer_probability
        FROM model_outputs mo
        WHERE mo.run_id = %s AND mo.perturbation_id IS NULL
        """,
        (baseline_run_id,)
    )
    baseline = {r["question_id"]: r["final_answer_probability"] for r in cursor.fetchall()
                if r["final_answer_probability"] is not None}

    cursor.execute(
        """
        SELECT mo.question_id, mo.final_answer_probability
        FROM model_outputs mo
        WHERE mo.run_id = %s AND mo.perturbation_id IS NULL
        """,
        (current_run_id,)
    )
    current = {r["question_id"]: r["final_answer_probability"] for r in cursor.fetchall()
               if r["final_answer_probability"] is not None}

    cursor.close()
    conn.close()

    regressions = []
    for question_id, base_prob in baseline.items():
        cur_prob = current.get(question_id)
        if cur_prob is not None and (base_prob - cur_prob) > threshold:
            regressions.append({
                "question_id": question_id,
                "baseline_probability": base_prob,
                "current_probability": cur_prob,
                "drop": base_prob - cur_prob
            })

    return regressions


if __name__ == "__main__":
    pipeline = HookedModelPipeline(model_name="gpt2")
    try:
        results = track_regression(pipeline, run_id=1)
        print(results)
        regressions = compare_runs(baseline_run_id=1, current_run_id=2)
        print(regressions)
    finally:
        pipeline.clear_hooks()