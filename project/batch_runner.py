"""
batch_runner.py

Week 2 - "Run batch jobs to categorize structural instability" (Diya)

Reads model_outputs already logged by comparative_logger.py for a given
run_id, computes output-level edit distance between each question's
original generation and each of its perturbation generations, categorizes
the result using INSTABILITY_THRESHOLDS (perturbation_config.py), and
writes one row per (question, perturbation) into instability_categories.

Usage:
    python batch_runner.py --run_id 3
"""

import argparse
from difflib import SequenceMatcher

from db.connection import get_connection
from perturbation_config import INSTABILITY_THRESHOLDS


def categorize_instability(edit_ratio: float) -> str:
    if edit_ratio >= INSTABILITY_THRESHOLDS["structural"]["edit_distance_min"]:
        return "structural"
    elif edit_ratio > 0:
        return "semantic"
    return "stable"


def compute_edit_ratio(text_a: str, text_b: str) -> float:
    if not text_a or not text_b:
        return 1.0
    return 1 - SequenceMatcher(None, text_a, text_b).ratio()


def fetch_question_ids_for_run(conn, run_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT DISTINCT question_id FROM model_outputs WHERE run_id = %s;
    """, (run_id,))
    rows = cur.fetchall()
    cur.close()
    return [r[0] for r in rows]


def fetch_original_output(conn, run_id, question_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT reasoning_chain FROM model_outputs
        WHERE run_id = %s AND question_id = %s AND perturbation_id IS NULL;
    """, (run_id, question_id))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def fetch_perturbed_outputs(conn, run_id, question_id):
    """Returns list of (perturbation_id, perturbation_type, reasoning_chain)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT mo.perturbation_id, p.perturbation_type, mo.reasoning_chain
        FROM model_outputs mo
        JOIN perturbations p ON p.id = mo.perturbation_id
        WHERE mo.run_id = %s AND mo.question_id = %s AND mo.perturbation_id IS NOT NULL;
    """, (run_id, question_id))
    rows = cur.fetchall()
    cur.close()
    return rows


def fetch_semantic_score(conn, run_id, question_id, perturbation_type):
    """Pulls the matching column from stability_scores for cross-reference."""
    column_map = {
        "prefix": "score_prefix",
        "name_swap": "score_name_swap",
        "synonym": "score_synonym",
        "whitespace": "score_whitespace",
    }
    col = column_map.get(perturbation_type)
    if not col:
        return None

    cur = conn.cursor()
    cur.execute(f"""
        SELECT {col} FROM stability_scores
        WHERE run_id = %s AND question_id = %s;
    """, (run_id, question_id))
    row = cur.fetchone()
    cur.close()
    return row[0] if row else None


def log_instability_category(conn, run_id, question_id, perturbation_id,
                              perturbation_type, output_edit_ratio,
                              semantic_similarity, category):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO instability_categories
            (run_id, question_id, perturbation_id, perturbation_type,
             output_edit_ratio, semantic_similarity, instability_category)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, question_id, perturbation_id) DO UPDATE SET
            output_edit_ratio = EXCLUDED.output_edit_ratio,
            semantic_similarity = EXCLUDED.semantic_similarity,
            instability_category = EXCLUDED.instability_category;
    """, (
        run_id, question_id, perturbation_id, perturbation_type,
        round(output_edit_ratio, 4),
        round(semantic_similarity, 4) if semantic_similarity is not None else None,
        category
    ))
    conn.commit()
    cur.close()


def run_batch_categorization(run_id):
    conn = get_connection()
    question_ids = fetch_question_ids_for_run(conn, run_id)
    print(f"Categorizing instability for run_id={run_id}, {len(question_ids)} questions...")

    counts = {"structural": 0, "semantic": 0, "stable": 0}

    try:
        for qid in question_ids:
            original_text = fetch_original_output(conn, run_id, qid)
            if original_text is None:
                print(f"  Q{qid}: no original output found, skipping")
                continue

            perturbed_outputs = fetch_perturbed_outputs(conn, run_id, qid)

            for pid, ptype, perturbed_text in perturbed_outputs:
                edit_ratio = compute_edit_ratio(original_text, perturbed_text)
                category = categorize_instability(edit_ratio)
                semantic_score = fetch_semantic_score(conn, run_id, qid, ptype)

                log_instability_category(
                    conn, run_id, qid, pid, ptype,
                    edit_ratio, semantic_score, category
                )
                counts[category] += 1

            print(f"  Q{qid}: {len(perturbed_outputs)} perturbations categorized")

        print(f"Done. Structural={counts['structural']}, "
              f"Semantic={counts['semantic']}, Stable={counts['stable']}")

    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True,
                         help="run_logs.id from a completed comparative_logger.py run")
    args = parser.parse_args()

    run_batch_categorization(args.run_id)