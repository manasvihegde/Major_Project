"""
comparative_logger.py

Week 2 - "Comparative data logging system for divergent text" (Manasvi)

For each question in the DB:
  1. Run the model on the original question text
  2. Run the model on each of its perturbation variants
  3. Log every generation to model_outputs
  4. Compute cosine similarity (via sentence-transformers) between the
     original output and each perturbed output
  5. Log the per-type + aggregate similarity to stability_scores

Usage:
    python comparative_logger.py --num_questions 20 --model gpt2
"""

import argparse
import time
from db.connection import get_connection, get_dict_connection
from model_pipeline import HookedModelPipeline
from perturbation_config import DEFAULT_EMBEDDER_MODEL

from sentence_transformers import SentenceTransformer, util


PERTURBATION_TYPE_COLUMN = {
    "prefix": "score_prefix",
    "name_swap": "score_name_swap",
    "synonym": "score_synonym",
    "whitespace": "score_whitespace",
}


def classify_stability(score):
    if score >= 0.85:
        return "stable"
    elif score >= 0.6:
        return "moderate"
    else:
        return "unstable"


def create_run(conn, model_name, max_new_tokens, device):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO run_logs (model_name, run_description, max_new_tokens,
                               decoding, device, status)
        VALUES (%s, %s, %s, %s, %s, 'running')
        RETURNING id;
    """, (model_name, "Comparative stability logging run", max_new_tokens, "greedy", device))
    run_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return run_id


def complete_run(conn, run_id, status="done"):
    cur = conn.cursor()
    cur.execute("""
        UPDATE run_logs SET completed_at = NOW(), status = %s WHERE id = %s;
    """, (status, run_id))
    conn.commit()
    cur.close()


def fetch_questions(conn, limit):
    cur = conn.cursor()
    cur.execute("SELECT id, question_text FROM questions ORDER BY id LIMIT %s;", (limit,))
    rows = cur.fetchall()
    cur.close()
    return rows  # list of (id, question_text)


def fetch_perturbations(conn, question_id):
    cur = conn.cursor()
    cur.execute("""
        SELECT id, perturbation_type, perturbed_text
        FROM perturbations WHERE question_id = %s;
    """, (question_id,))
    rows = cur.fetchall()
    cur.close()
    return rows  # list of (id, type, text)


def log_model_output(conn, run_id, question_id, perturbation_id, input_text,
                      generated_text, gen_time_ms):
    cur = conn.cursor()
    cur.execute("""
        INSERT INTO model_outputs
            (run_id, question_id, perturbation_id, input_text,
             reasoning_chain, token_count, generation_time_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        RETURNING id;
    """, (
        run_id, question_id, perturbation_id, input_text,
        generated_text, len(generated_text.split()), gen_time_ms
    ))
    output_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    return output_id


def log_stability_score(conn, run_id, question_id, per_type_scores, embedder_model):
    """
    FIX: stability_scores.overall_stability is NOT NULL in schema.sql.
    If a question has no perturbations (or none matched a known type),
    valid_scores is empty and overall would previously be None, causing
    an IntegrityError and killing the whole run. We now skip logging a
    row entirely in that case instead of inserting a NULL.
    """
    valid_scores = [v for v in per_type_scores.values() if v is not None]

    if not valid_scores:
        print(f"  Q{question_id}: no valid perturbation scores, "
              f"skipping stability_scores row")
        return None, None

    overall = sum(valid_scores) / len(valid_scores)
    category = classify_stability(overall)

    cur = conn.cursor()
    cur.execute("""
        INSERT INTO stability_scores
            (run_id, question_id, score_prefix, score_name_swap,
             score_synonym, score_whitespace, overall_stability,
             stability_category, embedder_model)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (run_id, question_id) DO UPDATE SET
            score_prefix = EXCLUDED.score_prefix,
            score_name_swap = EXCLUDED.score_name_swap,
            score_synonym = EXCLUDED.score_synonym,
            score_whitespace = EXCLUDED.score_whitespace,
            overall_stability = EXCLUDED.overall_stability,
            stability_category = EXCLUDED.stability_category;
    """, (
        run_id, question_id,
        per_type_scores.get("prefix"),
        per_type_scores.get("name_swap"),
        per_type_scores.get("synonym"),
        per_type_scores.get("whitespace"),
        overall, category, embedder_model
    ))
    conn.commit()
    cur.close()
    return overall, category


def run_comparative_logging(num_questions=20, model_name="gpt2", max_new_tokens=50):
    conn = get_connection()
    pipeline = HookedModelPipeline(model_name=model_name)
    embedder = SentenceTransformer(DEFAULT_EMBEDDER_MODEL)

    run_id = create_run(conn, model_name, max_new_tokens, pipeline.device)
    print(f"Started run_id={run_id}")

    questions = fetch_questions(conn, num_questions)
    print(f"Processing {len(questions)} questions...")

    try:
        for qid, question_text in questions:
            # 1. Original generation
            t0 = time.time()
            original_result = pipeline.generate_with_hooks(question_text, max_new_tokens)
            orig_time_ms = int((time.time() - t0) * 1000)
            log_model_output(
                conn, run_id, qid, None, question_text,
                original_result["generated_text"], orig_time_ms
            )
            original_embedding = embedder.encode(
                original_result["generated_text"], convert_to_tensor=True
            )

            # 2. Perturbation generations
            per_type_scores = {}
            perturbations = fetch_perturbations(conn, qid)

            for pid, ptype, ptext in perturbations:
                t1 = time.time()
                pert_result = pipeline.generate_with_hooks(ptext, max_new_tokens)
                pert_time_ms = int((time.time() - t1) * 1000)
                log_model_output(
                    conn, run_id, qid, pid, ptext,
                    pert_result["generated_text"], pert_time_ms
                )

                pert_embedding = embedder.encode(
                    pert_result["generated_text"], convert_to_tensor=True
                )
                similarity = util.cos_sim(original_embedding, pert_embedding).item()

                col = PERTURBATION_TYPE_COLUMN.get(ptype)
                if col:
                    per_type_scores[ptype] = round(similarity, 4)

            overall, category = log_stability_score(
                conn, run_id, qid, per_type_scores, DEFAULT_EMBEDDER_MODEL
            )
            print(f"  Q{qid}: overall_stability={overall} ({category})")

        complete_run(conn, run_id, status="done")
        print(f"Run {run_id} complete.")

    except Exception as e:
        complete_run(conn, run_id, status="failed")
        print(f"Run {run_id} failed: {e}")
        raise
    finally:
        pipeline.clear_hooks()
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_questions", type=int, default=20)
    parser.add_argument("--model", type=str, default="gpt2")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    args = parser.parse_args()

    run_comparative_logging(
        num_questions=args.num_questions,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
    )