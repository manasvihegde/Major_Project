"""
baseline_eval.py

Runs baseline benchmarking directly against the database (questions table)
instead of curated_reasoning_dataset.json, so it works regardless of which
path populated the data:
    - curate_dataset.py -> CuratedDatasetParser (dataset_name='curated_reasoning')
    - GSM8KParser + StrategyQAParser + HotpotQAParser (separate dataset_names)

Usage:
    python baseline_eval.py                          # all datasets
    python baseline_eval.py --dataset gsm8k           # just one dataset
    python baseline_eval.py --limit 50
"""

import argparse
import json
import re

from db.connection import get_dict_connection
from sentence_transformers import SentenceTransformer, util


EMBEDDER_MODEL = "all-MiniLM-L6-v2"
SEMANTIC_MATCH_THRESHOLD = 0.75  # cosine similarity considered a "correct" match for free-text answers


def call_model_backend_stub(prompt):
    """
    Mock interface wrapper.
    Replace with direct execution hooks calling Janvi's vLLM/Inference API.
    """
    return "Step 1: Extrapolate variables. Step 2: Formulate solution. The final answer is 42."


def fetch_questions(conn, dataset_name=None, limit=None):
    """
    Pulls questions (+ their question_type, so we know how to score them)
    from the DB. If dataset_name is given, restricts to that dataset;
    otherwise pulls across all datasets.
    """
    cur = conn.cursor()

    if dataset_name:
        query = """
            SELECT q.id, q.question_text, q.ground_truth_answer, q.question_type, d.name AS dataset_name
            FROM questions q
            JOIN datasets d ON d.id = q.dataset_id
            WHERE d.name = %s
            ORDER BY q.id
        """
        params = (dataset_name,)
    else:
        query = """
            SELECT q.id, q.question_text, q.ground_truth_answer, q.question_type, d.name AS dataset_name
            FROM questions q
            JOIN datasets d ON d.id = q.dataset_id
            ORDER BY q.id
        """
        params = ()

    if limit:
        query += " LIMIT %s"
        params = params + (limit,)

    cur.execute(query, params)
    rows = cur.fetchall()
    cur.close()
    return rows


def normalize(text: str) -> str:
    return re.sub(r'[^a-zA-Z0-9]', '', text.lower())


def is_correct(question_type, ground_truth, generated_text, embedder):
    """
    Scoring strategy differs by question_type:
      - math / boolean: exact-ish substring match on normalized text
        (numbers and "True"/"False" don't need semantic matching)
      - multi_hop: cosine similarity, since HotpotQA's free-text answers
        may be paraphrased rather than quoted verbatim by the model
    """
    if question_type == "multi_hop":
        truth_emb = embedder.encode(ground_truth, convert_to_tensor=True)
        pred_emb = embedder.encode(generated_text, convert_to_tensor=True)
        similarity = util.cos_sim(truth_emb, pred_emb).item()
        return similarity >= SEMANTIC_MATCH_THRESHOLD, similarity

    normalized_truth = normalize(ground_truth)
    normalized_pred = normalize(generated_text)
    return normalized_truth in normalized_pred, None


def run_baseline_benchmarking(dataset_name=None, limit=None):
    conn = get_dict_connection()
    print(f"📊 Fetching questions from DB "
          f"({'dataset=' + dataset_name if dataset_name else 'all datasets'})...")

    rows = fetch_questions(conn, dataset_name=dataset_name, limit=limit)
    conn.close()

    total_records = len(rows)
    if total_records == 0:
        print("⚠️  No questions found in the database. "
              "Have you run curate_dataset.py + run_parsers.py, or run_parsers.py's fallback parsers?")
        return

    print(f"⚙️ Running baseline passes across {total_records} questions...")

    embedder = SentenceTransformer(EMBEDDER_MODEL)

    aggregate_word_count = 0
    correct_matches = 0

    per_dataset_stats = {}

    for row in rows:
        prompt = row["question_text"]
        ground_truth = row["ground_truth_answer"]
        question_type = row["question_type"]
        ds_name = row["dataset_name"]

        generated_explanation = call_model_backend_stub(prompt)

        words = generated_explanation.split()
        aggregate_word_count += len(words)

        correct, similarity = is_correct(question_type, ground_truth, generated_explanation, embedder)
        if correct:
            correct_matches += 1

        stats = per_dataset_stats.setdefault(ds_name, {"total": 0, "correct": 0})
        stats["total"] += 1
        if correct:
            stats["correct"] += 1

    mean_length = aggregate_word_count / total_records
    computed_accuracy = (correct_matches / total_records) * 100

    print("\n📈 ======= BASELINE METRIC MATRIX =======")
    print(f"Average Generation Length : {mean_length:.2f} words")
    print(f"Overall Baseline Accuracy : {computed_accuracy:.2f}%")
    print("-----------------------------------------")
    for ds_name, stats in per_dataset_stats.items():
        ds_accuracy = (stats["correct"] / stats["total"]) * 100
        print(f"  {ds_name:15s} : {stats['correct']}/{stats['total']} ({ds_accuracy:.2f}%)")
    print("===========================================\n")

    telemetry_payload = {
        "dataset_filter": dataset_name or "all",
        "sample_size": total_records,
        "metrics": {
            "avg_explanation_length_words": mean_length,
            "baseline_accuracy_percentage": computed_accuracy,
        },
        "per_dataset": {
            ds_name: {
                "total": stats["total"],
                "correct": stats["correct"],
                "accuracy_percentage": round((stats["correct"] / stats["total"]) * 100, 2),
            }
            for ds_name, stats in per_dataset_stats.items()
        },
    }

    with open("baseline_metrics_log.json", "w", encoding="utf-8") as log_out:
        json.dump(telemetry_payload, log_out, indent=4)
    print("💾 Telemetry logs securely written to 'baseline_metrics_log.json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None,
                         help="Restrict to one datasets.name value "
                              "(e.g. 'gsm8k', 'strategyqa', 'hotpotqa', 'curated_reasoning'). "
                              "Omit to run across all datasets.")
    parser.add_argument("--limit", type=int, default=None,
                         help="Max number of questions to evaluate.")
    args = parser.parse_args()

    run_baseline_benchmarking(dataset_name=args.dataset, limit=args.limit)