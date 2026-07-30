"""
Week 3 - Kushi S Shetty
Cross-reference heatmaps to pinpoint Plausibility Trap.

Falls back gracefully when the attribution pipeline has not yet populated
xai_attribution_scores.
"""

import argparse
import csv
from db.connection import get_dict_connection

CSV_FILENAME = "plausibility_trap_report.csv"

CAUSAL_SCORE_COLUMN = "activation_patch_score"
CAUSAL_SCORE_THRESHOLD = 0.50
TRUTH_SCORE_THRESHOLD = 0.50


def attribution_table_has_rows(conn):
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) AS cnt FROM xai_attribution_scores")
    count = cursor.fetchone()["cnt"]
    cursor.close()
    return count > 0


def fetch_candidate_tokens(conn, run_id):
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT
            question_id,
            perturbation_id,
            token_index,
            token_text,
            {CAUSAL_SCORE_COLUMN} AS causal_score,
            truth_score,
            faithfulness_gap,
            is_in_stated_explanation
        FROM xai_attribution_scores
        WHERE run_id=%s
          AND {CAUSAL_SCORE_COLUMN}>=%s
          AND is_in_stated_explanation=FALSE
        ORDER BY {CAUSAL_SCORE_COLUMN} DESC
        """,
        (run_id, CAUSAL_SCORE_THRESHOLD),
    )

    rows = cursor.fetchall()
    cursor.close()
    return rows


def classify_trap(row):
    truth = row["truth_score"]

    if truth is not None and truth < TRUTH_SCORE_THRESHOLD:
        return "critical"

    return "moderate"


def create_empty_csv():
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow([
            "Run ID",
            "Question ID",
            "Perturbation ID",
            "Token Index",
            "Token",
            "Activation Score",
            "Truth Score",
            "Faithfulness Gap",
            "Severity"
        ])


def run_analysis(run_id):

    conn = get_dict_connection()

    if not attribution_table_has_rows(conn):
        create_empty_csv()

        print("\n===================================")
        print("xai_attribution_scores is EMPTY")
        print("Janvi's attribution patching step")
        print("has not been executed yet.")
        print("Generated empty plausibility_trap_report.csv")
        print("===================================")

        conn.close()
        return

    rows = fetch_candidate_tokens(conn, run_id)
    conn.close()

    if len(rows) == 0:
        create_empty_csv()

        print(f"\nRun {run_id}: No plausibility trap candidates found.")
        print("CSV generated.")

        return

    critical = 0
    moderate = 0

    with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:

        writer = csv.writer(f)

        writer.writerow([
            "Run ID",
            "Question ID",
            "Perturbation ID",
            "Token Index",
            "Token",
            "Activation Score",
            "Truth Score",
            "Faithfulness Gap",
            "Severity"
        ])

        for row in rows:

            severity = classify_trap(row)

            if severity == "critical":
                critical += 1
            else:
                moderate += 1

            writer.writerow([
                run_id,
                row["question_id"],
                row["perturbation_id"],
                row["token_index"],
                row["token_text"],
                round(row["causal_score"], 4),
                round(row["truth_score"], 4) if row["truth_score"] is not None else "",
                round(row["faithfulness_gap"], 4) if row["faithfulness_gap"] is not None else "",
                severity
            ])

    print("\n===================================")
    print(f"Run {run_id}")
    print("===================================")
    print(f"Candidates : {len(rows)}")
    print(f"Critical   : {critical}")
    print(f"Moderate   : {moderate}")
    print(f"Saved      : {CSV_FILENAME}")
    print("===================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, required=True)

    args = parser.parse_args()

    run_analysis(args.run_id)