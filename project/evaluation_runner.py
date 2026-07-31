"""
evaluation_runner.py

Week 2 - "Analyze text outputs for semantic vs structural shifts" /
          "Isolate highly unstable target samples" (Kushi S Shetty)

NOTE (refactor): Categorization itself (structural / semantic / stable) is
now owned entirely by batch_runner.py (Diya), which writes to
instability_categories using the shared INSTABILITY_THRESHOLDS in
perturbation_config.py. This script no longer recomputes categories or
writes to instability_categories — it only READS what batch_runner.py
already wrote, joins it with stability_scores, and surfaces the most
unstable question/perturbation pairs.

Pipeline order this script depends on, for a given run_id:
    1. comparative_logger.py  -> populates model_outputs, stability_scores
    2. batch_runner.py        -> populates instability_categories
    3. evaluation_runner.py   -> reads + reports (this file)

Usage:
    python evaluation_runner.py --run_id 3
    python evaluation_runner.py            # analyzes all completed runs
"""

import argparse
import csv

from db.connection import get_connection

# ============================================================
# Configuration
# ============================================================

CSV_FILENAME = "evaluation_runner.csv"

# ============================================================
# CSV FUNCTIONS
# ============================================================

def initialize_csv():
    with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "Run ID",
            "Question ID",
            "Perturbation ID",
            "Perturbation Type",
            "Output Edit Ratio",
            "Semantic Similarity",
            "Instability Category",
        ])


def write_csv_row(run_id, question_id, perturbation_id, perturbation_type,
                   output_edit_ratio, semantic_similarity, category):
    with open(CSV_FILENAME, "a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            run_id,
            question_id,
            perturbation_id,
            perturbation_type,
            round(output_edit_ratio, 4) if output_edit_ratio is not None else "",
            round(semantic_similarity, 4) if semantic_similarity is not None else "",
            category,
        ])


# ============================================================
# DATABASE READS
# ============================================================

def fetch_all_runs(conn):
    """Returns all completed experiment runs."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id
        FROM run_logs
        WHERE status = 'done'
        ORDER BY id;
    """)
    runs = [row[0] for row in cur.fetchall()]
    cur.close()
    return runs


def fetch_categorized_results(conn, run_id):
    """
    Reads instability_categories (written by batch_runner.py) for a run,
    ordered so the most unstable samples come first.

    Returns rows of:
        (question_id, perturbation_id, perturbation_type,
         output_edit_ratio, semantic_similarity, instability_category)
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            question_id,
            perturbation_id,
            perturbation_type,
            output_edit_ratio,
            semantic_similarity,
            instability_category
        FROM instability_categories
        WHERE run_id = %s
        ORDER BY
            CASE instability_category
                WHEN 'structural' THEN 0
                WHEN 'semantic' THEN 1
                ELSE 2
            END,
            output_edit_ratio DESC;
    """, (run_id,))
    rows = cur.fetchall()
    cur.close()
    return rows


# ============================================================
# CONSOLE OUTPUT
# ============================================================

def print_result(run_id, question_id, perturbation_type, output_edit_ratio,
                  semantic_similarity, category):
    edit_str = f"{output_edit_ratio:.4f}" if output_edit_ratio is not None else "  N/A"
    sem_str = f"{semantic_similarity:.4f}" if semantic_similarity is not None else "  N/A"
    print(
        f"[Run {run_id}] "
        f"Q{question_id:<4} | "
        f"{perturbation_type:<11} | "
        f"EditDist={edit_str} | "
        f"Semantic={sem_str} | "
        f"{category.upper()}"
    )


# ============================================================
# ANALYSIS
# ============================================================

def analyze_run(conn, run_id, top_n_unstable=10):
    """
    Reads pre-computed instability categories for a run (from
    batch_runner.py) and reports on them. Does NOT recompute or write
    categories — that is batch_runner.py's job.
    """
    results = fetch_categorized_results(conn, run_id)

    if not results:
        print(f"\nRun {run_id}: no instability_categories rows found. "
              f"Has batch_runner.py been run for this run_id yet?")
        return

    print(f"\nAnalyzing Run {run_id}")
    print(f"Categorized Rows Found : {len(results)}")

    counts = {"structural": 0, "semantic": 0, "stable": 0}

    for (question_id, perturbation_id, perturbation_type,
         output_edit_ratio, semantic_similarity, category) in results:

        write_csv_row(
            run_id, question_id, perturbation_id, perturbation_type,
            output_edit_ratio, semantic_similarity, category
        )

        print_result(
            run_id, question_id, perturbation_type,
            output_edit_ratio, semantic_similarity, category
        )

        counts[category] = counts.get(category, 0) + 1

    print("\n==========================================")
    print(f"Run {run_id} Summary")
    print("==========================================")
    print(f"Total Comparisons : {len(results)}")
    print(f"Stable            : {counts.get('stable', 0)}")
    print(f"Structural        : {counts.get('structural', 0)}")
    print(f"Semantic          : {counts.get('semantic', 0)}")
    print("==========================================")

    print(f"\nTop {top_n_unstable} Most Unstable Samples (Run {run_id}):")
    unstable_only = [r for r in results if r[5] != "stable"]
    for row in unstable_only[:top_n_unstable]:
        (question_id, perturbation_id, perturbation_type,
         output_edit_ratio, semantic_similarity, category) = row
        edit_str = f"{output_edit_ratio:.4f}" if output_edit_ratio is not None else "N/A"
        print(f"  Q{question_id} / pert {perturbation_id} ({perturbation_type}) "
              f"-> {category.upper()}, edit_dist={edit_str}")
    print()


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=int, default=None,
                         help="Specific run_logs.id to analyze. "
                              "If omitted, analyzes all completed runs.")
    parser.add_argument("--top_n", type=int, default=10,
                         help="How many top unstable samples to print per run.")
    args = parser.parse_args()

    print("\n==========================================")
    print("Week 2 - Instability Analysis & Reporting")
    print("==========================================")

    initialize_csv()

    conn = get_connection()

    try:
        if args.run_id is not None:
            runs = [args.run_id]
        else:
            runs = fetch_all_runs(conn)

        if not runs:
            print("No completed runs found.")
            return

        print(f"\nRuns To Analyze : {len(runs)}")

        for run_id in runs:
            analyze_run(conn, run_id, top_n_unstable=args.top_n)

        print("\n==========================================")
        print("Analysis Completed Successfully")
        print("==========================================")
        print(f"CSV Exported : {CSV_FILENAME}")

    except Exception as e:
        print("\nAnalysis Failed")
        print(e)

    finally:
        conn.close()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()