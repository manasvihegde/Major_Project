"""
Week 3 - Manasvi Hegde

Token Importance Heatmap Visualizer

Reads token-level attribution scores from the
xai_attribution_scores table and generates a
heatmap showing token importance.

If no attribution data exists, exits gracefully.
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

from db.connection import get_connection


VALID_COLUMNS = {
    "activation_patch_score",
    "shap_score",
    "lime_score",
    "integrated_grad_score",
    "truth_score",
}


def fetch_attribution_scores(
    run_id: int,
    question_id: int,
    score_column: str = "activation_patch_score",
):
    """
    Fetch token attribution scores from the database.
    """

    if score_column not in VALID_COLUMNS:
        raise ValueError(
            f"score_column must be one of {VALID_COLUMNS}"
        )

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT
            token_index,
            token_text,
            {score_column}
        FROM xai_attribution_scores
        WHERE run_id=%s
          AND question_id=%s
        ORDER BY token_index
        """,
        (run_id, question_id),
    )

    rows = cursor.fetchall()

    cursor.close()
    conn.close()

    tokens = []
    scores = []

    for row in rows:
        tokens.append(row[1] if row[1] else "")
        scores.append(float(row[2]) if row[2] is not None else 0.0)

    return {
        "tokens": tokens,
        "scores": scores,
    }


def serialize_token_importance(
    run_id,
    question_id,
    filepath,
    score_column="activation_patch_score",
):
    """
    Save attribution scores as JSON.
    """

    data = fetch_attribution_scores(
        run_id,
        question_id,
        score_column,
    )

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)

    return data


def load_token_importance(filepath):

    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def visualize_token_heatmap(
    tokens,
    scores,
    out_path=None,
    figsize=(12, 2),
):
    """
    Draw token importance heatmap.
    """

    if len(tokens) == 0:
        print("\nNo attribution scores found.")
        print("xai_attribution_scores table is empty.")
        print("Skipping heatmap generation.\n")
        return

    scores_arr = np.array(scores).reshape(1, -1)

    fig, ax = plt.subplots(figsize=figsize)

    heatmap = ax.imshow(
        scores_arr,
        cmap="Reds",
        aspect="auto",
    )

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(
        tokens,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticks([])

    plt.colorbar(
        heatmap,
        orientation="horizontal",
        pad=0.35,
        label="Token Importance",
    )

    plt.tight_layout()

    if out_path:

        os.makedirs(
            os.path.dirname(out_path),
            exist_ok=True,
        )

        plt.savefig(
            out_path,
            dpi=150,
            bbox_inches="tight",
        )

        print(f"\nHeatmap saved to:\n{out_path}")

    else:
        plt.show()

    plt.close()


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--question_id",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--score_column",
        default="activation_patch_score",
        choices=sorted(VALID_COLUMNS),
    )

    args = parser.parse_args()

    json_path = "data/outputs/token_importance.json"
    image_path = "data/outputs/heatmap.png"

    data = serialize_token_importance(
        args.run_id,
        args.question_id,
        json_path,
        args.score_column,
    )

    print(
        f"Loaded {len(data['tokens'])} attributed tokens "
        f"for Question {args.question_id}"
    )

    visualize_token_heatmap(
        data["tokens"],
        data["scores"],
        image_path,
    )


if __name__ == "__main__":
    main()