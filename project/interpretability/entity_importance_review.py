"""
Week 3 - Diya

Review Token Importance Maps for Key Entities

Cross-references xai_attribution_scores against important entities
(names, numbers, proper nouns) appearing in each question to determine
whether attribution methods actually focus on the tokens that matter.

Produces:
    entity_importance_review.csv
"""

import argparse
import csv
import re

from db.connection import get_dict_connection

CSV_FILENAME = "entity_importance_review.csv"

VALID_COLUMNS = {
    "activation_patch_score",
    "shap_score",
    "lime_score",
    "integrated_grad_score",
    "truth_score",
}

ENTITY_PATTERN = re.compile(
    r"[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*|\b\d+(?:\.\d+)?\b"
)


def extract_key_entities(question_text: str):
    """
    Extract probable entities using a lightweight regex.

    Captures:
        • Proper nouns
        • Multi-word names
        • Numbers
    """

    matches = ENTITY_PATTERN.findall(question_text)

    ignored = {
        "The",
        "A",
        "An",
        "Is",
        "Are",
        "Was",
        "Were",
        "Would",
        "Should",
        "Could",
        "Can",
        "Will",
        "If",
        "When",
        "What",
        "Which",
        "Who",
        "Why",
        "How",
    }

    seen = set()
    entities = []

    for match in matches:

        entity = match.strip()

        if not entity:
            continue

        if entity in ignored:
            continue

        if entity in seen:
            continue

        seen.add(entity)
        entities.append(entity)

    return entities


def fetch_attribution_scores(
    conn,
    run_id,
    question_id,
    score_column="activation_patch_score",
):
    """
    Fetch token-level attribution scores.
    """

    if score_column not in VALID_COLUMNS:
        raise ValueError(
            f"score_column must be one of {VALID_COLUMNS}"
        )

    cursor = conn.cursor()

    cursor.execute(
        f"""
        SELECT
            token_index,
            token_text,
            {score_column} AS score
        FROM xai_attribution_scores
        WHERE run_id=%s
          AND question_id=%s
        ORDER BY token_index
        """,
        (
            run_id,
            question_id,
        ),
    )

    rows = cursor.fetchall()

    cursor.close()

    return rows


def match_entity_to_tokens(entity, token_rows):
    """
    Match an extracted entity against consecutive tokens.

    Supports multi-word entities.
    """

    entity = entity.lower().replace(" ", "")

    matched = []

    tokens = [
        (row["token_text"] or "")
        for row in token_rows
    ]

    n = len(tokens)

    for start in range(n):

        window = ""

        for end in range(start, min(start + 5, n)):

            window += tokens[end].strip().lower()

            if window == entity:

                matched.extend(
                    token_rows[start:end + 1]
                )

                break

            if len(window) > len(entity):
                break

    return matched


def review_entities_for_question(
    conn,
    run_id,
    question_id,
    question_text,
    score_column="activation_patch_score",
):
    """
    Compute attribution statistics for each detected entity.
    """

    entities = extract_key_entities(question_text)

    token_rows = fetch_attribution_scores(
        conn,
        run_id,
        question_id,
        score_column,
    )

    if not token_rows:

        print(
            f"Q{question_id}: "
            f"No attribution rows found."
        )

        return []

    results = []

    for entity in entities:

        matched_tokens = match_entity_to_tokens(
            entity,
            token_rows,
        )

        if not matched_tokens:
            continue

        scores = [
            token["score"]
            for token in matched_tokens
            if token["score"] is not None
        ]

        if not scores:
            continue

        results.append(
            {
                "question_id": question_id,
                "entity": entity,
                "matched_tokens": len(matched_tokens),
                "max_score": max(scores),
                "avg_score": sum(scores) / len(scores),
            }
        )

    results.sort(
        key=lambda x: x["max_score"],
        reverse=True,
    )

    return results
def run_entity_review(
    run_id: int,
    limit: int = 50,
    score_column: str = "activation_patch_score"
):
    conn = get_dict_connection()
    cursor = conn.cursor()

    # Check whether the XAI table has been populated
    cursor.execute(
        "SELECT COUNT(*) AS cnt FROM xai_attribution_scores WHERE run_id=%s",
        (run_id,)
    )
    count = cursor.fetchone()["cnt"]

    if count == 0:
        print("\n==========================================")
        print("No attribution data found.")
        print("==========================================")
        print(
            "The table 'xai_attribution_scores' is empty for "
            f"run_id={run_id}."
        )
        print("\nRun the Attribution Patching pipeline first.")
        print("Skipping entity importance review.")
        cursor.close()
        conn.close()
        return []

    cursor.execute(
        """
        SELECT id, question_text
        FROM questions
        ORDER BY id
        LIMIT %s
        """,
        (limit,)
    )

    questions = cursor.fetchall()
    cursor.close()

    all_results = []

    for q in questions:
        results = review_entities_for_question(
            conn,
            run_id,
            q["id"],
            q["question_text"],
            score_column,
        )
        all_results.extend(results)

    conn.close()

    with open(
        CSV_FILENAME,
        "w",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "Question ID",
            "Entity",
            "Matched Tokens",
            "Max Score",
            "Avg Score",
        ])

        for r in all_results:
            writer.writerow([
                r["question_id"],
                r["entity"],
                r["matched_tokens"],
                round(r["max_score"], 4),
                round(r["avg_score"], 4),
            ])

    print("\n==========================================")
    print("Entity Importance Review Complete")
    print("==========================================")
    print(f"Questions processed : {len(questions)}")
    print(f"Entity matches      : {len(all_results)}")
    print(f"CSV saved           : {CSV_FILENAME}")
    print("==========================================")

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--run_id",
        type=int,
        default=1,
        help="Run ID to analyze",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of questions",
    )

    parser.add_argument(
        "--score_column",
        default="activation_patch_score",
        choices=[
            "activation_patch_score",
            "shap_score",
            "lime_score",
            "integrated_grad_score",
            "truth_score",
        ],
        help="Attribution score column",
    )

    args = parser.parse_args()

    run_entity_review(
        run_id=args.run_id,
        limit=args.limit,
        score_column=args.score_column,
    )