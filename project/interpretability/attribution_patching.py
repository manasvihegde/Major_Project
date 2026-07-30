from db.connection import get_connection
from model_pipeline import HookedModelPipeline
import torch

pipeline = HookedModelPipeline("gpt2")

conn = get_connection()
cur = conn.cursor()

cur.execute("""
SELECT
    p.id,
    q.id,
    p.perturbed_text
FROM perturbations p
JOIN questions q
ON p.question_id=q.id
LIMIT 100
""")

rows = cur.fetchall()

for perturbation_id, question_id, text in rows:

    inputs = pipeline.tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    )

    with torch.no_grad():
        outputs = pipeline.model(
            **inputs,
            output_hidden_states=True
        )

    hidden = outputs.hidden_states[-1].squeeze(0)

    for token_index, token_id in enumerate(inputs["input_ids"][0]):

        token = pipeline.tokenizer.decode([token_id])

        activation = float(hidden[token_index].norm().item())

        cur.execute("""
        INSERT INTO xai_attribution_scores
        (
            run_id,
            question_id,
            perturbation_id,
            token_index,
            token_text,
            activation_patch_score
        )
        VALUES (%s,%s,%s,%s,%s,%s)
        """,
        (
            1,
            question_id,
            perturbation_id,
            token_index,
            token,
            activation
        ))

conn.commit()
cur.close()
conn.close()

print("Finished inserting attribution scores.")