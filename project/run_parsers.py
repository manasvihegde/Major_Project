import os
from dotenv import load_dotenv

load_dotenv()

from parsers.gsm8k_parser import GSM8KParser
from parsers.curated_dataset_parser import CuratedDatasetParser
from db.connection import get_dict_connection


CURATED_JSON = "curated_reasoning_dataset.json"


def run_all_parsers(num_samples_each=100):

    if os.path.exists(CURATED_JSON):

        print("\nUsing curated reasoning dataset.")

        parsers = [
            CuratedDatasetParser(CURATED_JSON)
        ]

    else:

        print("\nCurated dataset not found.")
        print("Falling back to HuggingFace GSM8K.\n")

        parsers = [
            GSM8KParser(split="train")
        ]

    for parser in parsers:

        try:
            parser.parse(num_samples=num_samples_each)

        except Exception as e:
            print(f"ERROR in {parser.dataset_name}: {e}")

        finally:
            parser.close()


def verify_data():

    conn = get_dict_connection()
    cur = conn.cursor()

    print("\n========= DATABASE SUMMARY =========")

    cur.execute("""
        SELECT d.name, d.split, COUNT(q.id) AS total_questions
        FROM datasets d
        LEFT JOIN questions q ON q.dataset_id = d.id
        GROUP BY d.name, d.split
        ORDER BY d.name;
    """)

    rows = cur.fetchall()

    print("\nQuestions per dataset:")

    for r in rows:
        print(f"{r['name']:20s} {r['total_questions']}")

    cur.close()
    conn.close()


if __name__ == "__main__":

    run_all_parsers(num_samples_each=100)
    verify_data()