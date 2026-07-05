import os
from dotenv import load_dotenv
load_dotenv()

from parsers.gsm8k_parser      import GSM8KParser
#from parsers.hotpotqa_parser   import HotpotQAParser
#from parsers.strategyqa_parser import StrategyQAParser
from db.connection import get_dict_connection


def run_all_parsers(num_samples_each=100):
    parsers = [
        GSM8KParser(split="train"),
        #HotpotQAParser(split="train"),
        #StrategyQAParser(split="train"),
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
    cur  = conn.cursor()

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
        print(f"  {r['name']:15s} ({r['split']:5s}) : {r['total_questions']} questions")

    cur.execute("""
        SELECT perturbation_type, COUNT(*) AS total
        FROM perturbations
        GROUP BY perturbation_type
        ORDER BY perturbation_type;
    """)
    rows = cur.fetchall()
    print("\nPerturbations by type:")
    for r in rows:
        print(f"  {r['perturbation_type']:15s} : {r['total']}")

    cur.execute("""
        SELECT question_type, COUNT(*) AS total
        FROM questions
        GROUP BY question_type
        ORDER BY question_type;
    """)
    rows = cur.fetchall()
    print("\nQuestions by type:")
    for r in rows:
        print(f"  {r['question_type']:30s} : {r['total']}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    run_all_parsers(num_samples_each=100)
    verify_data()