import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

def init_database(schema_path="db/schema.sql", reset=False):
    """
    Reads schema.sql and executes it against the database.
    Set reset=True to drop and recreate all tables.
    """
    conn = psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", 5432),
        dbname=os.getenv("DB_NAME", "llm_reasoning_logs"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD")
    )
    conn.autocommit = True
    cur = conn.cursor()

    if reset:
        confirm = input("This will DROP all tables. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    with open(schema_path, "r") as f:
        sql = f.read()

    cur.execute(sql)
    print("Schema applied successfully.")

    # Verify tables were created
    cur.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        ORDER BY table_name;
    """)
    tables = cur.fetchall()
    print(f"\nTables created ({len(tables)}):")
    for t in tables:
        print(f"  - {t[0]}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    init_database(reset=True)