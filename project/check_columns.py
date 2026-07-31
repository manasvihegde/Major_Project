from db.connection import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
SELECT column_name
FROM information_schema.columns
WHERE table_name = 'model_outputs';
""")

columns = [row[0] for row in cur.fetchall()]
print(columns)

conn.close()