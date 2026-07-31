from db.connection import get_connection

conn = get_connection()
cur = conn.cursor()

cur.execute("""
SELECT conname,
       pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'questions'::regclass;
""")

print("Questions table:")
for row in cur.fetchall():
    print(row)

print()

cur.execute("""
SELECT conname,
       pg_get_constraintdef(oid)
FROM pg_constraint
WHERE conrelid = 'perturbations'::regclass;
""")

print("Perturbations table:")
for row in cur.fetchall():
    print(row)

conn.close()