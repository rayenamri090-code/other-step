import sqlite3
from config import DB_FILE

conn = sqlite3.connect(DB_FILE)
c = conn.cursor()

# collect unknown person_ids
c.execute("SELECT person_id FROM identities WHERE person_type = 'unknown'")
unknown_ids = [row[0] for row in c.fetchall()]

if unknown_ids:
    placeholders = ",".join("?" for _ in unknown_ids)

    c.execute(f"DELETE FROM face_embeddings WHERE person_id IN ({placeholders})", unknown_ids)
    c.execute(f"DELETE FROM visible_sessions WHERE person_id IN ({placeholders})", unknown_ids)
    c.execute(f"DELETE FROM access_sessions WHERE person_id IN ({placeholders})", unknown_ids)
    c.execute(f"DELETE FROM access_events WHERE person_id IN ({placeholders})", unknown_ids)
    c.execute(f"DELETE FROM alerts WHERE person_id IN ({placeholders})", unknown_ids)
    c.execute(f"DELETE FROM identities WHERE person_id IN ({placeholders})", unknown_ids)

    print(f"Deleted {len(unknown_ids)} unknown identities and related data.")
else:
    print("No unknown identities found.")

conn.commit()
conn.close()