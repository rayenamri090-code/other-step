import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "bizup_enterprise.db"


def get_conn():
    return sqlite3.connect(DB_FILE)


def fetch_all_identities():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT person_id, person_type, display_name, status
        FROM identities
        ORDER BY person_type, person_id
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def identity_exists(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("SELECT person_type FROM identities WHERE person_id = ?", (person_id,))
    row = c.fetchone()
    conn.close()
    return row


def delete_identity(person_id: str):
    info = identity_exists(person_id)

    if not info:
        print(f"[ERROR] Identity '{person_id}' does NOT exist.")
        return False

    person_type = info[0]

    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM visible_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_events WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM alerts WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM system_events WHERE person_id = ?", (person_id,))

    if person_type == "employee":
        c.execute("DELETE FROM employees WHERE person_id = ?", (person_id,))
    elif person_type == "visitor":
        c.execute("DELETE FROM visitors WHERE person_id = ?", (person_id,))

    c.execute("DELETE FROM identities WHERE person_id = ?", (person_id,))

    conn.commit()
    conn.close()

    print(f"[OK] Identity '{person_id}' ({person_type}) deleted.")
    return True


def delete_all_identities():
    rows = fetch_all_identities()

    if not rows:
        print("No identities found.")
        return

    for person_id, _, _, _ in rows:
        delete_identity(person_id)

    print(f"\n[OK] Deleted {len(rows)} identities.")


def main():
    print("\n===== IDENTITY CLEANER =====\n")
    print(f"Using DB: {DB_FILE}\n")

    rows = fetch_all_identities()

    if not rows:
        print("No identities in database.")
        return

    print("Current identities:")
    for r in rows:
        print(f" - {r[0]} | type={r[1]} | name={r[2]} | status={r[3]}")

    print("\nOptions:")
    print("1 → Delete ONE identity")
    print("2 → Delete ALL identities")

    choice = input("\nChoose: ").strip()

    if choice == "1":
        pid = input("Enter person_id: ").strip()

        confirm = input(f"Type DELETE to remove {pid}: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        delete_identity(pid)

    elif choice == "2":
        confirm = input("Type DELETE to remove EVERYTHING: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        delete_all_identities()

    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()