import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "bizup_enterprise.db"


def get_conn():
    return sqlite3.connect(DB_FILE)


def fetch_visitors():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT i.person_id, i.display_name, v.host_person_id, v.visit_reason, v.valid_from, v.valid_to
        FROM identities i
        LEFT JOIN visitors v ON v.person_id = i.person_id
        WHERE i.person_type = 'visitor'
        ORDER BY i.person_id ASC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def delete_visitor(person_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM visible_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_events WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM alerts WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM system_events WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM visitors WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM identities WHERE person_id = ?", (person_id,))

    conn.commit()
    conn.close()


def delete_all_visitors():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT person_id
        FROM identities
        WHERE person_type = 'visitor'
    """)
    person_ids = [row[0] for row in c.fetchall()]
    conn.close()

    for person_id in person_ids:
        delete_visitor(person_id)

    return person_ids


def main():
    print("\n=== Visitor Cleanup ===\n")
    print(f"Using DB: {DB_FILE}\n")

    rows = fetch_visitors()
    if not rows:
        print("No visitors found.")
        return

    print("Current visitors:")
    for row in rows:
        person_id, display_name, host_person_id, visit_reason, valid_from, valid_to = row
        print(
            f" - {person_id} | name={display_name} | "
            f"host={host_person_id} | reason={visit_reason} | "
            f"from={valid_from} | to={valid_to}"
        )

    print("\nOptions:")
    print("1. Delete one visitor")
    print("2. Delete all visitors")
    choice = input("\nChoose (1/2): ").strip()

    if choice == "1":
        person_id = input("Enter visitor person_id to delete: ").strip()
        if not person_id:
            print("No person_id entered. Cancelled.")
            return

        confirm = input(f"Type DELETE to remove {person_id}: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        delete_visitor(person_id)
        print(f"Deleted visitor data for {person_id}")

    elif choice == "2":
        confirm = input("Type DELETE to remove ALL visitors: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        deleted = delete_all_visitors()
        print(f"Deleted {len(deleted)} visitor(s): {deleted}")

    else:
        print("Invalid choice. Cancelled.")


if __name__ == "__main__":
    main()