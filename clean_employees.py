import sqlite3
from pathlib import Path

DB_FILE = Path(__file__).resolve().parent / "bizup_enterprise.db"


def get_conn():
    return sqlite3.connect(DB_FILE)


def fetch_employees():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT i.person_id, i.display_name, e.employee_code, e.department, e.role_name
        FROM identities i
        LEFT JOIN employees e ON e.person_id = i.person_id
        WHERE i.person_type = 'employee'
        ORDER BY i.person_id ASC
    """)
    rows = c.fetchall()
    conn.close()
    return rows


def employee_exists(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
        SELECT 1 FROM identities
        WHERE person_id = ? AND person_type = 'employee'
    """, (person_id,))
    row = c.fetchone()
    conn.close()
    return row is not None


def delete_employee(person_id: str):
    if not employee_exists(person_id):
        print(f"[ERROR] Employee '{person_id}' does NOT exist.")
        return False

    conn = get_conn()
    c = conn.cursor()

    c.execute("DELETE FROM face_embeddings WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM visible_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_sessions WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM access_events WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM alerts WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM system_events WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM employees WHERE person_id = ?", (person_id,))
    c.execute("DELETE FROM identities WHERE person_id = ?", (person_id,))

    conn.commit()
    conn.close()

    print(f"[OK] Employee '{person_id}' deleted successfully.")
    return True


def delete_all_employees():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
        SELECT person_id
        FROM identities
        WHERE person_type = 'employee'
    """)
    person_ids = [row[0] for row in c.fetchall()]
    conn.close()

    for person_id in person_ids:
        delete_employee(person_id)

    return person_ids


def main():
    print("\n=== Employee Cleanup ===\n")
    print(f"Using DB: {DB_FILE}\n")

    rows = fetch_employees()
    if not rows:
        print("No employees found.")
        return

    print("Current employees:")
    for row in rows:
        person_id, display_name, employee_code, department, role_name = row
        print(
            f" - {person_id} | name={display_name} | "
            f"code={employee_code} | dept={department} | role={role_name}"
        )

    print("\nOptions:")
    print("1. Delete one employee")
    print("2. Delete all employees")
    choice = input("\nChoose (1/2): ").strip()

    if choice == "1":
        person_id = input("Enter employee person_id to delete: ").strip()
        if not person_id:
            print("No person_id entered. Cancelled.")
            return

        confirm = input(f"Type DELETE to remove {person_id}: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        success = delete_employee(person_id)
        if not success:
            print("[INFO] Nothing was deleted.")

    elif choice == "2":
        confirm = input("Type DELETE to remove ALL employees: ").strip()
        if confirm != "DELETE":
            print("Cancelled.")
            return

        deleted = delete_all_employees()
        print(f"[OK] Deleted {len(deleted)} employee(s).")

    else:
        print("Invalid choice. Cancelled.")


if __name__ == "__main__":
    main()