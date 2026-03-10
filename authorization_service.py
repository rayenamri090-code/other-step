import sqlite3
from datetime import datetime
from config import DB_FILE


class AuthorizationService:
    def __init__(self):
        pass

    def _conn(self):
        return sqlite3.connect(DB_FILE)

    def _current_day_and_time(self):
        now = datetime.now()
        return now.strftime("%a"), now.strftime("%H:%M")

    def _employee_role(self, person_id):
        conn = self._conn()
        c = conn.cursor()
        c.execute("SELECT role_name FROM employees WHERE person_id = ?", (person_id,))
        row = c.fetchone()
        conn.close()
        return row[0] if row else None

    def _visitor_valid(self, person_id):
        conn = self._conn()
        c = conn.cursor()
        c.execute("""
        SELECT valid_from, valid_to FROM visitors
        WHERE person_id = ?
        """, (person_id,))
        row = c.fetchone()
        conn.close()

        if not row or not row[0] or not row[1]:
            return False

        now = datetime.now()
        vf = datetime.strptime(row[0], "%Y-%m-%d %H:%M:%S")
        vt = datetime.strptime(row[1], "%Y-%m-%d %H:%M:%S")
        return vf <= now <= vt

    def _policy_match(self, subject_type, subject_value, zone_name):
        day_name, time_now = self._current_day_and_time()

        conn = self._conn()
        c = conn.cursor()
        c.execute("""
        SELECT allowed_days, allowed_start, allowed_end
        FROM access_policies
        WHERE subject_type = ? AND subject_value = ? AND zone_name = ?
        """, (subject_type, subject_value, zone_name))
        rows = c.fetchall()
        conn.close()

        for allowed_days, allowed_start, allowed_end in rows:
            days = [d.strip() for d in allowed_days.split(",")]
            if day_name not in days:
                continue
            if allowed_start <= time_now <= allowed_end:
                return True

        return False

    def decide(self, person_id, person_type, zone_name):
        if person_id is None or person_type is None:
            return "DENIED", "No recognized identity"

        if person_type == "unknown":
            return "ALERT_PENDING", "Unknown identity pending validation"

        if person_type == "employee":
            role = self._employee_role(person_id)
            if role and self._policy_match("role", role, zone_name):
                return "AUTHORIZED", f"Employee role allowed: {role}"
            if self._policy_match("person", person_id, zone_name):
                return "AUTHORIZED", "Employee person-level policy matched"
            return "DENIED", "Employee not authorized for this zone/time"

        if person_type == "visitor":
            if not self._visitor_valid(person_id):
                return "DENIED", "Visitor validity window expired"
            if self._policy_match("visitor", person_id, zone_name):
                return "AUTHORIZED", "Visitor policy matched"
            return "DENIED", "Visitor not authorized for this zone/time"

        return "DENIED", "Unhandled identity type"