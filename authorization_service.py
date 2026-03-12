import sqlite3
from datetime import datetime

from config import DB_FILE, DEFAULT_DENY_IF_NO_RULE


class AuthorizationService:
    def __init__(self):
        self.default_deny_if_no_rule = DEFAULT_DENY_IF_NO_RULE

    def _conn(self):
        conn = sqlite3.connect(DB_FILE)
        conn.row_factory = sqlite3.Row
        return conn

    def _current_day_and_time(self):
        now = datetime.now()
        return now.strftime("%a"), now.strftime("%H:%M")

    def _employee_role(self, person_id):
        conn = self._conn()
        c = conn.cursor()
        c.execute(
            "SELECT role_name FROM employees WHERE person_id = ?",
            (person_id,),
        )
        row = c.fetchone()
        conn.close()
        return row["role_name"] if row else None

    def _visitor_valid(self, person_id):
        conn = self._conn()
        c = conn.cursor()
        c.execute("""
        SELECT valid_from, valid_to
        FROM visitors
        WHERE person_id = ?
        """, (person_id,))
        row = c.fetchone()
        conn.close()

        if not row or not row["valid_from"] or not row["valid_to"]:
            return False

        now = datetime.now()
        vf = datetime.strptime(row["valid_from"], "%Y-%m-%d %H:%M:%S")
        vt = datetime.strptime(row["valid_to"], "%Y-%m-%d %H:%M:%S")
        return vf <= now <= vt

    def _policy_match(self, subject_type, subject_value, zone_id):
        day_name, time_now = self._current_day_and_time()

        conn = self._conn()
        c = conn.cursor()
        c.execute("""
        SELECT allowed_days, allowed_start, allowed_end
        FROM access_policies
        WHERE subject_type = ?
          AND subject_value = ?
          AND zone_id = ?
          AND is_active = 1
        """, (subject_type, subject_value, zone_id))
        rows = c.fetchall()
        conn.close()

        for row in rows:
            allowed_days = row["allowed_days"]
            allowed_start = row["allowed_start"]
            allowed_end = row["allowed_end"]

            days = [d.strip() for d in allowed_days.split(",") if d.strip()]
            if day_name not in days:
                continue

            if allowed_start <= time_now <= allowed_end:
                return True

        return False

    def _fallback_if_no_policy(self, person_type):
        # For prototype stage:
        # - employees can default allow if no rule exists and config says so
        # - visitors should still require explicit policy
        if self.default_deny_if_no_rule:
            return "DENIED", "No matching access policy"

        if person_type == "employee":
            return "AUTHORIZED", "No policy found, default employee allow"

        return "DENIED", "No matching access policy"

    def decide(self, person_id, person_type, zone_name=None, zone_id=None):
        # Keep zone_name for backward compatibility in calls,
        # but authorization now uses zone_id.
        if person_id is None or person_type is None:
            return "DENIED", "No recognized identity"

        if zone_id is None:
            return "DENIED", "Missing zone_id for authorization"

        if person_type == "unknown":
            return "ALERT_PENDING", "Unknown identity pending validation"

        if person_type == "employee":
            role = self._employee_role(person_id)

            if role and self._policy_match("role", role, zone_id):
                return "AUTHORIZED", f"Employee role allowed: {role}"

            if self._policy_match("person", person_id, zone_id):
                return "AUTHORIZED", "Employee person-level policy matched"

            return self._fallback_if_no_policy("employee")

        if person_type == "visitor":
            if not self._visitor_valid(person_id):
                return "DENIED", "Visitor validity window expired"

            if self._policy_match("visitor", person_id, zone_id):
                return "AUTHORIZED", "Visitor policy matched"

            return self._fallback_if_no_policy("visitor")

        return "DENIED", "Unhandled identity type"