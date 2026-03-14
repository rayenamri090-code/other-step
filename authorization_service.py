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

    def _visitor_validity_status(self, person_id):
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
            return False, "Visitor validity window missing"

        now = datetime.now()
        vf = datetime.strptime(row["valid_from"], "%Y-%m-%d %H:%M:%S")
        vt = datetime.strptime(row["valid_to"], "%Y-%m-%d %H:%M:%S")

        if now < vf:
            return False, "Visitor validity window not started yet"

        if now > vt:
            return False, "Visitor validity window expired"

        return True, "Visitor validity window active"

    def _time_in_window(self, time_now, start_time, end_time):
        """
        Supports:
        - normal shift: 08:00 -> 17:00
        - overnight shift: 22:00 -> 06:00
        """
        if start_time <= end_time:
            return start_time <= time_now <= end_time

        return time_now >= start_time or time_now <= end_time

    def _schedule_status(self, subject_type, subject_value):
        """
        Returns:
        {
            "has_any_schedule": bool,
            "matched_now": bool,
            "reason": str,
        }
        """
        day_name, time_now = self._current_day_and_time()

        conn = self._conn()
        c = conn.cursor()
        c.execute("""
        SELECT allowed_days, shift_start, shift_end
        FROM work_schedules
        WHERE subject_type = ?
          AND subject_value = ?
          AND is_active = 1
        """, (subject_type, subject_value))
        rows = c.fetchall()
        conn.close()

        if not rows:
            return {
                "has_any_schedule": False,
                "matched_now": False,
                "reason": f"No active work schedule for {subject_type}={subject_value}",
            }

        found_day_match = False

        for row in rows:
            allowed_days = row["allowed_days"] or ""
            shift_start = row["shift_start"]
            shift_end = row["shift_end"]

            days = [d.strip() for d in allowed_days.split(",") if d.strip()]
            day_matches = day_name in days

            if day_matches:
                found_day_match = True

            if day_matches and self._time_in_window(time_now, shift_start, shift_end):
                return {
                    "has_any_schedule": True,
                    "matched_now": True,
                    "reason": f"Work schedule matched for {subject_type}={subject_value}",
                }

        if not found_day_match:
            return {
                "has_any_schedule": True,
                "matched_now": False,
                "reason": f"Work schedule exists but current day {day_name} is not allowed",
            }

        return {
            "has_any_schedule": True,
            "matched_now": False,
            "reason": f"Work schedule exists but current time {time_now} is outside shift hours",
        }

    def _policy_status(self, subject_type, subject_value, zone_id):
        """
        Returns:
        {
            "has_any_policy": bool,
            "matched_now": bool,
            "reason": str,
        }
        """
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

        if not rows:
            return {
                "has_any_policy": False,
                "matched_now": False,
                "reason": f"No active zone access policy for {subject_type}={subject_value} in zone {zone_id}",
            }

        found_day_match = False

        for row in rows:
            allowed_days = row["allowed_days"] or ""
            allowed_start = row["allowed_start"]
            allowed_end = row["allowed_end"]

            days = [d.strip() for d in allowed_days.split(",") if d.strip()]
            day_matches = day_name in days

            if day_matches:
                found_day_match = True

            if day_matches and self._time_in_window(time_now, allowed_start, allowed_end):
                return {
                    "has_any_policy": True,
                    "matched_now": True,
                    "reason": f"Zone access policy matched for {subject_type}={subject_value}",
                }

        if not found_day_match:
            return {
                "has_any_policy": True,
                "matched_now": False,
                "reason": f"Zone access policy exists but current day {day_name} is not allowed",
            }

        return {
            "has_any_policy": True,
            "matched_now": False,
            "reason": f"Zone access policy exists but current time {time_now} is outside allowed window",
        }

    def _fallback_if_no_policy(self, person_type):
        if self.default_deny_if_no_rule:
            return "DENIED", "No matching access policy"

        if person_type == "employee":
            return "AUTHORIZED", "No zone policy found, default employee allow"

        if person_type == "visitor":
            return "AUTHORIZED", "Visitor valid and no zone policy found, default visitor allow"

        return "DENIED", "No matching access policy"

    def decide(self, person_id, person_type, zone_name=None, zone_id=None):
        if person_id is None or person_type is None:
            return "DENIED", "No recognized identity"

        if zone_id is None:
            return "DENIED", "Missing zone_id for authorization"

        if person_type == "unknown":
            return "ALERT_PENDING", "Unknown identity pending validation"

        if person_type == "employee":
            role = self._employee_role(person_id)

            # ---------------------------------------------
            # 1. WORK SCHEDULE CHECK
            # person override first, then role schedule
            # ---------------------------------------------
            person_schedule = self._schedule_status("person", person_id)
            if person_schedule["matched_now"]:
                schedule_ok = True
            elif person_schedule["has_any_schedule"]:
                return "DENIED", f"Employee outside personal work schedule: {person_schedule['reason']}"
            else:
                role_schedule = None
                if role:
                    role_schedule = self._schedule_status("role", role)
                    if role_schedule["matched_now"]:
                        schedule_ok = True
                    elif role_schedule["has_any_schedule"]:
                        return "DENIED", f"Employee outside role work schedule: {role_schedule['reason']}"
                    else:
                        schedule_ok = True
                else:
                    schedule_ok = True

            # ---------------------------------------------
            # 2. ZONE ACCESS CHECK
            # person override first, then role policy
            # ---------------------------------------------
            person_policy = self._policy_status("person", person_id, zone_id)
            if person_policy["matched_now"]:
                return "AUTHORIZED", "Employee person-level zone policy matched"

            role_policy = None
            if role:
                role_policy = self._policy_status("role", role, zone_id)
                if role_policy["matched_now"]:
                    return "AUTHORIZED", f"Employee role zone policy matched: {role}"

            if person_policy["has_any_policy"]:
                return "DENIED", f"Employee person-level zone policy not currently valid: {person_policy['reason']}"

            if role_policy and role_policy["has_any_policy"]:
                return "DENIED", f"Employee role zone policy not currently valid: {role_policy['reason']}"

            return self._fallback_if_no_policy("employee")

        if person_type == "visitor":
            valid, validity_reason = self._visitor_validity_status(person_id)
            if not valid:
                return "DENIED", validity_reason

            visitor_policy = self._policy_status("visitor", person_id, zone_id)
            if visitor_policy["matched_now"]:
                return "AUTHORIZED", "Visitor zone policy matched"

            if visitor_policy["has_any_policy"]:
                return "DENIED", f"Visitor zone policy not currently valid: {visitor_policy['reason']}"

            return self._fallback_if_no_policy("visitor")

        return "DENIED", "Unhandled identity type"