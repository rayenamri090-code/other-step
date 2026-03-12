import json
import sqlite3
from datetime import datetime
from typing import Any

from config import (
    DB_FILE,
    CAMERA_ID,
    DEFAULT_ZONE_ID,
    DEFAULT_ZONE_NAME,
    DEFAULT_ZONE_TYPE,
    DEFAULT_IS_ACCESS_POINT,
)


# =========================================================
# Time Helpers
# =========================================================

def now_dt() -> datetime:
    return datetime.now()


def now_str() -> str:
    return now_dt().strftime("%Y-%m-%d %H:%M:%S")


# =========================================================
# Connection Helpers
# =========================================================

def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


# =========================================================
# Database Initialization
# =========================================================

def init_db():
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS identities (
        person_id TEXT PRIMARY KEY,
        person_type TEXT NOT NULL,
        display_name TEXT,
        status TEXT NOT NULL,
        created_at TEXT NOT NULL,
        last_seen TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS face_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        embedding_json TEXT NOT NULL,
        quality_score REAL DEFAULT 1.0,
        created_at TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES identities(person_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        person_id TEXT PRIMARY KEY,
        employee_code TEXT,
        department TEXT,
        role_name TEXT,
        schedule_id TEXT,
        FOREIGN KEY (person_id) REFERENCES identities(person_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS visitors (
        person_id TEXT PRIMARY KEY,
        host_person_id TEXT,
        visit_reason TEXT,
        valid_from TEXT,
        valid_to TEXT,
        FOREIGN KEY (person_id) REFERENCES identities(person_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS camera_zones (
        zone_id TEXT PRIMARY KEY,
        camera_id TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        is_access_point INTEGER NOT NULL DEFAULT 0,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_policies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_type TEXT NOT NULL,
        subject_value TEXT NOT NULL,
        zone_id TEXT NOT NULL,
        allowed_days TEXT NOT NULL,
        allowed_start TEXT NOT NULL,
        allowed_end TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS visible_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        person_type TEXT,
        camera_id TEXT NOT NULL,
        zone_id TEXT,
        track_id TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        duration_seconds REAL NOT NULL,
        appearance_count INTEGER NOT NULL DEFAULT 1,
        FOREIGN KEY (person_id) REFERENCES identities(person_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        person_type TEXT,
        zone_id TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        duration_seconds REAL,
        status TEXT NOT NULL,
        FOREIGN KEY (person_id) REFERENCES identities(person_id)
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        zone_id TEXT,
        track_id TEXT,
        person_id TEXT,
        person_type TEXT,
        action TEXT NOT NULL,
        confidence REAL,
        extra TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        zone_id TEXT,
        track_id TEXT,
        person_id TEXT,
        person_type TEXT,
        alert_type TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS system_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        event_type TEXT NOT NULL,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        zone_id TEXT,
        track_id TEXT,
        person_id TEXT,
        person_type TEXT,
        confidence REAL,
        payload_json TEXT
    )
    """)

    c.execute("""
    INSERT OR IGNORE INTO camera_zones (
        zone_id, camera_id, zone_name, zone_type, is_access_point, is_active, created_at
    )
    VALUES (?, ?, ?, ?, ?, 1, ?)
    """, (
        DEFAULT_ZONE_ID,
        CAMERA_ID,
        DEFAULT_ZONE_NAME,
        DEFAULT_ZONE_TYPE,
        DEFAULT_IS_ACCESS_POINT,
        now_str(),
    ))

    conn.commit()
    conn.close()


# =========================================================
# Zone Helpers
# =========================================================

def get_camera_zone(camera_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT zone_id, zone_name, zone_type, is_access_point
    FROM camera_zones
    WHERE camera_id = ? AND is_active = 1
    ORDER BY created_at ASC
    LIMIT 1
    """, (camera_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "zone_id": DEFAULT_ZONE_ID,
            "zone_name": DEFAULT_ZONE_NAME,
            "zone_type": DEFAULT_ZONE_TYPE,
            "is_access_point": DEFAULT_IS_ACCESS_POINT,
        }

    return {
        "zone_id": row["zone_id"],
        "zone_name": row["zone_name"],
        "zone_type": row["zone_type"],
        "is_access_point": int(row["is_access_point"]),
    }


# =========================================================
# Identity Helpers
# =========================================================

def ensure_identity(person_id: str, person_type: str, display_name: str | None = None, status: str = "active"):
    conn = get_conn()
    c = conn.cursor()
    now = now_str()

    c.execute("SELECT person_id FROM identities WHERE person_id = ?", (person_id,))
    row = c.fetchone()

    if row is None:
        c.execute("""
        INSERT INTO identities (person_id, person_type, display_name, status, created_at, last_seen)
        VALUES (?, ?, ?, ?, ?, ?)
        """, (person_id, person_type, display_name, status, now, now))
    else:
        c.execute("""
        UPDATE identities
        SET person_type = ?, display_name = COALESCE(?, display_name), status = ?, last_seen = ?
        WHERE person_id = ?
        """, (person_type, display_name, status, now, person_id))

    conn.commit()
    conn.close()


def update_last_seen(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE identities
    SET last_seen = ?
    WHERE person_id = ?
    """, (now_str(), person_id))
    conn.commit()
    conn.close()


def get_identity_info(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT person_id, person_type, display_name, status
    FROM identities
    WHERE person_id = ?
    """, (person_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "person_id": row["person_id"],
        "person_type": row["person_type"],
        "display_name": row["display_name"],
        "status": row["status"],
    }


def get_all_identities():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT person_id, person_type, display_name, status
    FROM identities
    ORDER BY person_type ASC, person_id ASC
    """)
    rows = c.fetchall()
    conn.close()

    return [
        {
            "person_id": row["person_id"],
            "person_type": row["person_type"],
            "display_name": row["display_name"],
            "status": row["status"],
        }
        for row in rows
    ]


def next_person_id(prefix: str, person_type: str) -> str:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT person_id FROM identities
    WHERE person_type = ? AND person_id LIKE ?
    """, (person_type, f"{prefix}%"))
    rows = c.fetchall()
    conn.close()

    nums = []
    for row in rows:
        pid = row["person_id"]
        try:
            nums.append(int(pid.split("_")[1]))
        except Exception:
            pass

    n = (max(nums) + 1) if nums else 1
    return f"{prefix}{n:03d}"


def create_unknown_identity() -> str:
    person_id = next_person_id("unknown_", "unknown")
    ensure_identity(
        person_id=person_id,
        person_type="unknown",
        display_name=person_id,
        status="pending_validation",
    )
    return person_id


# =========================================================
# Embeddings
# =========================================================

def add_embedding(person_id: str, embedding_json: str, quality_score: float = 1.0):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO face_embeddings (person_id, embedding_json, quality_score, created_at)
    VALUES (?, ?, ?, ?)
    """, (person_id, embedding_json, quality_score, now_str()))
    conn.commit()
    conn.close()


def get_all_identities_with_embeddings():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT i.person_id, i.person_type, i.status, fe.embedding_json
    FROM identities i
    JOIN face_embeddings fe ON fe.person_id = i.person_id
    """)
    rows = c.fetchall()
    conn.close()

    return [
        (
            row["person_id"],
            row["person_type"],
            row["status"],
            row["embedding_json"],
        )
        for row in rows
    ]


# =========================================================
# Alerts / Events
# =========================================================

def add_alert(
    camera_id: str,
    track_id: str | None,
    person_id: str | None,
    alert_type: str,
    notes: str = "",
    status: str = "open",
    zone_id: str | None = None,
    person_type: str | None = None,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO alerts (timestamp, camera_id, zone_id, track_id, person_id, person_type, alert_type, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str(), camera_id, zone_id, track_id, person_id, person_type, alert_type, status, notes))
    conn.commit()
    conn.close()


def add_access_event(
    camera_id: str,
    track_id: str | None,
    person_id: str | None,
    action: str,
    confidence: float | None = None,
    extra: str = "",
    zone_id: str | None = None,
    person_type: str | None = None,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO access_events (timestamp, camera_id, zone_id, track_id, person_id, person_type, action, confidence, extra)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (now_str(), camera_id, zone_id, track_id, person_id, person_type, action, confidence, extra))
    conn.commit()
    conn.close()


def add_system_event(
    event_type: str,
    camera_id: str,
    zone_id: str | None = None,
    track_id: str | None = None,
    person_id: str | None = None,
    person_type: str | None = None,
    confidence: float | None = None,
    payload: dict[str, Any] | None = None,
):
    conn = get_conn()
    c = conn.cursor()
    payload_json = json.dumps(payload or {}, ensure_ascii=False)

    c.execute("""
    INSERT INTO system_events (
        event_type, timestamp, camera_id, zone_id, track_id, person_id, person_type, confidence, payload_json
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        event_type,
        now_str(),
        camera_id,
        zone_id,
        track_id,
        person_id,
        person_type,
        confidence,
        payload_json,
    ))
    conn.commit()
    conn.close()


# =========================================================
# Sessions
# =========================================================

def open_access_session(person_id: str, person_type: str | None, zone_id: str, zone_name: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT id FROM access_sessions
    WHERE person_id = ? AND zone_id = ? AND status = 'active'
    """, (person_id, zone_id))
    row = c.fetchone()

    if row is None:
        c.execute("""
        INSERT INTO access_sessions (
            person_id, person_type, zone_id, zone_name, entry_time, exit_time, duration_seconds, status
        )
        VALUES (?, ?, ?, ?, ?, NULL, NULL, 'active')
        """, (person_id, person_type, zone_id, zone_name, now_str()))

    conn.commit()
    conn.close()


def close_access_session(person_id: str, zone_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT id, entry_time
    FROM access_sessions
    WHERE person_id = ? AND zone_id = ? AND status = 'active'
    ORDER BY id DESC
    LIMIT 1
    """, (person_id, zone_id))
    row = c.fetchone()

    if row:
        session_id = row["id"]
        entry_time = row["entry_time"]

        fmt = "%Y-%m-%d %H:%M:%S"
        duration = (datetime.strptime(now_str(), fmt) - datetime.strptime(entry_time, fmt)).total_seconds()

        c.execute("""
        UPDATE access_sessions
        SET exit_time = ?, duration_seconds = ?, status = 'closed'
        WHERE id = ?
        """, (now_str(), duration, session_id))

    conn.commit()
    conn.close()


def add_visible_session(
    person_id: str,
    person_type: str | None,
    camera_id: str,
    zone_id: str | None,
    track_id: str,
    start_time: str,
    end_time: str,
    duration_seconds: float,
    appearance_count: int = 1,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO visible_sessions (
        person_id, person_type, camera_id, zone_id, track_id, start_time, end_time, duration_seconds, appearance_count
    )
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        person_id,
        person_type,
        camera_id,
        zone_id,
        track_id,
        start_time,
        end_time,
        duration_seconds,
        appearance_count,
    ))
    conn.commit()
    conn.close()


# =========================================================
# Demo Seed
# =========================================================

def create_demo_seed():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) AS cnt FROM employees")
    count = c.fetchone()["cnt"]

    if count == 0:
        ensure_identity("emp_001", "employee", "Employee 001", "active")

        c.execute("""
        INSERT OR IGNORE INTO employees (person_id, employee_code, department, role_name, schedule_id)
        VALUES ('emp_001', 'E001', 'IT', 'engineer', 'sched_office')
        """)

        c.execute("""
        INSERT INTO access_policies (
            subject_type, subject_value, zone_id, allowed_days, allowed_start, allowed_end, is_active
        )
        VALUES ('role', 'engineer', ?, 'Mon,Tue,Wed,Thu,Fri,Sat,Sun', '00:00', '23:59', 1)
        """, (DEFAULT_ZONE_ID,))

    conn.commit()
    conn.close()


# =========================================================
# Analytics / Reporting
# =========================================================

def get_person_appearance_count(person_id: str, date_str: str | None = None) -> int:
    conn = get_conn()
    c = conn.cursor()

    if date_str:
        c.execute("""
        SELECT COUNT(*)
        FROM visible_sessions
        WHERE person_id = ?
          AND date(start_time) = ?
        """, (person_id, date_str))
    else:
        c.execute("""
        SELECT COUNT(*)
        FROM visible_sessions
        WHERE person_id = ?
        """, (person_id,))

    count = c.fetchone()[0]
    conn.close()
    return int(count)


def get_all_appearance_counts(date_str: str | None = None):
    conn = get_conn()
    c = conn.cursor()

    if date_str:
        c.execute("""
        SELECT i.person_id, i.person_type, COUNT(vs.id) AS appearances
        FROM identities i
        JOIN visible_sessions vs ON vs.person_id = i.person_id
        WHERE date(vs.start_time) = ?
        GROUP BY i.person_id, i.person_type
        ORDER BY i.person_type ASC, appearances DESC, i.person_id ASC
        """, (date_str,))
    else:
        c.execute("""
        SELECT i.person_id, i.person_type, COUNT(vs.id) AS appearances
        FROM identities i
        JOIN visible_sessions vs ON vs.person_id = i.person_id
        GROUP BY i.person_id, i.person_type
        ORDER BY i.person_type ASC, appearances DESC, i.person_id ASC
        """)

    rows = c.fetchall()
    conn.close()

    return [
        {
            "person_id": row["person_id"],
            "person_type": row["person_type"],
            "appearances": int(row["appearances"]),
        }
        for row in rows
    ]


def get_person_total_visible_time(person_id: str, date_str: str | None = None) -> float:
    conn = get_conn()
    c = conn.cursor()

    if date_str:
        c.execute("""
        SELECT COALESCE(SUM(duration_seconds), 0)
        FROM visible_sessions
        WHERE person_id = ?
          AND date(start_time) = ?
        """, (person_id, date_str))
    else:
        c.execute("""
        SELECT COALESCE(SUM(duration_seconds), 0)
        FROM visible_sessions
        WHERE person_id = ?
        """, (person_id,))

    total_seconds = float(c.fetchone()[0] or 0.0)
    conn.close()
    return total_seconds


def get_all_total_visible_times(date_str: str | None = None):
    conn = get_conn()
    c = conn.cursor()

    if date_str:
        c.execute("""
        SELECT i.person_id, i.person_type, COALESCE(SUM(vs.duration_seconds), 0) AS total_seconds
        FROM identities i
        JOIN visible_sessions vs ON vs.person_id = i.person_id
        WHERE date(vs.start_time) = ?
        GROUP BY i.person_id, i.person_type
        ORDER BY i.person_type ASC, total_seconds DESC, i.person_id ASC
        """, (date_str,))
    else:
        c.execute("""
        SELECT i.person_id, i.person_type, COALESCE(SUM(vs.duration_seconds), 0) AS total_seconds
        FROM identities i
        JOIN visible_sessions vs ON vs.person_id = i.person_id
        GROUP BY i.person_id, i.person_type
        ORDER BY i.person_type ASC, total_seconds DESC, i.person_id ASC
        """)

    rows = c.fetchall()
    conn.close()

    return [
        {
            "person_id": row["person_id"],
            "person_type": row["person_type"],
            "total_visible_seconds": float(row["total_seconds"] or 0.0),
        }
        for row in rows
    ]


def get_person_daily_first_last_entry(person_id: str, date_str: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT
        MIN(entry_time) AS first_entry,
        MAX(COALESCE(exit_time, entry_time)) AS last_entry
    FROM access_sessions
    WHERE person_id = ?
      AND date(entry_time) = ?
    """, (person_id, date_str))

    row = c.fetchone()
    conn.close()

    identity = get_identity_info(person_id)

    return {
        "person_id": person_id,
        "person_type": identity["person_type"] if identity else None,
        "date": date_str,
        "first_entry": row["first_entry"] if row and row["first_entry"] else None,
        "last_entry": row["last_entry"] if row and row["last_entry"] else None,
    }


def get_all_daily_first_last_entries(date_str: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT
        i.person_id,
        i.person_type,
        MIN(a.entry_time) AS first_entry,
        MAX(COALESCE(a.exit_time, a.entry_time)) AS last_entry
    FROM identities i
    JOIN access_sessions a ON a.person_id = i.person_id
    WHERE date(a.entry_time) = ?
    GROUP BY i.person_id, i.person_type
    ORDER BY i.person_type ASC, i.person_id ASC
    """, (date_str,))

    rows = c.fetchall()
    conn.close()

    return [
        {
            "person_id": row["person_id"],
            "person_type": row["person_type"],
            "date": date_str,
            "first_entry": row["first_entry"],
            "last_entry": row["last_entry"],
        }
        for row in rows
    ]


def get_person_daily_work_hours(person_id: str, date_str: str):
    identity = get_identity_info(person_id)
    if identity is None or identity["person_type"] != "employee":
        return None

    total_seconds = get_person_total_visible_time(person_id, date_str)

    return {
        "person_id": person_id,
        "person_type": "employee",
        "date": date_str,
        "total_visible_seconds": total_seconds,
        "total_visible_hours": round(total_seconds / 3600.0, 4),
    }


def get_all_daily_work_hours(date_str: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT
        i.person_id,
        i.person_type,
        COALESCE(SUM(vs.duration_seconds), 0) AS total_seconds
    FROM identities i
    JOIN visible_sessions vs ON vs.person_id = i.person_id
    WHERE i.person_type = 'employee'
      AND date(vs.start_time) = ?
    GROUP BY i.person_id, i.person_type
    ORDER BY total_seconds DESC, i.person_id ASC
    """, (date_str,))

    rows = c.fetchall()
    conn.close()

    return [
        {
            "person_id": row["person_id"],
            "person_type": row["person_type"],
            "date": date_str,
            "total_visible_seconds": float(row["total_seconds"] or 0.0),
            "total_visible_hours": round(float(row["total_seconds"] or 0.0) / 3600.0, 4),
        }
        for row in rows
    ]


def get_grouped_daily_report(date_str: str):
    appearances = get_all_appearance_counts(date_str)
    visible_times = get_all_total_visible_times(date_str)
    first_last = get_all_daily_first_last_entries(date_str)
    work_hours = get_all_daily_work_hours(date_str)

    grouped = {
        "employee": {},
        "visitor": {},
        "unknown": {},
    }

    def ensure_bucket(person_type: str, person_id: str):
        grouped.setdefault(person_type, {}).setdefault(person_id, {
            "person_id": person_id,
            "person_type": person_type,
            "appearances": 0,
            "total_visible_seconds": 0.0,
            "first_entry": None,
            "last_entry": None,
            "total_visible_hours": None,
        })

    for item in appearances:
        ensure_bucket(item["person_type"], item["person_id"])
        grouped[item["person_type"]][item["person_id"]]["appearances"] = item["appearances"]

    for item in visible_times:
        ensure_bucket(item["person_type"], item["person_id"])
        grouped[item["person_type"]][item["person_id"]]["total_visible_seconds"] = item["total_visible_seconds"]

    for item in first_last:
        ensure_bucket(item["person_type"], item["person_id"])
        grouped[item["person_type"]][item["person_id"]]["first_entry"] = item["first_entry"]
        grouped[item["person_type"]][item["person_id"]]["last_entry"] = item["last_entry"]

    for item in work_hours:
        ensure_bucket(item["person_type"], item["person_id"])
        grouped[item["person_type"]][item["person_id"]]["total_visible_hours"] = item["total_visible_hours"]

    return {
        "date": date_str,
        "employee": list(grouped["employee"].values()),
        "visitor": list(grouped["visitor"].values()),
        "unknown": list(grouped["unknown"].values()),
    }