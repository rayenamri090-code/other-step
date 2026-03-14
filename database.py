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
# Schema / Migration Helpers
# =========================================================

def _column_exists(conn, table_name: str, column_name: str) -> bool:
    c = conn.cursor()
    c.execute(f"PRAGMA table_info({table_name})")
    rows = c.fetchall()
    return any(row["name"] == column_name for row in rows)


def _ensure_column(conn, table_name: str, column_name: str, column_sql: str):
    if not _column_exists(conn, table_name, column_name):
        c = conn.cursor()
        c.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_sql}")
        conn.commit()


def _ensure_indexes(conn):
    c = conn.cursor()

    c.execute("CREATE INDEX IF NOT EXISTS idx_face_embeddings_person_id ON face_embeddings(person_id)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_identities_person_type_status ON identities(person_type, status)")
    c.execute("CREATE INDEX IF NOT EXISTS idx_identities_last_seen ON identities(last_seen)")

    c.execute("CREATE INDEX IF NOT EXISTS idx_camera_zones_camera_active ON camera_zones(camera_id, is_active)")

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_access_policies_lookup
    ON access_policies(subject_type, subject_value, zone_id, is_active)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_visible_sessions_person_date
    ON visible_sessions(person_id, start_time)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_access_sessions_person_zone_status
    ON access_sessions(person_id, zone_id, status)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_access_events_person_time
    ON access_events(person_id, timestamp)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_alerts_person_time
    ON alerts(person_id, timestamp)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_system_events_person_time
    ON system_events(person_id, timestamp)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_work_schedules_active
    ON work_schedules(is_active)
    """)

    c.execute("""
    CREATE INDEX IF NOT EXISTS idx_identities_attributes_locked
    ON identities(attributes_locked)
    """)

    conn.commit()


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
    CREATE TABLE IF NOT EXISTS work_schedules (
        schedule_id TEXT PRIMARY KEY,
        schedule_name TEXT NOT NULL,
        allowed_days TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        is_active INTEGER NOT NULL DEFAULT 1,
        created_at TEXT NOT NULL
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

    # -----------------------------------------------------
    # Migration-safe extra columns for future Odoo workflow
    # -----------------------------------------------------
    _ensure_column(conn, "identities", "is_active", "INTEGER NOT NULL DEFAULT 1")
    _ensure_column(conn, "identities", "merged_into_person_id", "TEXT")
    _ensure_column(conn, "identities", "resolved_at", "TEXT")
    _ensure_column(conn, "identities", "resolution_note", "TEXT")

    # -----------------------------------------------------
    # Attribute locking columns
    # -----------------------------------------------------
    _ensure_column(conn, "identities", "predicted_gender", "TEXT")
    _ensure_column(conn, "identities", "predicted_age_range", "TEXT")
    _ensure_column(conn, "identities", "attributes_locked", "INTEGER NOT NULL DEFAULT 0")
    _ensure_column(conn, "identities", "attributes_updated_at", "TEXT")

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

    _ensure_indexes(conn)

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

def ensure_identity(
    person_id: str,
    person_type: str,
    display_name: str | None = None,
    status: str = "active",
    is_active: int = 1,
):
    conn = get_conn()
    c = conn.cursor()
    now = now_str()

    c.execute("SELECT person_id FROM identities WHERE person_id = ?", (person_id,))
    row = c.fetchone()

    if row is None:
        c.execute("""
        INSERT INTO identities (
            person_id, person_type, display_name, status, created_at, last_seen,
            is_active, merged_into_person_id, resolved_at, resolution_note,
            predicted_gender, predicted_age_range, attributes_locked, attributes_updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0, NULL)
        """, (person_id, person_type, display_name, status, now, now, is_active))
    else:
        c.execute("""
        UPDATE identities
        SET person_type = ?,
            display_name = COALESCE(?, display_name),
            status = ?,
            last_seen = ?,
            is_active = ?
        WHERE person_id = ?
        """, (person_type, display_name, status, now, is_active, person_id))

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
    SELECT person_id, person_type, display_name, status,
           COALESCE(is_active, 1) AS is_active,
           merged_into_person_id, resolved_at, resolution_note,
           predicted_gender, predicted_age_range,
           COALESCE(attributes_locked, 0) AS attributes_locked,
           attributes_updated_at
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
        "is_active": int(row["is_active"]),
        "merged_into_person_id": row["merged_into_person_id"],
        "resolved_at": row["resolved_at"],
        "resolution_note": row["resolution_note"],
        "predicted_gender": row["predicted_gender"],
        "predicted_age_range": row["predicted_age_range"],
        "attributes_locked": int(row["attributes_locked"]),
        "attributes_updated_at": row["attributes_updated_at"],
    }


def get_identity_attributes(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT predicted_gender, predicted_age_range,
           COALESCE(attributes_locked, 0) AS attributes_locked,
           attributes_updated_at
    FROM identities
    WHERE person_id = ?
    """, (person_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "predicted_gender": row["predicted_gender"],
        "predicted_age_range": row["predicted_age_range"],
        "attributes_locked": int(row["attributes_locked"]),
        "attributes_updated_at": row["attributes_updated_at"],
    }


def save_identity_attributes(
    person_id: str,
    predicted_gender: str | None = None,
    predicted_age_range: str | None = None,
    lock_attributes: int = 1,
):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE identities
    SET predicted_gender = COALESCE(?, predicted_gender),
        predicted_age_range = COALESCE(?, predicted_age_range),
        attributes_locked = ?,
        attributes_updated_at = ?
    WHERE person_id = ?
    """, (
        predicted_gender,
        predicted_age_range,
        int(lock_attributes),
        now_str(),
        person_id,
    ))
    conn.commit()
    conn.close()


def clear_identity_attributes(person_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE identities
    SET predicted_gender = NULL,
        predicted_age_range = NULL,
        attributes_locked = 0,
        attributes_updated_at = ?
    WHERE person_id = ?
    """, (now_str(), person_id))
    conn.commit()
    conn.close()


def are_attributes_locked(person_id: str) -> bool:
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT COALESCE(attributes_locked, 0) AS attributes_locked
    FROM identities
    WHERE person_id = ?
    """, (person_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return False

    return int(row["attributes_locked"]) == 1


def get_all_identities():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT person_id, person_type, display_name, status,
           COALESCE(is_active, 1) AS is_active,
           merged_into_person_id, resolved_at, resolution_note,
           predicted_gender, predicted_age_range,
           COALESCE(attributes_locked, 0) AS attributes_locked,
           attributes_updated_at
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
            "is_active": int(row["is_active"]),
            "merged_into_person_id": row["merged_into_person_id"],
            "resolved_at": row["resolved_at"],
            "resolution_note": row["resolution_note"],
            "predicted_gender": row["predicted_gender"],
            "predicted_age_range": row["predicted_age_range"],
            "attributes_locked": int(row["attributes_locked"]),
            "attributes_updated_at": row["attributes_updated_at"],
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
        is_active=1,
    )
    return person_id


def create_employee_identity(
    display_name: str,
    employee_code: str | None = None,
    department: str | None = None,
    role_name: str | None = None,
    schedule_id: str | None = None,
    person_id: str | None = None,
) -> str:
    if not person_id:
        person_id = next_person_id("emp_", "employee")

    ensure_identity(
        person_id=person_id,
        person_type="employee",
        display_name=display_name,
        status="active",
        is_active=1,
    )

    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO employees (
        person_id, employee_code, department, role_name, schedule_id
    )
    VALUES (?, ?, ?, ?, ?)
    """, (person_id, employee_code, department, role_name, schedule_id))
    conn.commit()
    conn.close()

    return person_id


def create_visitor_identity(
    display_name: str,
    host_person_id: str | None = None,
    visit_reason: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    person_id: str | None = None,
) -> str:
    if not person_id:
        person_id = next_person_id("visitor_", "visitor")

    ensure_identity(
        person_id=person_id,
        person_type="visitor",
        display_name=display_name,
        status="active",
        is_active=1,
    )

    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT OR REPLACE INTO visitors (
        person_id, host_person_id, visit_reason, valid_from, valid_to
    )
    VALUES (?, ?, ?, ?, ?)
    """, (person_id, host_person_id, visit_reason, valid_from, valid_to))
    conn.commit()
    conn.close()

    return person_id


def set_identity_blocked(person_id: str, note: str = ""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE identities
    SET status = 'blocked',
        is_active = 0,
        resolved_at = ?,
        resolution_note = CASE
            WHEN ? = '' THEN resolution_note
            ELSE ?
        END
    WHERE person_id = ?
    """, (now_str(), note, note, person_id))
    conn.commit()
    conn.close()


def set_identity_merged(source_person_id: str, target_person_id: str, note: str = ""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    UPDATE identities
    SET status = 'merged',
        is_active = 0,
        merged_into_person_id = ?,
        resolved_at = ?,
        resolution_note = ?
    WHERE person_id = ?
    """, (target_person_id, now_str(), note, source_person_id))
    conn.commit()
    conn.close()


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


def copy_embeddings_to_identity(source_person_id: str, target_person_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT embedding_json, quality_score
    FROM face_embeddings
    WHERE person_id = ?
    ORDER BY id ASC
    """, (source_person_id,))
    rows = c.fetchall()

    for row in rows:
        c.execute("""
        INSERT INTO face_embeddings (person_id, embedding_json, quality_score, created_at)
        VALUES (?, ?, ?, ?)
        """, (
            target_person_id,
            row["embedding_json"],
            row["quality_score"],
            now_str(),
        ))

    conn.commit()
    conn.close()


def get_all_identities_with_embeddings():
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT i.person_id, i.person_type, i.status, fe.embedding_json
    FROM identities i
    JOIN face_embeddings fe ON fe.person_id = i.person_id
    WHERE COALESCE(i.is_active, 1) = 1
      AND i.status NOT IN ('blocked', 'merged')
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
# Work Schedules
# =========================================================

def create_work_schedule(
    schedule_id: str,
    schedule_name: str,
    allowed_days: str,
    start_time: str,
    end_time: str,
    is_active: int = 1,
):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO work_schedules (
        schedule_id, schedule_name, allowed_days, start_time, end_time, is_active, created_at
    )
    VALUES (
        ?, ?, ?, ?, ?, ?,
        COALESCE((SELECT created_at FROM work_schedules WHERE schedule_id = ?), ?)
    )
    """, (
        schedule_id,
        schedule_name,
        allowed_days,
        start_time,
        end_time,
        is_active,
        schedule_id,
        now_str(),
    ))

    conn.commit()
    conn.close()


def get_work_schedule(schedule_id: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT schedule_id, schedule_name, allowed_days, start_time, end_time, is_active, created_at
    FROM work_schedules
    WHERE schedule_id = ?
    """, (schedule_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "schedule_id": row["schedule_id"],
        "schedule_name": row["schedule_name"],
        "allowed_days": row["allowed_days"],
        "start_time": row["start_time"],
        "end_time": row["end_time"],
        "is_active": int(row["is_active"]),
        "created_at": row["created_at"],
    }


def get_all_work_schedules(active_only: bool = False):
    conn = get_conn()
    c = conn.cursor()

    if active_only:
        c.execute("""
        SELECT schedule_id, schedule_name, allowed_days, start_time, end_time, is_active, created_at
        FROM work_schedules
        WHERE is_active = 1
        ORDER BY schedule_name ASC, schedule_id ASC
        """)
    else:
        c.execute("""
        SELECT schedule_id, schedule_name, allowed_days, start_time, end_time, is_active, created_at
        FROM work_schedules
        ORDER BY schedule_name ASC, schedule_id ASC
        """)

    rows = c.fetchall()
    conn.close()

    return [
        {
            "schedule_id": row["schedule_id"],
            "schedule_name": row["schedule_name"],
            "allowed_days": row["allowed_days"],
            "start_time": row["start_time"],
            "end_time": row["end_time"],
            "is_active": int(row["is_active"]),
            "created_at": row["created_at"],
        }
        for row in rows
    ]


def delete_work_schedule(schedule_id: str) -> bool:
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT 1 FROM work_schedules WHERE schedule_id = ?", (schedule_id,))
    row = c.fetchone()
    if row is None:
        conn.close()
        return False

    c.execute("DELETE FROM work_schedules WHERE schedule_id = ?", (schedule_id,))
    conn.commit()
    conn.close()
    return True


# =========================================================
# Resolution / Merge Helpers
# =========================================================

def reassign_history_to_identity(source_person_id: str, target_person_id: str, target_person_type: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    UPDATE visible_sessions
    SET person_id = ?, person_type = ?
    WHERE person_id = ?
    """, (target_person_id, target_person_type, source_person_id))

    c.execute("""
    UPDATE access_sessions
    SET person_id = ?, person_type = ?
    WHERE person_id = ?
    """, (target_person_id, target_person_type, source_person_id))

    c.execute("""
    UPDATE access_events
    SET person_id = ?, person_type = ?
    WHERE person_id = ?
    """, (target_person_id, target_person_type, source_person_id))

    c.execute("""
    UPDATE alerts
    SET person_id = ?, person_type = ?
    WHERE person_id = ?
    """, (target_person_id, target_person_type, source_person_id))

    c.execute("""
    UPDATE system_events
    SET person_id = ?, person_type = ?
    WHERE person_id = ?
    """, (target_person_id, target_person_type, source_person_id))

    conn.commit()
    conn.close()


def resolve_unknown_to_existing_identity(
    unknown_person_id: str,
    target_person_id: str,
    note: str = "",
    copy_embeddings: bool = True,
    reassign_history: bool = True,
):
    unknown_info = get_identity_info(unknown_person_id)
    target_info = get_identity_info(target_person_id)

    if unknown_info is None:
        raise ValueError(f"Unknown source identity not found: {unknown_person_id}")

    if target_info is None:
        raise ValueError(f"Target identity not found: {target_person_id}")

    if unknown_info["person_type"] != "unknown":
        raise ValueError(f"Source identity is not an unknown: {unknown_person_id}")

    if unknown_person_id == target_person_id:
        raise ValueError("Source and target identities cannot be the same")

    if copy_embeddings:
        copy_embeddings_to_identity(unknown_person_id, target_person_id)

    if reassign_history:
        reassign_history_to_identity(
            source_person_id=unknown_person_id,
            target_person_id=target_person_id,
            target_person_type=target_info["person_type"],
        )

    set_identity_merged(
        source_person_id=unknown_person_id,
        target_person_id=target_person_id,
        note=note or f"Resolved into {target_person_id}",
    )

    return {
        "source_person_id": unknown_person_id,
        "target_person_id": target_person_id,
        "target_person_type": target_info["person_type"],
        "copied_embeddings": copy_embeddings,
        "reassigned_history": reassign_history,
        "status": "merged",
    }


def resolve_unknown_to_new_employee(
    unknown_person_id: str,
    display_name: str,
    employee_code: str | None = None,
    department: str | None = None,
    role_name: str | None = None,
    schedule_id: str | None = None,
    note: str = "",
    reassign_history: bool = True,
):
    target_person_id = create_employee_identity(
        display_name=display_name,
        employee_code=employee_code,
        department=department,
        role_name=role_name,
        schedule_id=schedule_id,
    )

    result = resolve_unknown_to_existing_identity(
        unknown_person_id=unknown_person_id,
        target_person_id=target_person_id,
        note=note or f"Resolved unknown as new employee {target_person_id}",
        copy_embeddings=True,
        reassign_history=reassign_history,
    )

    return result


def resolve_unknown_to_new_visitor(
    unknown_person_id: str,
    display_name: str,
    host_person_id: str | None = None,
    visit_reason: str | None = None,
    valid_from: str | None = None,
    valid_to: str | None = None,
    note: str = "",
    reassign_history: bool = True,
):
    target_person_id = create_visitor_identity(
        display_name=display_name,
        host_person_id=host_person_id,
        visit_reason=visit_reason,
        valid_from=valid_from,
        valid_to=valid_to,
    )

    result = resolve_unknown_to_existing_identity(
        unknown_person_id=unknown_person_id,
        target_person_id=target_person_id,
        note=note or f"Resolved unknown as new visitor {target_person_id}",
        copy_embeddings=True,
        reassign_history=reassign_history,
    )

    return result


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
        exit_time_str = now_str()
        duration = (datetime.strptime(exit_time_str, fmt) - datetime.strptime(entry_time, fmt)).total_seconds()

        c.execute("""
        UPDATE access_sessions
        SET exit_time = ?, duration_seconds = ?, status = 'closed'
        WHERE id = ?
        """, (exit_time_str, duration, session_id))

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
    now = now_str()

    c.execute("""
    INSERT OR IGNORE INTO work_schedules (
        schedule_id, schedule_name, allowed_days, start_time, end_time, is_active, created_at
    )
    VALUES (?, ?, ?, ?, ?, 1, ?)
    """, (
        "sched_office",
        "Office Schedule",
        "Mon,Tue,Wed,Thu,Fri",
        "08:00",
        "18:00",
        now,
    ))

    c.execute("SELECT COUNT(*) AS cnt FROM employees")
    employee_count = c.fetchone()["cnt"]

    if employee_count == 0:
        c.execute("""
        INSERT OR IGNORE INTO identities (
            person_id, person_type, display_name, status, created_at, last_seen,
            is_active, merged_into_person_id, resolved_at, resolution_note,
            predicted_gender, predicted_age_range, attributes_locked, attributes_updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, 0, NULL)
        """, (
            "emp_001",
            "employee",
            "Employee 001",
            "active",
            now,
            now,
            1,
        ))

        c.execute("""
        INSERT OR IGNORE INTO employees (
            person_id, employee_code, department, role_name, schedule_id
        )
        VALUES ('emp_001', 'E001', 'IT', 'engineer', 'sched_office')
        """)

    c.execute("""
    SELECT 1
    FROM access_policies
    WHERE subject_type = 'role'
      AND subject_value = 'engineer'
      AND zone_id = ?
      AND allowed_days = 'Mon,Tue,Wed,Thu,Fri,Sat,Sun'
      AND allowed_start = '00:00'
      AND allowed_end = '23:59'
      AND is_active = 1
    LIMIT 1
    """, (DEFAULT_ZONE_ID,))
    existing_policy = c.fetchone()

    if existing_policy is None:
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

    def sort_bucket(bucket: dict):
        return sorted(
            bucket.values(),
            key=lambda x: (x.get("total_visible_seconds", 0.0), x["person_id"]),
            reverse=True,
        )

    return {
        "date": date_str,
        "employee": sort_bucket(grouped["employee"]),
        "visitor": sort_bucket(grouped["visitor"]),
        "unknown": sort_bucket(grouped["unknown"]),
    }


if __name__ == "__main__":
    init_db()
    create_demo_seed()
    print("database initialized")