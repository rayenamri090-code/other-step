import sqlite3
from datetime import datetime
from config import DB_FILE, CAMERA_ID, DEFAULT_ZONE_NAME, DEFAULT_ZONE_TYPE, DEFAULT_IS_ACCESS_POINT


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def get_conn():
    return sqlite3.connect(DB_FILE)


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
        created_at TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS employees (
        person_id TEXT PRIMARY KEY,
        employee_code TEXT,
        department TEXT,
        role_name TEXT,
        schedule_id TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS visitors (
        person_id TEXT PRIMARY KEY,
        host_person_id TEXT,
        visit_reason TEXT,
        valid_from TEXT,
        valid_to TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS camera_zones (
        camera_id TEXT PRIMARY KEY,
        zone_name TEXT NOT NULL,
        zone_type TEXT NOT NULL,
        is_access_point INTEGER NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_policies (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        subject_type TEXT NOT NULL,   -- person / role / visitor
        subject_value TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        allowed_days TEXT NOT NULL,   -- Mon,Tue,Wed,...
        allowed_start TEXT NOT NULL,  -- HH:MM
        allowed_end TEXT NOT NULL     -- HH:MM
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS visible_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        track_id TEXT NOT NULL,
        start_time TEXT NOT NULL,
        end_time TEXT NOT NULL,
        duration_seconds REAL NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT NOT NULL,
        zone_name TEXT NOT NULL,
        entry_time TEXT NOT NULL,
        exit_time TEXT,
        duration_seconds REAL,
        status TEXT NOT NULL
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT NOT NULL,
        camera_id TEXT NOT NULL,
        track_id TEXT,
        person_id TEXT,
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
        track_id TEXT,
        person_id TEXT,
        alert_type TEXT NOT NULL,
        status TEXT NOT NULL,
        notes TEXT
    )
    """)

    c.execute("""
    INSERT OR IGNORE INTO camera_zones (camera_id, zone_name, zone_type, is_access_point)
    VALUES (?, ?, ?, ?)
    """, (CAMERA_ID, DEFAULT_ZONE_NAME, DEFAULT_ZONE_TYPE, DEFAULT_IS_ACCESS_POINT))

    conn.commit()
    conn.close()


def get_camera_zone(camera_id: str):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    SELECT zone_name, zone_type, is_access_point
    FROM camera_zones
    WHERE camera_id = ?
    """, (camera_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        return {
            "zone_name": DEFAULT_ZONE_NAME,
            "zone_type": DEFAULT_ZONE_TYPE,
            "is_access_point": DEFAULT_IS_ACCESS_POINT
        }

    return {
        "zone_name": row[0],
        "zone_type": row[1],
        "is_access_point": int(row[2])
    }


def ensure_identity(person_id: str, person_type: str, display_name=None, status="active"):
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
        SET last_seen = ?
        WHERE person_id = ?
        """, (now, person_id))

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
    return rows


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
    for (pid,) in rows:
        try:
            nums.append(int(pid.split("_")[1]))
        except Exception:
            pass

    n = (max(nums) + 1) if nums else 1
    return f"{prefix}{n:03d}"


def create_unknown_identity() -> str:
    person_id = next_person_id("unknown_", "unknown")
    ensure_identity(person_id, "unknown", display_name=person_id, status="pending_validation")
    return person_id


def add_alert(camera_id: str, track_id: str | None, person_id: str | None, alert_type: str, notes: str = "", status: str = "open"):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO alerts (timestamp, camera_id, track_id, person_id, alert_type, status, notes)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_str(), camera_id, track_id, person_id, alert_type, status, notes))
    conn.commit()
    conn.close()


def add_access_event(camera_id: str, track_id: str | None, person_id: str | None, action: str, confidence: float | None = None, extra: str = ""):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO access_events (timestamp, camera_id, track_id, person_id, action, confidence, extra)
    VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (now_str(), camera_id, track_id, person_id, action, confidence, extra))
    conn.commit()
    conn.close()


def open_access_session(person_id: str, zone_name: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT id FROM access_sessions
    WHERE person_id = ? AND zone_name = ? AND status = 'active'
    """, (person_id, zone_name))
    row = c.fetchone()

    if row is None:
        c.execute("""
        INSERT INTO access_sessions (person_id, zone_name, entry_time, exit_time, duration_seconds, status)
        VALUES (?, ?, ?, NULL, NULL, 'active')
        """, (person_id, zone_name, now_str()))

    conn.commit()
    conn.close()


def close_access_session(person_id: str, zone_name: str):
    conn = get_conn()
    c = conn.cursor()

    c.execute("""
    SELECT id, entry_time FROM access_sessions
    WHERE person_id = ? AND zone_name = ? AND status = 'active'
    ORDER BY id DESC LIMIT 1
    """, (person_id, zone_name))
    row = c.fetchone()

    if row:
        session_id, entry_time = row
        fmt = "%Y-%m-%d %H:%M:%S"
        duration = (datetime.strptime(now_str(), fmt) - datetime.strptime(entry_time, fmt)).total_seconds()

        c.execute("""
        UPDATE access_sessions
        SET exit_time = ?, duration_seconds = ?, status = 'closed'
        WHERE id = ?
        """, (now_str(), duration, session_id))

    conn.commit()
    conn.close()


def add_visible_session(person_id: str, camera_id: str, track_id: str, start_time: str, end_time: str, duration_seconds: float):
    conn = get_conn()
    c = conn.cursor()
    c.execute("""
    INSERT INTO visible_sessions (person_id, camera_id, track_id, start_time, end_time, duration_seconds)
    VALUES (?, ?, ?, ?, ?, ?)
    """, (person_id, camera_id, track_id, start_time, end_time, duration_seconds))
    conn.commit()
    conn.close()


def create_demo_seed():
    conn = get_conn()
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM employees")
    count = c.fetchone()[0]
    if count == 0:
        ensure_identity("emp_001", "employee", "Employee 001", "active")
        c.execute("""
        INSERT OR IGNORE INTO employees (person_id, employee_code, department, role_name, schedule_id)
        VALUES ('emp_001', 'E001', 'IT', 'engineer', 'sched_office')
        """)

        c.execute("""
        INSERT INTO access_policies (subject_type, subject_value, zone_name, allowed_days, allowed_start, allowed_end)
        VALUES ('role', 'engineer', ?, 'Mon,Tue,Wed,Thu,Fri,Sat,Sun', '00:00', '23:59')
        """, (DEFAULT_ZONE_NAME,))

    conn.commit()
    conn.close()