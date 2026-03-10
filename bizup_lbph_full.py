import cv2
import os
import numpy as np
from datetime import datetime
import time
import sqlite3
import json
import paho.mqtt.client as mqtt

# =========================
# CONFIG
# =========================
EMPLOYEE_DATA_DIR = "lbph_data"
UNKNOWN_DATA_DIR = "unknown_data"

EMPLOYEE_MODEL_FILE = "employee_lbph_model.yml"
EMPLOYEE_LABELS_FILE = "employee_labels.npy"

UNKNOWN_MODEL_FILE = "unknown_lbph_model.yml"
UNKNOWN_LABELS_FILE = "unknown_labels.npy"

DB_FILE = "bizup_access.db"

# YuNet model
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
YUNET_SCORE_THRESHOLD = 0.85
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000
YUNET_INPUT_SIZE = (320, 320)

CAPTURE_DELAY_SEC = 0.9
COUNTDOWN_SEC = 3

POSES = ["LOOK STRAIGHT", "TURN LEFT", "TURN RIGHT", "LOOK UP", "LOOK DOWN"]
SAMPLES_PER_POSE = 5
SAMPLES_PER_PERSON = len(POSES) * SAMPLES_PER_POSE

EMP_CONF_THRESHOLD = 65
UNKNOWN_CONF_THRESHOLD = 60

MIN_FACE_SIZE = 80
RECOGNITION_FACE_SIZE = (200, 200)

LOG_COOLDOWN = 3
FACE_RESET_DELAY = 2.0
SESSION_TIMEOUT = 2.0

UNKNOWN_STABLE_SECONDS = 2.0
UNKNOWN_SAMPLES_TO_CREATE = 5

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "bizup/access/events"
MQTT_KEEPALIVE = 60

os.makedirs(EMPLOYEE_DATA_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DATA_DIR, exist_ok=True)

last_log_time = 0
last_logged_name = None
last_logged_action = None

current_identity = None
current_session_start = None
last_face_seen_time = 0

pending_unknown_start = None
pending_unknown_faces = []


# =========================
# MQTT
# =========================
def mqtt_connect():
    try:
        client = mqtt.Client()
        client.connect(MQTT_BROKER, MQTT_PORT, MQTT_KEEPALIVE)
        client.loop_start()
        print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        return client
    except Exception as e:
        print(f"[MQTT] Connection failed: {e}")
        return None


def mqtt_publish_event(client, person_id, action, confidence=None, extra=""):
    if client is None:
        return

    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "person_id": person_id,
        "action": action,
        "confidence": None if confidence is None else float(confidence),
        "extra": extra
    }

    try:
        client.publish(MQTT_TOPIC, json.dumps(payload))
    except Exception as e:
        print(f"[MQTT] Publish failed: {e}")


# =========================
# DATABASE
# =========================
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    CREATE TABLE IF NOT EXISTS access_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp TEXT,
        person_id TEXT,
        action TEXT,
        confidence REAL,
        extra TEXT
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS identities (
        person_id TEXT PRIMARY KEY,
        person_type TEXT,
        created_at TEXT,
        last_seen TEXT,
        total_time_seconds REAL DEFAULT 0
    )
    """)

    c.execute("""
    CREATE TABLE IF NOT EXISTS person_sessions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        person_id TEXT,
        session_start TEXT,
        session_end TEXT,
        duration_seconds REAL
    )
    """)

    conn.commit()
    conn.close()


def save_event(person, action, confidence=None, extra=""):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    INSERT INTO access_events (timestamp, person_id, action, confidence, extra)
    VALUES (?, ?, ?, ?, ?)
    """, (
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        person,
        action,
        confidence,
        extra
    ))

    conn.commit()
    conn.close()


def ensure_identity(person_id, person_type):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("SELECT person_id FROM identities WHERE person_id = ?", (person_id,))
    row = c.fetchone()

    if row is None:
        c.execute("""
        INSERT INTO identities (person_id, person_type, created_at, last_seen, total_time_seconds)
        VALUES (?, ?, ?, ?, 0)
        """, (person_id, person_type, now, now))
    else:
        c.execute("""
        UPDATE identities
        SET last_seen = ?
        WHERE person_id = ?
        """, (now, person_id))

    conn.commit()
    conn.close()


def update_identity_last_seen(person_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    c.execute("""
    UPDATE identities
    SET last_seen = ?
    WHERE person_id = ?
    """, (now, person_id))

    conn.commit()
    conn.close()


def add_session(person_id, start_ts, end_ts, duration_seconds):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    INSERT INTO person_sessions (person_id, session_start, session_end, duration_seconds)
    VALUES (?, ?, ?, ?)
    """, (person_id, start_ts, end_ts, duration_seconds))

    c.execute("""
    UPDATE identities
    SET total_time_seconds = COALESCE(total_time_seconds, 0) + ?,
        last_seen = ?
    WHERE person_id = ?
    """, (duration_seconds, end_ts, person_id))

    conn.commit()
    conn.close()


def get_total_time(person_id):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    SELECT total_time_seconds
    FROM identities
    WHERE person_id = ?
    """, (person_id,))
    row = c.fetchone()

    conn.close()
    return 0 if row is None or row[0] is None else float(row[0])


# =========================
# DRAWING
# =========================
def draw_center_guide(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.28)
    cv2.circle(frame, center, radius, (0, 255, 0), 2)


def format_seconds(seconds):
    seconds = int(seconds)
    mins = seconds // 60
    secs = seconds % 60
    hrs = mins // 60
    mins = mins % 60

    if hrs > 0:
        return f"{hrs:02d}:{mins:02d}:{secs:02d}"
    return f"{mins:02d}:{secs:02d}"


# =========================
# FACE DETECTION (YuNet)
# =========================
def preprocess_gray(gray):
    return cv2.equalizeHist(gray)


def create_yunet_detector():
    if not os.path.exists(YUNET_MODEL):
        raise FileNotFoundError(
            f"YuNet model not found: {YUNET_MODEL}\n"
            f"Put face_detection_yunet_2023mar.onnx in the same folder as this script."
        )

    detector = cv2.FaceDetectorYN.create(
        YUNET_MODEL,
        "",
        YUNET_INPUT_SIZE,
        YUNET_SCORE_THRESHOLD,
        YUNET_NMS_THRESHOLD,
        YUNET_TOP_K
    )
    return detector


def detect_face(frame, detector):
    """
    Detect the biggest face using YuNet.
    Returns:
        face_gray_resized, (x, y, w, h)
    """
    h, w = frame.shape[:2]
    detector.setInputSize((w, h))

    _, faces = detector.detect(frame)

    if faces is None or len(faces) == 0:
        return None, None

    # faces format: [x, y, w, h, score, lmk1x, lmk1y, ...]
    faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
    x, y, fw, fh = faces[0][:4]

    x, y, fw, fh = int(x), int(y), int(fw), int(fh)

    x = max(0, x)
    y = max(0, y)
    fw = min(fw, w - x)
    fh = min(fh, h - y)

    if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
        return None, None

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = preprocess_gray(gray)

    face = gray[y:y + fh, x:x + fw]
    if face.size == 0:
        return None, None

    face = cv2.resize(face, RECOGNITION_FACE_SIZE)
    return face, (x, y, fw, fh)


# =========================
# ID HELPERS
# =========================
def next_id(base_dir, prefix):
    existing = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith(prefix)
    ]

    nums = []
    for e in existing:
        try:
            nums.append(int(e.split("_")[1]))
        except Exception:
            pass

    n = (max(nums) + 1) if nums else 1
    return f"{prefix}{n:03d}"


def next_emp():
    return next_id(EMPLOYEE_DATA_DIR, "emp_")


def next_unknown():
    return next_id(UNKNOWN_DATA_DIR, "unknown_")


# =========================
# MODEL HELPERS
# =========================
def train_model_from_dir(data_dir, model_file, labels_file):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for name in sorted(os.listdir(data_dir)):
        person_dir = os.path.join(data_dir, name)

        if not os.path.isdir(person_dir):
            continue

        png_files = [fn for fn in os.listdir(person_dir) if fn.lower().endswith(".png")]
        if len(png_files) == 0:
            continue

        if name not in label_map:
            label_map[name] = label_id
            label_id += 1

        for fn in png_files:
            img_path = os.path.join(person_dir, fn)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, RECOGNITION_FACE_SIZE)
            images.append(img)
            labels.append(label_map[name])

    if len(images) < 2 or len(label_map) < 1:
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    recognizer.train(images, np.array(labels))
    recognizer.save(model_file)

    np.save(labels_file, np.array(list(label_map.items()), dtype=object))
    return recognizer, label_map


def load_model(model_file, labels_file):
    if not (os.path.exists(model_file) and os.path.exists(labels_file)):
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_file)

    pairs = np.load(labels_file, allow_pickle=True)
    label_map = {name: int(i) for name, i in pairs}
    return recognizer, label_map


def train_all_models(mqtt_client=None):
    emp_recognizer, emp_label_map = train_model_from_dir(
        EMPLOYEE_DATA_DIR,
        EMPLOYEE_MODEL_FILE,
        EMPLOYEE_LABELS_FILE
    )

    unk_recognizer, unk_label_map = train_model_from_dir(
        UNKNOWN_DATA_DIR,
        UNKNOWN_MODEL_FILE,
        UNKNOWN_LABELS_FILE
    )

    extra = f"employees={len(emp_label_map)} unknowns={len(unk_label_map)}"
    save_event("SYSTEM", "MODELS_TRAINED", extra=extra)
    mqtt_publish_event(mqtt_client, "SYSTEM", "MODELS_TRAINED", extra=extra)

    print(f"Employee identities: {len(emp_label_map)}")
    print(f"Unknown identities : {len(unk_label_map)}")

    return emp_recognizer, emp_label_map, unk_recognizer, unk_label_map


# =========================
# ENROLLMENT
# =========================
def enroll_employee(cap, detector, mqtt_client=None):
    emp = next_emp()
    person_dir = os.path.join(EMPLOYEE_DATA_DIR, emp)
    os.makedirs(person_dir, exist_ok=True)

    ensure_identity(emp, "employee")
    print(f"Starting enrollment: {emp}")

    start = time.time()
    while time.time() - start < COUNTDOWN_SEC:
        ok, frame = cap.read()
        if not ok:
            continue

        draw_center_guide(frame)
        remaining = COUNTDOWN_SEC - int(time.time() - start)

        cv2.putText(frame, f"ENROLLING {emp} - START IN {max(0, remaining)}",
                    (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, "Keep face centered",
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imshow("Bizup Face System", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
            save_event(emp, "ENROLL_CANCELLED")
            mqtt_publish_event(mqtt_client, emp, "ENROLL_CANCELLED")
            return

    collected = 0

    for pose in POSES:
        pose_samples = 0

        prep_start = time.time()
        while time.time() - prep_start < 1.2:
            ok, frame = cap.read()
            if not ok:
                continue

            draw_center_guide(frame)
            cv2.putText(frame, pose,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
            cv2.putText(frame, f"Get ready... {collected}/{SAMPLES_PER_PERSON}",
                        (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            cv2.imshow("Bizup Face System", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                save_event(emp, "ENROLL_CANCELLED")
                mqtt_publish_event(mqtt_client, emp, "ENROLL_CANCELLED")
                return

        while pose_samples < SAMPLES_PER_POSE:
            ok, frame = cap.read()
            if not ok:
                continue

            face, box = detect_face(frame, detector)

            draw_center_guide(frame)
            cv2.putText(frame, pose,
                        (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

            if face is not None:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                cv2.imwrite(os.path.join(person_dir, f"{ts}.png"), face)

                collected += 1
                pose_samples += 1

                x, y, w, h = box
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"Saved {collected}/{SAMPLES_PER_PERSON}",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                cv2.imshow("Bizup Face System", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    save_event(emp, "ENROLL_CANCELLED")
                    mqtt_publish_event(mqtt_client, emp, "ENROLL_CANCELLED")
                    return

                time.sleep(CAPTURE_DELAY_SEC)
            else:
                cv2.putText(frame, "No face detected",
                            (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

                cv2.imshow("Bizup Face System", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q")):
                    save_event(emp, "ENROLL_CANCELLED")
                    mqtt_publish_event(mqtt_client, emp, "ENROLL_CANCELLED")
                    return

    extra = f"samples={collected}"
    save_event(emp, "ENROLL_COMPLETE", extra=extra)
    mqtt_publish_event(mqtt_client, emp, "ENROLL_COMPLETE", extra=extra)

    print(f"Enrollment finished: {emp}")


# =========================
# UNKNOWN CREATION
# =========================
def create_unknown_identity(face_samples, mqtt_client=None):
    unk_id = next_unknown()
    person_dir = os.path.join(UNKNOWN_DATA_DIR, unk_id)
    os.makedirs(person_dir, exist_ok=True)

    ensure_identity(unk_id, "unknown")

    for i, face in enumerate(face_samples):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(os.path.join(person_dir, f"{ts}_{i}.png"), face)

    save_event(unk_id, "UNKNOWN_REGISTERED", extra=f"samples={len(face_samples)}")
    mqtt_publish_event(mqtt_client, unk_id, "UNKNOWN_REGISTERED", extra=f"samples={len(face_samples)}")

    return unk_id


# =========================
# RECOGNITION
# =========================
def predict_identity(face, emp_recognizer, emp_inv_map, unk_recognizer, unk_inv_map):
    """
    Returns:
        identity, confidence, person_type

    person_type in: employee / unknown / new_unknown_candidate
    """
    if emp_recognizer is not None:
        label, conf = emp_recognizer.predict(face)
        name = emp_inv_map.get(label)
        if name is not None and conf <= EMP_CONF_THRESHOLD:
            return name, conf, "employee"

    if unk_recognizer is not None:
        label, conf = unk_recognizer.predict(face)
        name = unk_inv_map.get(label)
        if name is not None and conf <= UNKNOWN_CONF_THRESHOLD:
            return name, conf, "unknown"

    return None, None, "new_unknown_candidate"


# =========================
# SESSION MANAGEMENT
# =========================
def start_session(person_id, person_type):
    global current_identity, current_session_start

    if current_identity != person_id:
        close_current_session_if_needed(force=True)

        current_identity = person_id
        current_session_start = time.time()
        ensure_identity(person_id, person_type)


def close_current_session_if_needed(force=False):
    global current_identity, current_session_start, last_face_seen_time

    if current_identity is None or current_session_start is None:
        return

    now = time.time()
    if force or (now - last_face_seen_time > SESSION_TIMEOUT):
        session_end_time = max(last_face_seen_time, current_session_start)
        start_dt = datetime.fromtimestamp(current_session_start).strftime("%Y-%m-%d %H:%M:%S")
        end_dt = datetime.fromtimestamp(session_end_time).strftime("%Y-%m-%d %H:%M:%S")
        duration = max(0, session_end_time - current_session_start)

        add_session(current_identity, start_dt, end_dt, duration)
        print(f"[SESSION CLOSED] {current_identity} duration={duration:.1f}s")

        current_identity = None
        current_session_start = None


# =========================
# MAIN
# =========================
def main():
    global last_log_time, last_logged_name, last_logged_action
    global last_face_seen_time
    global pending_unknown_start, pending_unknown_faces

    init_db()
    mqtt_client = mqtt_connect()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    yunet_detector = create_yunet_detector()

    emp_recognizer, emp_label_map = load_model(EMPLOYEE_MODEL_FILE, EMPLOYEE_LABELS_FILE)
    unk_recognizer, unk_label_map = load_model(UNKNOWN_MODEL_FILE, UNKNOWN_LABELS_FILE)

    emp_inv_map = {v: k for k, v in emp_label_map.items()}
    unk_inv_map = {v: k for k, v in unk_label_map.items()}

    print("Controls: E enroll employee | T train all | Q quit")
    print(f"Database: {DB_FILE}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_center_guide(frame)
        face, box = detect_face(frame, yunet_detector)
        now = time.time()

        if face is None:
            if now - last_face_seen_time > FACE_RESET_DELAY:
                pending_unknown_start = None
                pending_unknown_faces = []

            close_current_session_if_needed()
        else:
            last_face_seen_time = now

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

            identity, conf, person_type = predict_identity(
                face, emp_recognizer, emp_inv_map, unk_recognizer, unk_inv_map
            )

            if person_type in ("employee", "unknown"):
                pending_unknown_start = None
                pending_unknown_faces = []

                start_session(identity, person_type)
                update_identity_last_seen(identity)

                action = "ACCESS_GRANTED" if person_type == "employee" else "ACCESS_DENIED"

                should_log = (
                    (now - last_log_time > LOG_COOLDOWN) or
                    (identity != last_logged_name) or
                    (action != last_logged_action)
                )

                if should_log:
                    extra = f"type={person_type}"
                    save_event(identity, action, conf, extra)
                    mqtt_publish_event(mqtt_client, identity, action, conf, extra)

                    last_log_time = now
                    last_logged_name = identity
                    last_logged_action = action

                session_time = 0 if current_session_start is None else (now - current_session_start)
                total_time = get_total_time(identity) + session_time

                color = (0, 255, 0) if person_type == "employee" else (0, 0, 255)
                label_text = f"{identity} ({conf:.0f})"
                cv2.putText(frame, label_text, (x, y - 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                cv2.putText(frame, f"Session: {format_seconds(session_time)}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(frame, f"Total: {format_seconds(total_time)}", (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            elif person_type == "new_unknown_candidate":
                action = "ACCESS_DENIED"

                if pending_unknown_start is None:
                    pending_unknown_start = now
                    pending_unknown_faces = [face.copy()]
                else:
                    if len(pending_unknown_faces) < UNKNOWN_SAMPLES_TO_CREATE:
                        pending_unknown_faces.append(face.copy())

                stable_time = now - pending_unknown_start if pending_unknown_start else 0

                if stable_time >= UNKNOWN_STABLE_SECONDS and len(pending_unknown_faces) >= UNKNOWN_SAMPLES_TO_CREATE:
                    new_unknown_id = create_unknown_identity(pending_unknown_faces, mqtt_client)

                    unk_recognizer, unk_label_map = train_model_from_dir(
                        UNKNOWN_DATA_DIR,
                        UNKNOWN_MODEL_FILE,
                        UNKNOWN_LABELS_FILE
                    )
                    unk_inv_map = {v: k for k, v in unk_label_map.items()}

                    start_session(new_unknown_id, "unknown")
                    update_identity_last_seen(new_unknown_id)

                    extra = "new persistent unknown created"
                    save_event(new_unknown_id, action, None, extra)
                    mqtt_publish_event(mqtt_client, new_unknown_id, action, None, extra)

                    last_log_time = now
                    last_logged_name = new_unknown_id
                    last_logged_action = action

                    pending_unknown_start = None
                    pending_unknown_faces = []

                    session_time = 0 if current_session_start is None else (now - current_session_start)
                    total_time = get_total_time(new_unknown_id) + session_time

                    cv2.putText(frame, f"{new_unknown_id} (NEW)", (x, y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame, f"Session: {format_seconds(session_time)}", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.putText(frame, f"Total: {format_seconds(total_time)}", (x, y + h + 25),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "UNKNOWN - Stabilizing...", (x, y - 35),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(frame,
                                f"{stable_time:.1f}s / {UNKNOWN_STABLE_SECONDS:.1f}s",
                                (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                    should_log = (
                        (now - last_log_time > LOG_COOLDOWN) or
                        ("UNKNOWN_CANDIDATE" != last_logged_name) or
                        (action != last_logged_action)
                    )

                    if should_log:
                        save_event("UNKNOWN_CANDIDATE", action, None, "waiting for stable registration")
                        mqtt_publish_event(mqtt_client, "UNKNOWN_CANDIDATE", action, None,
                                           "waiting for stable registration")

                        last_log_time = now
                        last_logged_name = "UNKNOWN_CANDIDATE"
                        last_logged_action = action

        cv2.putText(frame, "E enroll | T train all | Q quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bizup Face System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("e"), ord("E")):
            enroll_employee(cap, yunet_detector, mqtt_client)
        elif key in (ord("t"), ord("T")):
            emp_recognizer, emp_label_map, unk_recognizer, unk_label_map = train_all_models(mqtt_client)
            emp_inv_map = {v: k for k, v in emp_label_map.items()}
            unk_inv_map = {v: k for k, v in unk_label_map.items()}
        elif key in (ord("q"), ord("Q")):
            break

    close_current_session_if_needed(force=True)

    cap.release()
    cv2.destroyAllWindows()

    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()


if __name__ == "__main__":
    main()