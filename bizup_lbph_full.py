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
DATA_DIR = "lbph_data"
MODEL_FILE = "lbph_model.yml"
LABELS_FILE = "labels.npy"
DB_FILE = "bizup_access.db"
UNKNOWN_DIR = "unknown_faces"

CAPTURE_DELAY_SEC = 0.9
COUNTDOWN_SEC = 3

POSES = ["LOOK STRAIGHT", "TURN LEFT", "TURN RIGHT", "LOOK UP", "LOOK DOWN"]
SAMPLES_PER_POSE = 5
SAMPLES_PER_PERSON = len(POSES) * SAMPLES_PER_POSE

CONF_THRESHOLD = 65
LOG_COOLDOWN = 3
MIN_FACE_SIZE = 80
RECOGNITION_FACE_SIZE = (200, 200)

# Unknown handling
FACE_RESET_DELAY = 2.0   # reset unknown snapshot lock if no face is seen for N seconds

# MQTT
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "bizup/access/events"
MQTT_KEEPALIVE = 60

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(UNKNOWN_DIR, exist_ok=True)

CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_log_time = 0
last_logged_name = None
last_logged_action = None

unknown_already_saved = False
last_face_seen_time = 0


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


# =========================
# DRAWING
# =========================
def draw_center_guide(frame):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    radius = int(min(w, h) * 0.28)
    cv2.circle(frame, center, radius, (0, 255, 0), 2)


# =========================
# FACE DETECTION
# =========================
def preprocess_gray(gray):
    return cv2.equalizeHist(gray)

def detect_face(gray):
    gray = preprocess_gray(gray)

    faces = CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=6,
        minSize=(MIN_FACE_SIZE, MIN_FACE_SIZE)
    )

    if len(faces) == 0:
        return None, None

    x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
    face = gray[y:y + h, x:x + w]

    if face.shape[0] < MIN_FACE_SIZE or face.shape[1] < MIN_FACE_SIZE:
        return None, None

    face = cv2.resize(face, RECOGNITION_FACE_SIZE)
    return face, (x, y, w, h)


# =========================
# UNKNOWN SNAPSHOT
# =========================
def save_unknown(frame):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(UNKNOWN_DIR, f"unknown_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path


# =========================
# EMPLOYEE IDS
# =========================
def next_emp():
    existing = [
        d for d in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, d)) and d.startswith("emp_")
    ]

    nums = []
    for e in existing:
        try:
            nums.append(int(e.split("_")[1]))
        except:
            pass

    n = (max(nums) + 1) if nums else 1
    return f"emp_{n:03d}"


# =========================
# MODEL
# =========================
def train_model(mqtt_client=None):
    images = []
    labels = []
    label_map = {}
    label_id = 0

    for name in sorted(os.listdir(DATA_DIR)):
        person_dir = os.path.join(DATA_DIR, name)

        if not os.path.isdir(person_dir):
            continue

        if name not in label_map:
            label_map[name] = label_id
            label_id += 1

        for fn in os.listdir(person_dir):
            if not fn.lower().endswith(".png"):
                continue

            img_path = os.path.join(person_dir, fn)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                continue

            img = cv2.resize(img, RECOGNITION_FACE_SIZE)
            images.append(img)
            labels.append(label_map[name])

    if len(images) < 2:
        print("Not enough samples to train.")
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create(
        radius=1,
        neighbors=8,
        grid_x=8,
        grid_y=8
    )

    recognizer.train(images, np.array(labels))
    recognizer.save(MODEL_FILE)

    np.save(LABELS_FILE, np.array(list(label_map.items()), dtype=object))

    extra = f"ids={len(label_map)} samples={len(images)}"
    save_event("SYSTEM", "MODEL_TRAINED", extra=extra)
    mqtt_publish_event(mqtt_client, "SYSTEM", "MODEL_TRAINED", extra=extra)

    print(f"Model trained: {len(label_map)} identities, {len(images)} samples.")
    return recognizer, label_map

def load_model():
    if not (os.path.exists(MODEL_FILE) and os.path.exists(LABELS_FILE)):
        return None, {}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(MODEL_FILE)

    pairs = np.load(LABELS_FILE, allow_pickle=True)
    label_map = {name: int(i) for name, i in pairs}

    return recognizer, label_map


# =========================
# ENROLLMENT
# =========================
def enroll(cap, mqtt_client=None):
    emp = next_emp()
    person_dir = os.path.join(DATA_DIR, emp)
    os.makedirs(person_dir, exist_ok=True)

    print(f"Starting enrollment: {emp}")

    # Countdown
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

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            face, box = detect_face(gray)

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
# MAIN
# =========================
def main():
    global last_log_time, last_logged_name, last_logged_action
    global unknown_already_saved, last_face_seen_time

    init_db()
    mqtt_client = mqtt_connect()

    cap = cv2.VideoCapture(0)
    recognizer, label_map = load_model()
    inv_map = {v: k for k, v in label_map.items()}

    print("Controls: E enroll | T train | Q quit")
    print(f"Database: {DB_FILE}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        draw_center_guide(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face, box = detect_face(gray)

        # Reset state if no face has been seen for a while
        if face is None:
            if time.time() - last_face_seen_time > FACE_RESET_DELAY:
                unknown_already_saved = False
                last_logged_name = None
                last_logged_action = None

        if face is not None:
            last_face_seen_time = time.time()

            x, y, w, h = box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            if recognizer is not None:
                face = cv2.resize(face, RECOGNITION_FACE_SIZE)
                label, conf = recognizer.predict(face)

                name = inv_map.get(label, "UNKNOWN")
                if conf > CONF_THRESHOLD:
                    name = "UNKNOWN"

                action = "ACCESS_GRANTED" if name != "UNKNOWN" else "ACCESS_DENIED"

                now = time.time()
                should_log = (
                    (now - last_log_time > LOG_COOLDOWN) or
                    (name != last_logged_name) or
                    (action != last_logged_action)
                )

                if should_log:
                    extra = ""

                    if name == "UNKNOWN":
                        if not unknown_already_saved:
                            extra = save_unknown(frame)
                            unknown_already_saved = True
                    else:
                        unknown_already_saved = False

                    save_event(name, action, conf, extra)
                    mqtt_publish_event(mqtt_client, name, action, conf, extra)

                    last_log_time = now
                    last_logged_name = name
                    last_logged_action = action

                color = (0, 255, 0) if name != "UNKNOWN" else (0, 0, 255)
                cv2.putText(frame, f"{name} ({conf:.0f})",
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(frame, "No model trained (press T)",
                            (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.putText(frame, "E enroll | T train | Q quit",
                    (10, frame.shape[0] - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Bizup Face System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("e"), ord("E")):
            enroll(cap, mqtt_client)
        elif key in (ord("t"), ord("T")):
            recognizer, label_map = train_model(mqtt_client)
            inv_map = {v: k for k, v in label_map.items()}
        elif key in (ord("q"), ord("Q")):
            break

    cap.release()
    cv2.destroyAllWindows()

    if mqtt_client is not None:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

if __name__ == "__main__":
    main()