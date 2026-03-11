from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

# ===== Camera Source =====
CAMERA_SOURCE = 0

# ===== Models =====
YUNET_MODEL = MODELS_DIR / "face_detection_yunet_2023mar.onnx"
SFACE_MODEL = MODELS_DIR / "face_recognition_sface_2021dec.onnx"

# ===== Camera Identity / Zone =====
CAMERA_ID = "cam_001"
DEFAULT_ZONE_NAME = "Main Entrance"
DEFAULT_ZONE_TYPE = "access"
DEFAULT_IS_ACCESS_POINT = 1

# ===== Session Timeouts =====
VISIBLE_SESSION_TIMEOUT_SEC = 10
ACCESS_SESSION_TIMEOUT_SEC = 30

# ===== Camera =====
CAMERA_INDEX = 0
FRAME_WIDTH = 960
FRAME_HEIGHT = 540
WINDOW_NAME = "Face Tracking Test"

# ===== Unknown handling =====
UNKNOWN_EMBEDDINGS_TO_SAVE = 5

# ===== Detection =====
DETECTION_INPUT_SIZE = (320, 320)
SCORE_THRESHOLD = 0.82
NMS_THRESHOLD = 0.3
TOP_K = 5000
MIN_FACE_WIDTH = 80
MIN_FACE_HEIGHT = 80
MIN_DETECTION_SCORE = 0.82

# ===== Tracking =====
TRACK_MATCH_DISTANCE_PX = 120
TRACK_MAX_MISSING_FRAMES = 10
TRACK_MIN_STABLE_FRAMES = 3

# ===== Recognition =====
RECOGNITION_MATCH_THRESHOLD = 0.45
RECOGNITION_COOLDOWN_SEC = 2.0
UNKNOWN_PERSON_NAME = "UNKNOWN"

# ===== Database =====
DB_FILE = BASE_DIR / "bizup_enterprise.db"

# ===== MQTT =====
MQTT_ENABLED = True
MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_KEEPALIVE = 60
MQTT_TOPIC_ACCESS = "bizup/access"
MQTT_TOPIC_ALERT = "bizup/alert"

# ===== Events / Sessions =====
EVENT_COOLDOWN_SEC = 5
LOG_COOLDOWN_SEC = 5
SESSION_TIMEOUT_SEC = 30

def validate_config():
    missing = []

    if not YUNET_MODEL.exists():
        missing.append(f"Missing YuNet model: {YUNET_MODEL}")

    if not SFACE_MODEL.exists():
        missing.append(f"Missing SFace model: {SFACE_MODEL}")

    if missing:
        raise FileNotFoundError("\n".join(missing))