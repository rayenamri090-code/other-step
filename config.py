import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DB_FILE = os.path.join(BASE_DIR, "bizup_enterprise.db")

CAMERA_ID = "cam_001"
DEFAULT_ZONE_NAME = "entry_zone"
DEFAULT_ZONE_TYPE = "entry"
DEFAULT_IS_ACCESS_POINT = 1

YUNET_MODEL = os.path.join(MODELS_DIR, "face_detection_yunet_2023mar.onnx")
SFACE_MODEL = os.path.join(MODELS_DIR, "face_recognition_sface_2021dec.onnx")

YUNET_INPUT_SIZE = (320, 320)
YUNET_SCORE_THRESHOLD = 0.85
YUNET_NMS_THRESHOLD = 0.3
YUNET_TOP_K = 5000

MIN_FACE_SIZE = 70
COSINE_THRESHOLD = 0.42

TRACK_MAX_MISSING_FRAMES = 20
TRACK_MATCH_DISTANCE = 120

UNKNOWN_STABLE_FRAMES = 8
UNKNOWN_EMBEDDINGS_TO_SAVE = 5

VISIBLE_SESSION_TIMEOUT_SEC = 2.0
ACCESS_SESSION_TIMEOUT_SEC = 10.0
LOG_COOLDOWN_SEC = 3.0

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "bizup/enterprise/events"
MQTT_KEEPALIVE = 60

WINDOW_NAME = "Bizup Enterprise Prototype"
CAMERA_SOURCE = 0