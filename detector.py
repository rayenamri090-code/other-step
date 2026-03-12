import cv2

from config import (
    YUNET_MODEL,
    DETECTION_INPUT_SIZE,
    SCORE_THRESHOLD,
    NMS_THRESHOLD,
    TOP_K,
    MIN_FACE_WIDTH,
    MIN_FACE_HEIGHT,
    MIN_DETECTION_SCORE,
)


class FaceDetector:
    def __init__(self):
        if not YUNET_MODEL.exists():
            raise FileNotFoundError(f"Missing YuNet model: {YUNET_MODEL}")

        self.detector = cv2.FaceDetectorYN.create(
            str(YUNET_MODEL),
            "",
            DETECTION_INPUT_SIZE,
            SCORE_THRESHOLD,
            NMS_THRESHOLD,
            TOP_K,
        )

    def detect_all(self, frame):
        if frame is None or frame.size == 0:
            return []

        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))

        try:
            _, faces = self.detector.detect(frame)
        except Exception:
            return []

        results = []
        if faces is None:
            return results

        for row in faces:
            x, y, fw, fh = [int(v) for v in row[:4]]

            if fw < MIN_FACE_WIDTH or fh < MIN_FACE_HEIGHT:
                continue

            score = float(row[-1]) if len(row) > 14 else 0.0
            if score < MIN_DETECTION_SCORE:
                continue

            # clamp bbox to image bounds
            x = max(0, x)
            y = max(0, y)
            fw = min(fw, w - x)
            fh = min(fh, h - y)

            if fw <= 0 or fh <= 0:
                continue

            results.append({
                "face_row": row,
                "bbox": (x, y, fw, fh),
                "score": score,
            })

        return results