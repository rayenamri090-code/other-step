import os
import cv2
from config import (
    YUNET_MODEL,
    YUNET_INPUT_SIZE,
    YUNET_SCORE_THRESHOLD,
    YUNET_NMS_THRESHOLD,
    YUNET_TOP_K,
    MIN_FACE_SIZE,
)


class FaceDetector:
    def __init__(self):
        if not os.path.exists(YUNET_MODEL):
            raise FileNotFoundError(f"Missing YuNet model: {YUNET_MODEL}")

        self.detector = cv2.FaceDetectorYN.create(
            YUNET_MODEL,
            "",
            YUNET_INPUT_SIZE,
            YUNET_SCORE_THRESHOLD,
            YUNET_NMS_THRESHOLD,
            YUNET_TOP_K
        )

    def detect_all(self, frame):
        h, w = frame.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)

        results = []
        if faces is None:
            return results

        for row in faces:
            x, y, fw, fh = [int(v) for v in row[:4]]
            if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
                continue

            score = float(row[14]) if len(row) > 14 else 0.0
            results.append({
                "face_row": row,
                "bbox": (x, y, fw, fh),
                "score": score
            })

        return results