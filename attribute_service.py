import cv2
import numpy as np

from config import (
    GENDER_PROTO,
    GENDER_MODEL,
    GENDER_CONFIDENCE_MIN,
)


class AttributeService:
    """
    Auxiliary attribute analysis only.
    Not for authentication.
    Not for authorization.
    """

    GENDER_LABELS = ["male", "female"]

    def __init__(self):
        self.gender_enabled = False
        self.gender_net = None
        self.gender_confidence_min = GENDER_CONFIDENCE_MIN

        if GENDER_PROTO.exists() and GENDER_MODEL.exists():
            try:
                self.gender_net = cv2.dnn.readNet(str(GENDER_MODEL), str(GENDER_PROTO))
                self.gender_enabled = True
                print("[ATTR] Gender model loaded")
            except Exception as e:
                self.gender_enabled = False
                self.gender_net = None
                print(f"[ATTR] Gender model load failed: {e}")
        else:
            print("[ATTR] Gender model files not found, gender prediction disabled")

    def _safe_crop_face(self, frame, bbox, pad_ratio=0.15):
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        pad_x = int(bw * pad_ratio)
        pad_y = int(bh * pad_ratio)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        return crop

    def predict_gender(self, frame, bbox):
        if not self.gender_enabled or self.gender_net is None:
            return {
                "gender_prediction": None,
                "gender_confidence": None,
            }

        face_crop = self._safe_crop_face(frame, bbox)
        if face_crop is None:
            return {
                "gender_prediction": None,
                "gender_confidence": None,
            }

        try:
            blob = cv2.dnn.blobFromImage(
                image=face_crop,
                scalefactor=1.0,
                size=(227, 227),
                mean=(78.4263377603, 87.7689143744, 114.895847746),
                swapRB=False,
                crop=False,
            )

            self.gender_net.setInput(blob)
            preds = self.gender_net.forward().flatten()

            if preds.size < 2:
                return {
                    "gender_prediction": None,
                    "gender_confidence": None,
                }

            idx = int(np.argmax(preds))
            conf = float(preds[idx])

            if conf < self.gender_confidence_min:
                return {
                    "gender_prediction": None,
                    "gender_confidence": conf,
                }

            return {
                "gender_prediction": self.GENDER_LABELS[idx],
                "gender_confidence": conf,
            }

        except Exception:
            return {
                "gender_prediction": None,
                "gender_confidence": None,
            }