import cv2
import numpy as np
import onnxruntime as ort


class EmotionService:

    EMOTION_LABELS = [
        "anger",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    ]

    def __init__(self, model_path: str):
        self.session = ort.InferenceSession(
            model_path,
            providers=["CPUExecutionProvider"],
        )

        self.input_name = self.session.get_inputs()[0].name

    def _crop_face(self, frame, bbox, pad=0.20):
        x, y, w, h = bbox
        H, W = frame.shape[:2]

        px = int(w * pad)
        py = int(h * pad)

        x1 = max(0, x - px)
        y1 = max(0, y - py)
        x2 = min(W, x + w + px)
        y2 = min(H, y + h + py)

        crop = frame[y1:y2, x1:x2]
        if crop is None or crop.size == 0:
            return None

        if crop.shape[0] < 60 or crop.shape[1] < 60:
            return None

        return crop

    def _preprocess(self, face):
        face = cv2.resize(face, (224, 224))
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = face.astype(np.float32) / 255.0
        face = np.transpose(face, (2, 0, 1))
        face = np.expand_dims(face, axis=0)
        return face

    def predict_emotion(self, frame, bbox):

        crop = self._crop_face(frame, bbox)
        if crop is None:
            return None, None

        tensor = self._preprocess(crop)

        logits = self.session.run(None, {self.input_name: tensor})[0][0]
        probs = np.exp(logits) / np.sum(np.exp(logits))

        idx = int(np.argmax(probs))

        return self.EMOTION_LABELS[idx], float(probs[idx])