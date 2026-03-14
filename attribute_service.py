from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from config import MODELS_DIR, GENDER_CONFIDENCE_MIN


class AttributeService:
    """
    Auxiliary attribute analysis only.
    Not for authentication.
    Not for authorization.

    Supports:
    - FairFace age + gender prediction
    - ONNX emotion prediction

    Important:
    - identity/auth/access decisions must NOT depend on these outputs
    - caller decides stability voting / locking / DB persistence
    """

    GENDER_LABELS = ["male", "female"]

    AGE_LABELS = [
        "0-2",
        "3-9",
        "10-19",
        "20-29",
        "30-39",
        "40-49",
        "50-59",
        "60-69",
        "70+",
    ]

    # enet_b0_8_best_afew
    EMOTION_LABELS = [
        "angry",
        "contempt",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprised",
    ]

    def __init__(self):
        # -----------------------------
        # FairFace age/gender
        # -----------------------------
        self.gender_enabled = False
        self.age_enabled = False
        self.model = None
        self.gender_confidence_min = GENDER_CONFIDENCE_MIN

        self.model_path = MODELS_DIR / "fairface_alldata_20191111.pt"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        if self.model_path.exists():
            try:
                self.model = models.resnet34(weights=None)
                self.model.fc = torch.nn.Linear(self.model.fc.in_features, 18)

                state_dict = torch.load(self.model_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
                self.model.to(self.device)
                self.model.eval()

                self.gender_enabled = True
                self.age_enabled = True
                print("[ATTR] FairFace age+gender model loaded")
            except Exception as e:
                self.model = None
                self.gender_enabled = False
                self.age_enabled = False
                print(f"[ATTR] FairFace model load failed: {e}")
        else:
            print("[ATTR] FairFace model file not found, age/gender prediction disabled")

        # -----------------------------
        # Emotion ONNX
        # -----------------------------
        self.emotion_enabled = False
        self.emotion_session = None
        self.emotion_input_name = None
        self.emotion_output_name = None

        self.emotion_model_path = MODELS_DIR / "emotion" / "enet_b0_8_best_afew.onnx"

        if self.emotion_model_path.exists():
            try:
                providers = ["CPUExecutionProvider"]
                self.emotion_session = ort.InferenceSession(
                    str(self.emotion_model_path),
                    providers=providers,
                )

                self.emotion_input_name = self.emotion_session.get_inputs()[0].name
                self.emotion_output_name = self.emotion_session.get_outputs()[0].name

                self.emotion_enabled = True
                print("[ATTR] Emotion ONNX model loaded")
            except Exception as e:
                self.emotion_enabled = False
                self.emotion_session = None
                print(f"[ATTR] Emotion model load failed: {e}")
        else:
            print("[ATTR] Emotion model file not found, emotion prediction disabled")

    # =========================================================
    # Empty Results
    # =========================================================

    def _empty_result(self):
        return {
            "gender_prediction": None,
            "gender_confidence": None,
            "age_prediction": None,
            "age_confidence": None,
            "emotion_prediction": None,
            "emotion_confidence": None,
        }

    def _empty_emotion_result(self):
        return {
            "emotion_prediction": None,
            "emotion_confidence": None,
        }

    # =========================================================
    # Face Crop Helpers
    # =========================================================

    def _safe_crop_face(self, frame, bbox, pad_ratio=0.22):
        if frame is None or frame.size == 0:
            return None

        h, w = frame.shape[:2]
        x, y, bw, bh = bbox

        if bw <= 0 or bh <= 0:
            return None

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

        ch, cw = crop.shape[:2]
        if cw < 60 or ch < 60:
            return None

        return crop

    def _is_crop_quality_good_enough(self, face_crop):
        """
        Cheap quality gate to reduce noisy predictions.
        Rejects very dark, very blurry, or too-small crops.
        """
        if face_crop is None or face_crop.size == 0:
            return False

        h, w = face_crop.shape[:2]
        if w < 60 or h < 60:
            return False

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

        mean_brightness = float(np.mean(gray))
        if mean_brightness < 25:
            return False

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < 35:
            return False

        return True

    # =========================================================
    # FairFace Helpers
    # =========================================================

    def _prepare_tensor(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    # =========================================================
    # Emotion Helpers
    # =========================================================

    def _prepare_emotion_input(self, face_crop):
        """
        Preprocess for enet_b0_8_best_afew ONNX model.
        Shape: (1, 3, 224, 224), float32, normalized to ImageNet stats.
        """
        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = resized.astype(np.float32) / 255.0

        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        img = (img - mean) / std
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, axis=0).astype(np.float32)

        return img

    # =========================================================
    # Main Predictions
    # =========================================================

    def predict_attributes(self, frame, bbox):
        """
        Returns:
        {
            "gender_prediction": str | None,
            "gender_confidence": float | None,
            "age_prediction": str | None,
            "age_confidence": float | None,
            "emotion_prediction": str | None,
            "emotion_confidence": float | None,
        }

        Important:
        - gender_prediction is filtered by confidence threshold
        - age_prediction is returned when inference succeeds
        - emotion_prediction is returned when inference succeeds
        - caller decides stability and DB locking
        """
        result = self._empty_result()

        face_crop = self._safe_crop_face(frame, bbox)
        if face_crop is None:
            return result

        if not self._is_crop_quality_good_enough(face_crop):
            return result

        # -----------------------------
        # FairFace age/gender
        # -----------------------------
        if self.model is not None:
            try:
                tensor = self._prepare_tensor(face_crop)
                if tensor is not None:
                    with torch.no_grad():
                        logits = self.model(tensor).squeeze(0)

                    # FairFace flattened output layout:
                    # race   = logits[0:7]
                    # gender = logits[7:9]
                    # age    = logits[9:18]
                    gender_logits = logits[7:9]
                    age_logits = logits[9:18]

                    if len(self.GENDER_LABELS) != len(gender_logits):
                        raise ValueError(
                            f"GENDER_LABELS has {len(self.GENDER_LABELS)} labels but model returned {len(gender_logits)} gender logits"
                        )

                    if len(self.AGE_LABELS) != len(age_logits):
                        raise ValueError(
                            f"AGE_LABELS has {len(self.AGE_LABELS)} labels but model returned {len(age_logits)} age logits"
                        )

                    gender_probs = F.softmax(gender_logits, dim=0).detach().cpu().numpy()
                    age_probs = F.softmax(age_logits, dim=0).detach().cpu().numpy()

                    gender_idx = int(np.argmax(gender_probs))
                    age_idx = int(np.argmax(age_probs))

                    gender_conf = float(gender_probs[gender_idx])
                    age_conf = float(age_probs[age_idx])

                    gender_prediction = None
                    if gender_conf >= self.gender_confidence_min:
                        gender_prediction = self.GENDER_LABELS[gender_idx]

                    age_prediction = self.AGE_LABELS[age_idx]

                    result["gender_prediction"] = gender_prediction
                    result["gender_confidence"] = gender_conf
                    result["age_prediction"] = age_prediction
                    result["age_confidence"] = age_conf

            except Exception as e:
                print(f"[ATTR] Age/gender prediction failed: {e}")

        # -----------------------------
        # Emotion ONNX
        # -----------------------------
        if self.emotion_enabled and self.emotion_session is not None:
            try:
                emotion_input = self._prepare_emotion_input(face_crop)
                outputs = self.emotion_session.run(
                    [self.emotion_output_name],
                    {self.emotion_input_name: emotion_input},
                )

                logits = outputs[0]
                logits = np.array(logits).squeeze()

                if logits.ndim != 1:
                    raise ValueError(f"Unexpected emotion logits shape: {logits.shape}")

                exp_logits = np.exp(logits - np.max(logits))
                probs = exp_logits / np.sum(exp_logits)

                emotion_idx = int(np.argmax(probs))
                emotion_conf = float(probs[emotion_idx])

                if emotion_idx < 0 or emotion_idx >= len(self.EMOTION_LABELS):
                    raise ValueError(f"Emotion index out of range: {emotion_idx}")

                emotion_prediction = self.EMOTION_LABELS[emotion_idx]

                result["emotion_prediction"] = emotion_prediction
                result["emotion_confidence"] = emotion_conf

            except Exception as e:
                print(f"[ATTR] Emotion prediction failed: {e}")

        return result

    # =========================================================
    # Convenience Methods
    # =========================================================

    def predict_gender(self, frame, bbox):
        result = self.predict_attributes(frame, bbox)
        return {
            "gender_prediction": result.get("gender_prediction"),
            "gender_confidence": result.get("gender_confidence"),
        }

    def predict_age(self, frame, bbox):
        result = self.predict_attributes(frame, bbox)
        return {
            "age_prediction": result.get("age_prediction"),
            "age_confidence": result.get("age_confidence"),
        }

    def predict_emotion(self, frame, bbox):
        result = self.predict_attributes(frame, bbox)
        return {
            "emotion_prediction": result.get("emotion_prediction"),
            "emotion_confidence": result.get("emotion_confidence"),
        }