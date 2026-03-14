import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms

from config import MODELS_DIR, GENDER_CONFIDENCE_MIN


class AttributeService:
    """
    Auxiliary attribute analysis only.
    Not for authentication.
    Not for authorization.

    FairFace-based age + gender prediction.
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

    def __init__(self):
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

    def _empty_result(self):
        return {
            "gender_prediction": None,
            "gender_confidence": None,
            "age_prediction": None,
            "age_confidence": None,
        }

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

    def _prepare_tensor(self, face_crop):
        if face_crop is None or face_crop.size == 0:
            return None

        rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(rgb).unsqueeze(0).to(self.device)
        return tensor

    def predict_attributes(self, frame, bbox):
        """
        Returns:
        {
            "gender_prediction": str | None,
            "gender_confidence": float | None,
            "age_prediction": str | None,
            "age_confidence": float | None,
        }

        Important:
        - gender_prediction is filtered by confidence threshold
        - age_prediction is always returned when inference succeeds
        - caller decides stability and DB locking
        """
        if self.model is None:
            return self._empty_result()

        face_crop = self._safe_crop_face(frame, bbox)
        if face_crop is None:
            return self._empty_result()

        if not self._is_crop_quality_good_enough(face_crop):
            return self._empty_result()

        try:
            tensor = self._prepare_tensor(face_crop)
            if tensor is None:
                return self._empty_result()

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

            return {
                "gender_prediction": gender_prediction,
                "gender_confidence": gender_conf,
                "age_prediction": age_prediction,
                "age_confidence": age_conf,
            }

        except Exception as e:
            print(f"[ATTR] Prediction failed: {e}")
            return self._empty_result()

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