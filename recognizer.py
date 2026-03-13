import json
import cv2
import numpy as np

from config import SFACE_MODEL, RECOGNITION_MATCH_THRESHOLD
from database import get_all_identities_with_embeddings


def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    if norm < 1e-12:
        return emb
    return emb / norm


def cosine_similarity(a, b):
    return float(np.dot(a, b))


class FaceRecognizer:
    def __init__(self):
        if not SFACE_MODEL.exists():
            raise FileNotFoundError(f"Missing SFace model: {SFACE_MODEL}")

        self.recognizer = cv2.FaceRecognizerSF.create(str(SFACE_MODEL), "")
        self.threshold = RECOGNITION_MATCH_THRESHOLD

        # Structure:
        # {
        #   "employee": { "emp_001": [emb1, emb2, ...] },
        #   "visitor":  { ... },
        #   "unknown":  { ... }
        # }
        self.embeddings = {
            "employee": {},
            "visitor": {},
            "unknown": {},
        }

        self.reload_embeddings()

    def reload_embeddings(self):
        self.embeddings = {
            "employee": {},
            "visitor": {},
            "unknown": {},
        }

        rows = get_all_identities_with_embeddings()

        for person_id, person_type, status, emb_json in rows:
            if status in ("blocked", "merged", "inactive"):
                continue

            try:
                emb = normalize_embedding(json.loads(emb_json))
            except Exception:
                continue

            if emb.size == 0:
                continue

            self.embeddings.setdefault(person_type, {}).setdefault(person_id, []).append(emb)

    def extract_embedding(self, frame, face_row):
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned)
        return normalize_embedding(feat)

    def _best_match_in_type(self, query_emb, person_type):
        best_id = None
        best_score = -1.0

        for person_id, emb_list in self.embeddings.get(person_type, {}).items():
            if not emb_list:
                continue

            score = max(cosine_similarity(query_emb, emb) for emb in emb_list)
            if score > best_score:
                best_score = score
                best_id = person_id

        return best_id, best_score

    def _best_match_global(self, query_emb):
        best_person_id = None
        best_person_type = None
        best_score = -1.0

        for person_type in ("employee", "visitor", "unknown"):
            person_id, score = self._best_match_in_type(query_emb, person_type)

            if person_id is not None and score > best_score:
                best_person_id = person_id
                best_person_type = person_type
                best_score = score

        return best_person_id, best_person_type, best_score

    def recognize(self, query_emb):
        person_id, person_type, score = self._best_match_global(query_emb)

        if person_id is not None and score >= self.threshold:
            return {
                "person_id": person_id,
                "person_type": person_type,
                "score": float(score),
            }

        return {
            "person_id": None,
            "person_type": None,
            "score": None,
        }

    def recognize_top_k(self, query_emb, k=3):
        candidates = []

        for person_type, people in self.embeddings.items():
            for person_id, emb_list in people.items():
                if not emb_list:
                    continue

                score = max(cosine_similarity(query_emb, emb) for emb in emb_list)
                candidates.append({
                    "person_id": person_id,
                    "person_type": person_type,
                    "score": float(score),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates[:k]

    def has_embeddings(self, person_type=None):
        if person_type is None:
            return any(
                len(people) > 0
                for people in self.embeddings.values()
            )

        return len(self.embeddings.get(person_type, {})) > 0