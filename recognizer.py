import json
import cv2
import numpy as np

from config import SFACE_MODEL, RECOGNITION_MATCH_THRESHOLD
from database import get_all_identities_with_embeddings


# Optional ambiguity tuning
RECOGNITION_MIN_MARGIN = 0.02


def normalize_embedding(embedding):
    emb = np.asarray(embedding, dtype=np.float32).flatten()
    norm = np.linalg.norm(emb)
    if norm < 1e-12:
        return emb
    return emb / norm


def cosine_similarity(a, b):
    a = np.asarray(a, dtype=np.float32).flatten()
    b = np.asarray(b, dtype=np.float32).flatten()

    if a.size == 0 or b.size == 0:
        return -1.0

    if a.shape != b.shape:
        return -1.0

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

            if person_type not in ("employee", "visitor", "unknown"):
                continue

            try:
                emb = normalize_embedding(json.loads(emb_json))
            except Exception:
                continue

            if emb.size == 0:
                continue

            self.embeddings[person_type].setdefault(person_id, []).append(emb)

    def extract_embedding(self, frame, face_row):
        aligned = self.recognizer.alignCrop(frame, face_row)
        feat = self.recognizer.feature(aligned)
        return normalize_embedding(feat)

    def _score_embedding_list(self, query_emb, emb_list):
        """
        Current strategy: max similarity against stored embeddings.
        Kept simple for prototype stability.
        """
        query_emb = normalize_embedding(query_emb)

        best_score = -1.0
        valid_found = False

        for emb in emb_list:
            score = cosine_similarity(query_emb, emb)
            if score < -0.5:
                # invalid score due to shape mismatch or bad embedding
                continue

            valid_found = True
            if score > best_score:
                best_score = score

        if not valid_found:
            return None

        return float(best_score)

    def _best_match_in_type(self, query_emb, person_type):
        best_id = None
        best_score = -1.0

        for person_id, emb_list in self.embeddings.get(person_type, {}).items():
            if not emb_list:
                continue

            score = self._score_embedding_list(query_emb, emb_list)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_id = person_id

        return best_id, best_score

    def _collect_all_candidates(self, query_emb):
        candidates = []

        for person_type, people in self.embeddings.items():
            for person_id, emb_list in people.items():
                if not emb_list:
                    continue

                score = self._score_embedding_list(query_emb, emb_list)
                if score is None:
                    continue

                candidates.append({
                    "person_id": person_id,
                    "person_type": person_type,
                    "score": float(score),
                })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return candidates

    def _best_match_global(self, query_emb):
        candidates = self._collect_all_candidates(query_emb)

        if not candidates:
            return None, None, -1.0

        best = candidates[0]
        return best["person_id"], best["person_type"], best["score"]

    def recognize(self, query_emb):
        candidates = self._collect_all_candidates(query_emb)

        if not candidates:
            return {
                "person_id": None,
                "person_type": None,
                "score": None,
            }

        best = candidates[0]
        second = candidates[1] if len(candidates) > 1 else None

        best_score = best["score"]
        margin_ok = True

        if second is not None:
            margin = best_score - second["score"]
            margin_ok = margin >= RECOGNITION_MIN_MARGIN

        if best_score >= self.threshold and margin_ok:
            return {
                "person_id": best["person_id"],
                "person_type": best["person_type"],
                "score": float(best_score),
            }

        return {
            "person_id": None,
            "person_type": None,
            "score": None,
        }

    def recognize_top_k(self, query_emb, k=3):
        candidates = self._collect_all_candidates(query_emb)
        return candidates[:k]

    def has_embeddings(self, person_type=None):
        if person_type is None:
            return any(len(people) > 0 for people in self.embeddings.values())

        return len(self.embeddings.get(person_type, {})) > 0