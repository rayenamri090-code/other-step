import json

from database import (
    create_unknown_identity,
    add_embedding,
    add_alert,
)
from config import UNKNOWN_EMBEDDINGS_TO_SAVE


class IdentityService:
    def __init__(self, recognizer, camera_id):
        self.recognizer = recognizer
        self.camera_id = camera_id

    def process_track_identity(self, track, embedding):
        match = self.recognizer.recognize(embedding)

        if match["person_id"] is not None:
            track["identity"] = match["person_id"]
            track["identity_type"] = match["person_type"]
            track["identity_score"] = match["score"]
            track["stable_unknown_frames"] = 0
            return track

        track["stable_unknown_frames"] = track.get("stable_unknown_frames", 0) + 1

        track["identity"] = None
        track["identity_type"] = "unknown_candidate"
        track["identity_score"] = None

        return track

    def _try_reuse_existing_unknown(self, embedding):
        person_id, score = self.recognizer._best_match(embedding, "unknown")

        if person_id is not None and score >= self.recognizer.threshold:
            return {
                "person_id": person_id,
                "score": score,
            }

        return None

    def convert_unknown_candidate_if_stable(self, track, embedding):
        if track.get("stable_unknown_frames", 0) < UNKNOWN_EMBEDDINGS_TO_SAVE:
            return None

        reused = self._try_reuse_existing_unknown(embedding)
        if reused is not None:
            track["identity"] = reused["person_id"]
            track["identity_type"] = "unknown"
            track["identity_score"] = reused["score"]
            return reused["person_id"]

        person_id = create_unknown_identity()
        emb_json = json.dumps(embedding.astype(float).tolist())

        for _ in range(UNKNOWN_EMBEDDINGS_TO_SAVE):
            add_embedding(person_id, emb_json, 1.0)

        add_alert(
            camera_id=self.camera_id,
            track_id=track["track_id"],
            person_id=person_id,
            alert_type="UNKNOWN_PERSON_DETECTED",
            notes="Auto-created unknown identity pending validation",
            status="open",
        )

        self.recognizer.reload_embeddings()

        track["identity"] = person_id
        track["identity_type"] = "unknown"
        track["identity_score"] = 0.0
        track["pending_alert_sent"] = True
        return person_id