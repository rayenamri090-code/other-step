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

        # recognition stabilization settings
        self.min_candidate_hits_to_confirm = 3
        self.min_lock_score = 0.60
        self.min_reconfirm_score = 0.50

    def _reset_candidate(self, track):
        track["candidate_identity"] = None
        track["candidate_identity_type"] = None
        track["candidate_identity_score"] = None
        track["candidate_identity_hits"] = 0

    def _set_confirmed_identity(self, track, person_id, person_type, score):
        track["identity"] = person_id
        track["identity_type"] = person_type
        track["identity_score"] = score
        track["stable_unknown_frames"] = 0

        track["identity_locked"] = score is not None and score >= self.min_lock_score
        track["identity_lock_score"] = score

        self._reset_candidate(track)

    def _same_identity(self, track, match):
        return (
            track.get("identity") == match.get("person_id")
            and track.get("identity_type") == match.get("person_type")
        )

    def _same_candidate(self, track, match):
        return (
            track.get("candidate_identity") == match.get("person_id")
            and track.get("candidate_identity_type") == match.get("person_type")
        )

    def process_track_identity(self, track, embedding):
        match = self.recognizer.recognize(embedding)

        # -------------------------------------------------
        # 1. If recognized
        # -------------------------------------------------
        if match["person_id"] is not None:
            matched_id = match["person_id"]
            matched_type = match["person_type"]
            matched_score = match["score"]

            # If identity already locked, keep it unless same identity is seen again
            if track.get("identity_locked", False):
                if self._same_identity(track, match):
                    track["identity_score"] = matched_score
                    track["identity_lock_score"] = matched_score
                    track["stable_unknown_frames"] = 0
                    self._reset_candidate(track)
                    return track

                # locked identity exists, but new match is different
                # do not switch immediately, require stronger candidate buildup
                if self._same_candidate(track, match):
                    track["candidate_identity_hits"] = track.get("candidate_identity_hits", 0) + 1
                    track["candidate_identity_score"] = matched_score
                else:
                    track["candidate_identity"] = matched_id
                    track["candidate_identity_type"] = matched_type
                    track["candidate_identity_score"] = matched_score
                    track["candidate_identity_hits"] = 1

                # only unlock/switch if repeated strong contradiction exists
                if (
                    track.get("candidate_identity_hits", 0) >= self.min_candidate_hits_to_confirm + 1
                    and matched_score >= self.min_lock_score
                ):
                    self._set_confirmed_identity(track, matched_id, matched_type, matched_score)

                return track

            # If current confirmed identity is same, reinforce it
            if self._same_identity(track, match):
                track["identity_score"] = matched_score
                track["stable_unknown_frames"] = 0
                self._reset_candidate(track)

                if matched_score >= self.min_lock_score:
                    track["identity_locked"] = True
                    track["identity_lock_score"] = matched_score

                return track

            # Build candidate identity over several hits
            if self._same_candidate(track, match):
                track["candidate_identity_hits"] = track.get("candidate_identity_hits", 0) + 1
                track["candidate_identity_score"] = matched_score
            else:
                track["candidate_identity"] = matched_id
                track["candidate_identity_type"] = matched_type
                track["candidate_identity_score"] = matched_score
                track["candidate_identity_hits"] = 1

            # Confirm only after repeated consistent recognition
            if track.get("candidate_identity_hits", 0) >= self.min_candidate_hits_to_confirm:
                self._set_confirmed_identity(track, matched_id, matched_type, matched_score)
                return track

            # keep unresolved until enough confirmations
            return track

        # -------------------------------------------------
        # 2. If not recognized
        # -------------------------------------------------
        # keep already-confirmed identity for a while instead of dropping instantly
        if track.get("identity") is not None:
            track["stable_unknown_frames"] = 0
            self._reset_candidate(track)
            return track

        track["stable_unknown_frames"] = track.get("stable_unknown_frames", 0) + 1

        track["identity"] = None
        track["identity_type"] = "unknown_candidate"
        track["identity_score"] = None
        track["identity_locked"] = False
        track["identity_lock_score"] = None

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
        # if already resolved, do nothing
        if track.get("identity") is not None and track.get("identity_type") != "unknown_candidate":
            return None

        if track.get("stable_unknown_frames", 0) < UNKNOWN_EMBEDDINGS_TO_SAVE:
            return None

        reused = self._try_reuse_existing_unknown(embedding)
        if reused is not None:
            self._set_confirmed_identity(track, reused["person_id"], "unknown", reused["score"])
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

        self._set_confirmed_identity(track, person_id, "unknown", 0.0)
        track["pending_alert_sent"] = True
        return person_id