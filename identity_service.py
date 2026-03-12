import json

from database import (
    create_unknown_identity,
    add_embedding,
    add_alert,
)
from config import (
    UNKNOWN_EMBEDDINGS_TO_SAVE,
    UNKNOWN_STABLE_FRAMES_REQUIRED,
    UNKNOWN_REUSE_THRESHOLD,
    IDENTITY_CONFIRM_FRAMES,
    IDENTITY_LOCK_MIN_SCORE,
    IDENTITY_UNLOCK_STRONGER_SCORE_DIFF,
)


class IdentityService:
    def __init__(self, recognizer, camera_id, zone_id=None):
        self.recognizer = recognizer
        self.camera_id = camera_id
        self.zone_id = zone_id

        # config-driven stabilization settings
        self.min_candidate_hits_to_confirm = IDENTITY_CONFIRM_FRAMES
        self.min_lock_score = IDENTITY_LOCK_MIN_SCORE
        self.stronger_score_diff = IDENTITY_UNLOCK_STRONGER_SCORE_DIFF

    # =========================================================
    # Internal helpers
    # =========================================================

    def _reset_candidate(self, track):
        track["candidate_identity"] = None
        track["candidate_identity_type"] = None
        track["candidate_identity_score"] = None
        track["candidate_identity_hits"] = 0

    def _clear_runtime_flags(self, track):
        track["identity_just_confirmed"] = False
        track["identity_just_switched"] = False
        track["unknown_just_created"] = False
        track["unknown_just_reused"] = False

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

    def _set_confirmed_identity(self, track, person_id, person_type, score):
        previous_identity = track.get("identity")
        previous_type = track.get("identity_type")

        track["identity"] = person_id
        track["identity_type"] = person_type
        track["identity_score"] = score
        track["stable_unknown_frames"] = 0

        track["identity_locked"] = score is not None and score >= self.min_lock_score
        track["identity_lock_score"] = score if track["identity_locked"] else None

        self._reset_candidate(track)

        track["identity_just_confirmed"] = True
        track["identity_just_switched"] = (
            previous_identity is not None
            and (previous_identity != person_id or previous_type != person_type)
        )

    def _promote_candidate_if_ready(self, track):
        candidate_id = track.get("candidate_identity")
        candidate_type = track.get("candidate_identity_type")
        candidate_score = track.get("candidate_identity_score")
        candidate_hits = track.get("candidate_identity_hits", 0)

        if candidate_id is None:
            return False

        if candidate_hits >= self.min_candidate_hits_to_confirm:
            self._set_confirmed_identity(track, candidate_id, candidate_type, candidate_score)
            return True

        return False

    def _should_switch_locked_identity(self, track, match):
        current_score = track.get("identity_score")
        new_score = match.get("score")

        if current_score is None or new_score is None:
            return False

        candidate_hits = track.get("candidate_identity_hits", 0)
        if candidate_hits < self.min_candidate_hits_to_confirm + 1:
            return False

        # only switch if contradiction is consistently stronger
        if new_score >= self.min_lock_score and (new_score - current_score) >= self.stronger_score_diff:
            return True

        return False

    # =========================================================
    # Main recognition stabilization
    # =========================================================

    def process_track_identity(self, track, embedding):
        self._clear_runtime_flags(track)

        match = self.recognizer.recognize(embedding)

        # -------------------------------------------------
        # 1. Recognized
        # -------------------------------------------------
        if match["person_id"] is not None:
            matched_id = match["person_id"]
            matched_type = match["person_type"]
            matched_score = match["score"]

            # A confirmed locked identity should resist noise
            if track.get("identity_locked", False):
                if self._same_identity(track, match):
                    track["identity_score"] = matched_score
                    track["identity_lock_score"] = matched_score
                    track["stable_unknown_frames"] = 0
                    self._reset_candidate(track)
                    return track

                # build contradiction candidate
                if self._same_candidate(track, match):
                    track["candidate_identity_hits"] = track.get("candidate_identity_hits", 0) + 1
                    track["candidate_identity_score"] = matched_score
                else:
                    track["candidate_identity"] = matched_id
                    track["candidate_identity_type"] = matched_type
                    track["candidate_identity_score"] = matched_score
                    track["candidate_identity_hits"] = 1

                if self._should_switch_locked_identity(track, match):
                    self._set_confirmed_identity(track, matched_id, matched_type, matched_score)

                return track

            # If current confirmed identity is seen again, reinforce it
            if self._same_identity(track, match):
                track["identity_score"] = matched_score
                track["stable_unknown_frames"] = 0
                self._reset_candidate(track)

                if matched_score >= self.min_lock_score:
                    track["identity_locked"] = True
                    track["identity_lock_score"] = matched_score

                return track

            # No confirmed identity yet or new contradiction on unlocked track
            if self._same_candidate(track, match):
                track["candidate_identity_hits"] = track.get("candidate_identity_hits", 0) + 1
                track["candidate_identity_score"] = matched_score
            else:
                track["candidate_identity"] = matched_id
                track["candidate_identity_type"] = matched_type
                track["candidate_identity_score"] = matched_score
                track["candidate_identity_hits"] = 1

            if self._promote_candidate_if_ready(track):
                return track

            return track

        # -------------------------------------------------
        # 2. Not recognized
        # -------------------------------------------------
        # If already confirmed, keep identity instead of dropping instantly
        if track.get("identity") is not None and track.get("identity_type") != "unknown_candidate":
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

    # =========================================================
    # Unknown handling
    # =========================================================

    def _try_reuse_existing_unknown(self, embedding):
        person_id, score = self.recognizer._best_match_in_type(embedding, "unknown")

        if person_id is not None and score >= UNKNOWN_REUSE_THRESHOLD:
            return {
                "person_id": person_id,
                "score": score,
            }

        return None

    def convert_unknown_candidate_if_stable(self, track, embedding):
        # clear runtime flags handled in process_track_identity already

        # if already resolved to a proper identity, do nothing
        if track.get("identity") is not None and track.get("identity_type") != "unknown_candidate":
            return None

        # do not create unknown too early
        if track.get("stable_unknown_frames", 0) < UNKNOWN_STABLE_FRAMES_REQUIRED:
            return None

        # avoid duplicate creation for same track
        if track.get("unknown_created", False):
            return track.get("identity")

        reused = self._try_reuse_existing_unknown(embedding)
        if reused is not None:
            self._set_confirmed_identity(track, reused["person_id"], "unknown", reused["score"])
            track["unknown_just_reused"] = True
            return reused["person_id"]

        person_id = create_unknown_identity()
        emb_json = json.dumps(embedding.astype(float).tolist())

        # save several embeddings to make later reuse possible
        for _ in range(UNKNOWN_EMBEDDINGS_TO_SAVE):
            add_embedding(person_id, emb_json, 1.0)

        add_alert(
            camera_id=self.camera_id,
            track_id=track["track_id"],
            person_id=person_id,
            alert_type="UNKNOWN_PERSON_DETECTED",
            notes="Auto-created unknown identity pending validation",
            status="open",
            zone_id=self.zone_id,
            person_type="unknown",
        )

        self.recognizer.reload_embeddings()

        self._set_confirmed_identity(track, person_id, "unknown", 0.0)
        track["pending_alert_sent"] = True
        track["unknown_created"] = True
        track["unknown_just_created"] = True

        return person_id