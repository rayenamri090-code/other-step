import json
import time

import numpy as np

from database import (
    create_unknown_identity,
    add_embedding,
    add_alert,
)
from config import (
    UNKNOWN_STABLE_FRAMES_REQUIRED,
    UNKNOWN_REUSE_THRESHOLD,
    IDENTITY_CONFIRM_FRAMES,
    IDENTITY_LOCK_MIN_SCORE,
    IDENTITY_UNLOCK_STRONGER_SCORE_DIFF,
)


# =========================================================
# Unknown enrichment tuning
# These can later be moved to config.py if you want.
# =========================================================
UNKNOWN_ENRICH_MAX_EXTRA_EMBEDDINGS_PER_TRACK = 3
UNKNOWN_ENRICH_COOLDOWN_SEC = 1.5
UNKNOWN_ENRICH_MIN_DISTANCE = 0.08

# =========================================================
# Stronger unknown reuse / creation stabilization
# =========================================================
UNKNOWN_REUSE_CONFIRM_FRAMES = 2
UNKNOWN_CREATE_CONFIRM_FRAMES = 3


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

    def _ensure_track_identity_fields(self, track):
        track.setdefault("identity", None)
        track.setdefault("identity_type", None)
        track.setdefault("identity_score", None)

        track.setdefault("identity_locked", False)
        track.setdefault("identity_lock_score", None)

        track.setdefault("stable_unknown_frames", 0)

        track.setdefault("candidate_identity", None)
        track.setdefault("candidate_identity_type", None)
        track.setdefault("candidate_identity_score", None)
        track.setdefault("candidate_identity_hits", 0)

        track.setdefault("unknown_reuse_candidate_id", None)
        track.setdefault("unknown_reuse_candidate_score", None)
        track.setdefault("unknown_reuse_candidate_hits", 0)
        track.setdefault("unknown_no_reuse_frames", 0)

        # Means this track has already been resolved once into a concrete unknown identity,
        # whether by reuse or by new creation. Prevents repeated unknown creation/reuse
        # for the same persistent track.
        track.setdefault("unknown_resolved_once", False)

        track.setdefault("unknown_extra_embeddings_saved", 0)
        track.setdefault("unknown_last_embedding_save_ts", 0.0)
        track.setdefault("unknown_last_saved_embedding", None)

        track.setdefault("pending_alert_sent", False)

        self._clear_runtime_flags(track)

    def _reset_candidate(self, track):
        track["candidate_identity"] = None
        track["candidate_identity_type"] = None
        track["candidate_identity_score"] = None
        track["candidate_identity_hits"] = 0

    def _reset_unknown_resolution_state(self, track):
        track["unknown_reuse_candidate_id"] = None
        track["unknown_reuse_candidate_score"] = None
        track["unknown_reuse_candidate_hits"] = 0
        track["unknown_no_reuse_frames"] = 0

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
        self._reset_unknown_resolution_state(track)

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

        # Only switch if contradiction is consistently stronger
        if new_score >= self.min_lock_score and (new_score - current_score) >= self.stronger_score_diff:
            return True

        return False

    def _embedding_distance(self, emb1, emb2):
        if emb1 is None or emb2 is None:
            return 1.0

        v1 = np.asarray(emb1, dtype=np.float32).flatten()
        v2 = np.asarray(emb2, dtype=np.float32).flatten()

        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)

        if n1 == 0.0 or n2 == 0.0:
            return 1.0

        sim = float(np.dot(v1, v2) / (n1 * n2))
        sim = max(-1.0, min(1.0, sim))
        return 1.0 - sim

    def _maybe_enrich_unknown_identity(self, track, embedding):
        """
        Save a few additional varied embeddings for an already confirmed unknown.
        This improves future reuse stability.
        """
        if track.get("identity_type") != "unknown":
            return False

        person_id = track.get("identity")
        if not person_id:
            return False

        saved_count = track.get("unknown_extra_embeddings_saved", 0)
        if saved_count >= UNKNOWN_ENRICH_MAX_EXTRA_EMBEDDINGS_PER_TRACK:
            return False

        now_ts = time.time()
        last_save_ts = track.get("unknown_last_embedding_save_ts", 0.0)
        if (now_ts - last_save_ts) < UNKNOWN_ENRICH_COOLDOWN_SEC:
            return False

        last_saved_embedding = track.get("unknown_last_saved_embedding")
        distance = self._embedding_distance(embedding, last_saved_embedding)

        if last_saved_embedding is not None and distance < UNKNOWN_ENRICH_MIN_DISTANCE:
            return False

        emb_json = json.dumps(np.asarray(embedding, dtype=float).tolist())
        confidence = track.get("identity_score")
        if confidence is None:
            confidence = 1.0

        add_embedding(person_id, emb_json, float(confidence))
        self.recognizer.reload_embeddings()

        track["unknown_extra_embeddings_saved"] = saved_count + 1
        track["unknown_last_embedding_save_ts"] = now_ts
        track["unknown_last_saved_embedding"] = np.asarray(embedding, dtype=np.float32).copy()

        return True

    # =========================================================
    # Main recognition stabilization
    # =========================================================

    def process_track_identity(self, track, embedding):
        self._ensure_track_identity_fields(track)
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
                    self._reset_unknown_resolution_state(track)

                    if matched_type == "unknown":
                        self._maybe_enrich_unknown_identity(track, embedding)

                    return track

                # Build contradiction candidate
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
                self._reset_unknown_resolution_state(track)

                if matched_score >= self.min_lock_score:
                    track["identity_locked"] = True
                    track["identity_lock_score"] = matched_score

                if matched_type == "unknown":
                    self._maybe_enrich_unknown_identity(track, embedding)

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
                if track.get("identity_type") == "unknown":
                    self._maybe_enrich_unknown_identity(track, embedding)
                return track

            return track

        # -------------------------------------------------
        # 2. Not recognized
        # -------------------------------------------------
        # If already confirmed, keep identity instead of dropping instantly
        if track.get("identity") is not None and track.get("identity_type") != "unknown_candidate":
            track["stable_unknown_frames"] = 0
            self._reset_candidate(track)
            self._reset_unknown_resolution_state(track)
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

    def _update_unknown_reuse_candidate(self, track, reused):
        candidate_id = track.get("unknown_reuse_candidate_id")

        if reused is None:
            track["unknown_reuse_candidate_id"] = None
            track["unknown_reuse_candidate_score"] = None
            track["unknown_reuse_candidate_hits"] = 0
            track["unknown_no_reuse_frames"] = track.get("unknown_no_reuse_frames", 0) + 1
            return

        if candidate_id == reused["person_id"]:
            track["unknown_reuse_candidate_hits"] = track.get("unknown_reuse_candidate_hits", 0) + 1
            track["unknown_reuse_candidate_score"] = reused["score"]
        else:
            track["unknown_reuse_candidate_id"] = reused["person_id"]
            track["unknown_reuse_candidate_score"] = reused["score"]
            track["unknown_reuse_candidate_hits"] = 1

        track["unknown_no_reuse_frames"] = 0

    def convert_unknown_candidate_if_stable(self, track, embedding):
        self._ensure_track_identity_fields(track)

        # If already resolved to a proper identity, do nothing
        if track.get("identity") is not None and track.get("identity_type") != "unknown_candidate":
            return None

        # Do not create/reuse unknown too early
        if track.get("stable_unknown_frames", 0) < UNKNOWN_STABLE_FRAMES_REQUIRED:
            return None

        # Avoid duplicate resolution for same track
        if track.get("unknown_resolved_once", False):
            return track.get("identity")

        reused = self._try_reuse_existing_unknown(embedding)
        self._update_unknown_reuse_candidate(track, reused)

        # -------------------------------------------------
        # 1. Strong reuse path
        # -------------------------------------------------
        reuse_candidate_id = track.get("unknown_reuse_candidate_id")
        reuse_candidate_hits = track.get("unknown_reuse_candidate_hits", 0)
        reuse_candidate_score = track.get("unknown_reuse_candidate_score")

        if reuse_candidate_id and reuse_candidate_hits >= UNKNOWN_REUSE_CONFIRM_FRAMES:
            self._set_confirmed_identity(track, reuse_candidate_id, "unknown", reuse_candidate_score)
            track["unknown_just_reused"] = True

            track["unknown_resolved_once"] = True
            track["unknown_extra_embeddings_saved"] = 0
            track["unknown_last_embedding_save_ts"] = time.time()
            track["unknown_last_saved_embedding"] = np.asarray(embedding, dtype=np.float32).copy()

            return reuse_candidate_id

        # -------------------------------------------------
        # 2. Create new unknown only after repeated no-reuse evidence
        # -------------------------------------------------
        no_reuse_frames = track.get("unknown_no_reuse_frames", 0)
        if no_reuse_frames < UNKNOWN_CREATE_CONFIRM_FRAMES:
            return None

        person_id = create_unknown_identity()
        emb_json = json.dumps(np.asarray(embedding, dtype=float).tolist())

        # Save one initial embedding.
        # Additional varied embeddings may be added later while the same track persists.
        add_embedding(person_id, emb_json, 1.0)

        # Identity-layer alert: new unknown entity created in the database.
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

        # Synthetic self-confirmation score for newly created unknown identity
        self._set_confirmed_identity(track, person_id, "unknown", 1.0)
        track["pending_alert_sent"] = True
        track["unknown_resolved_once"] = True
        track["unknown_just_created"] = True

        track["unknown_extra_embeddings_saved"] = 0
        track["unknown_last_embedding_save_ts"] = time.time()
        track["unknown_last_saved_embedding"] = np.asarray(embedding, dtype=np.float32).copy()

        return person_id