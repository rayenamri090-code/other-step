import time
from datetime import datetime

from config import (
    VISIBLE_SESSION_TIMEOUT_SEC,
    ACCESS_SESSION_TIMEOUT_SEC,
    CAMERA_ID,
)
from database import (
    add_visible_session,
    open_access_session,
    close_access_session,
)


class SessionService:
    def __init__(self, zone_id, zone_name):
        self.zone_id = zone_id
        self.zone_name = zone_name

    @staticmethod
    def _ts_to_str(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def _ensure_track_session_fields(self, track):
        track.setdefault("visible_session_start_ts", None)
        track.setdefault("visible_session_start_str", None)
        track.setdefault("last_seen_ts", None)

        # Number of visible session segments created within this current track lifecycle
        track.setdefault("visible_segment_count", 0)

        # Runtime/debug only, not authoritative DB analytics
        track.setdefault("total_visible_time_hint_sec", 0.0)

        # Access session state
        track.setdefault("access_session_open", False)

        # Runtime counters for live reporting/debug
        track.setdefault("access_granted_count", 0)
        track.setdefault("access_denied_count", 0)
        track.setdefault("access_alert_count", 0)

        # Last recorded access decision info
        track.setdefault("last_access_decision", None)
        track.setdefault("last_access_decision_ts", 0.0)

    def _finalize_visible_session(self, track, person_id, person_type, end_ts=None):
        self._ensure_track_session_fields(track)

        start_ts = track.get("visible_session_start_ts")
        last_seen_ts = track.get("last_seen_ts")

        if start_ts is None:
            return

        if end_ts is None:
            end_ts = last_seen_ts

        if end_ts is None:
            return

        if end_ts < start_ts:
            end_ts = start_ts

        duration = max(0.0, end_ts - start_ts)

        add_visible_session(
            person_id=person_id,
            person_type=person_type,
            camera_id=CAMERA_ID,
            zone_id=self.zone_id,
            track_id=track["track_id"],
            start_time=self._ts_to_str(start_ts),
            end_time=self._ts_to_str(end_ts),
            duration_seconds=duration,
            appearance_count=track.get("visible_segment_count", 1),
        )

        track["total_visible_time_hint_sec"] = track.get("total_visible_time_hint_sec", 0.0) + duration
        track["visible_session_start_ts"] = None
        track["visible_session_start_str"] = None
        track["visible_segment_count"] = 0

    def _finalize_access_session(self, track, person_id):
        self._ensure_track_session_fields(track)

        if track.get("access_session_open", False):
            close_access_session(person_id, self.zone_id)
            track["access_session_open"] = False

    def on_track_seen(self, track, person_id):
        self._ensure_track_session_fields(track)

        person_type = track.get("identity_type")
        now_ts = time.time()
        last_seen_ts = track.get("last_seen_ts")

        # Split visible session if person was absent too long
        if (
            last_seen_ts is not None
            and track.get("visible_session_start_ts") is not None
            and (now_ts - last_seen_ts) > VISIBLE_SESSION_TIMEOUT_SEC
        ):
            self._finalize_visible_session(
                track=track,
                person_id=person_id,
                person_type=person_type,
                end_ts=last_seen_ts,
            )

        # Close stale access session if gap is too long
        if (
            last_seen_ts is not None
            and track.get("access_session_open", False)
            and (now_ts - last_seen_ts) > ACCESS_SESSION_TIMEOUT_SEC
        ):
            self._finalize_access_session(track, person_id)

        # Start a new visible session if needed
        if track.get("visible_session_start_ts") is None:
            track["visible_session_start_ts"] = now_ts
            track["visible_session_start_str"] = self._ts_to_str(now_ts)
            track["visible_segment_count"] = track.get("visible_segment_count", 0) + 1

        # Start a new access session if needed
        if person_id and not track.get("access_session_open", False):
            open_access_session(
                person_id=person_id,
                person_type=person_type,
                zone_id=self.zone_id,
                zone_name=self.zone_name,
            )
            track["access_session_open"] = True

        track["last_seen_ts"] = now_ts

    def record_access_decision(self, track, decision: str):
        """
        Lightweight runtime counters for live reporting/debug.
        Intended to be called only when the caller has already decided
        this access decision is worth recording.
        """
        self._ensure_track_session_fields(track)

        now_ts = time.time()
        track["last_access_decision"] = decision
        track["last_access_decision_ts"] = now_ts

        if decision == "AUTHORIZED":
            track["access_granted_count"] = track.get("access_granted_count", 0) + 1
        elif decision == "ALERT_PENDING":
            track["access_alert_count"] = track.get("access_alert_count", 0) + 1
        else:
            track["access_denied_count"] = track.get("access_denied_count", 0) + 1

    def on_track_removed(self, track):
        self._ensure_track_session_fields(track)

        person_id = track.get("identity")
        person_type = track.get("identity_type")

        if not person_id:
            return

        self._finalize_visible_session(
            track=track,
            person_id=person_id,
            person_type=person_type,
            end_ts=track.get("last_seen_ts"),
        )
        self._finalize_access_session(track, person_id)

    def should_timeout_visible(self, track, now_ts):
        self._ensure_track_session_fields(track)

        last_seen = track.get("last_seen_ts")
        if last_seen is None:
            return False
        return (now_ts - last_seen) > VISIBLE_SESSION_TIMEOUT_SEC

    def should_timeout_access(self, track, now_ts):
        self._ensure_track_session_fields(track)

        last_seen = track.get("last_seen_ts")
        if last_seen is None:
            return False
        return (now_ts - last_seen) > ACCESS_SESSION_TIMEOUT_SEC