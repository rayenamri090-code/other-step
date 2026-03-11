import time
from datetime import datetime

from config import VISIBLE_SESSION_TIMEOUT_SEC, ACCESS_SESSION_TIMEOUT_SEC, CAMERA_ID
from database import add_visible_session, open_access_session, close_access_session


class SessionService:
    def __init__(self, zone_name):
        self.zone_name = zone_name

    @staticmethod
    def _ts_to_str(ts: float) -> str:
        return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")

    def on_track_seen(self, track, person_id):
        now_ts = time.time()

        if track.get("visible_session_start_ts") is None:
            track["visible_session_start_ts"] = now_ts
            track["visible_session_start_str"] = self._ts_to_str(now_ts)

        track["last_seen_ts"] = now_ts

        if person_id and not track.get("access_session_open", False):
            open_access_session(person_id, self.zone_name)
            track["access_session_open"] = True

    def _finalize_visible_session(self, track, person_id):
        start_ts = track.get("visible_session_start_ts")
        end_ts = track.get("last_seen_ts")

        if start_ts is None or end_ts is None:
            return

        if end_ts < start_ts:
            end_ts = start_ts

        start_str = self._ts_to_str(start_ts)
        end_str = self._ts_to_str(end_ts)
        duration = max(0.0, end_ts - start_ts)

        add_visible_session(
            person_id=person_id,
            camera_id=CAMERA_ID,
            track_id=track["track_id"],
            start_time=start_str,
            end_time=end_str,
            duration_seconds=duration,
        )

    def _finalize_access_session(self, track, person_id):
        if track.get("access_session_open", False):
            close_access_session(person_id, self.zone_name)
            track["access_session_open"] = False

    def on_track_removed(self, track):
        person_id = track.get("identity")
        if not person_id:
            return

        self._finalize_visible_session(track, person_id)
        self._finalize_access_session(track, person_id)

    def should_timeout_visible(self, track, now_ts):
        last_seen = track.get("last_seen_ts")
        if last_seen is None:
            return False
        return (now_ts - last_seen) > VISIBLE_SESSION_TIMEOUT_SEC

    def should_timeout_access(self, track, now_ts):
        last_seen = track.get("last_seen_ts")
        if last_seen is None:
            return False
        return (now_ts - last_seen) > ACCESS_SESSION_TIMEOUT_SEC