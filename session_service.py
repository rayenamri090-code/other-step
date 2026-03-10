import time
from datetime import datetime
from config import VISIBLE_SESSION_TIMEOUT_SEC, ACCESS_SESSION_TIMEOUT_SEC, CAMERA_ID
from database import add_visible_session, open_access_session, close_access_session


class SessionService:
    def __init__(self, zone_name):
        self.zone_name = zone_name

    def on_track_seen(self, track, person_id):
        now_ts = time.time()

        if track["visible_session_start_ts"] is None:
            track["visible_session_start_ts"] = now_ts
            track["visible_session_start_str"] = datetime.fromtimestamp(now_ts).strftime("%Y-%m-%d %H:%M:%S")

        track["last_seen_ts"] = now_ts

        if person_id and not track["access_session_open"]:
            open_access_session(person_id, self.zone_name)
            track["access_session_open"] = True

    def on_track_removed(self, track):
        person_id = track.get("identity")
        if not person_id:
            return

        start_ts = track.get("visible_session_start_ts")
        end_ts = track.get("last_seen_ts")
        if start_ts and end_ts:
            start_str = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M:%S")
            end_str = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M:%S")
            duration = max(0.0, end_ts - start_ts)
            add_visible_session(person_id, CAMERA_ID, track["track_id"], start_str, end_str, duration)

        if track.get("access_session_open"):
            close_access_session(person_id, self.zone_name)
            track["access_session_open"] = False

    def should_timeout_visible(self, track, now_ts):
        last_seen = track.get("last_seen_ts")
        return last_seen is not None and (now_ts - last_seen) > VISIBLE_SESSION_TIMEOUT_SEC

    def should_timeout_access(self, track, now_ts):
        last_seen = track.get("last_seen_ts")
        return last_seen is not None and (now_ts - last_seen) > ACCESS_SESSION_TIMEOUT_SEC