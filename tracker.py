from math import hypot
from config import TRACK_MATCH_DISTANCE_PX, TRACK_MAX_MISSING_FRAMES


class MultiFaceTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)

    @staticmethod
    def _distance(b1, b2):
        c1 = MultiFaceTracker._center(b1)
        c2 = MultiFaceTracker._center(b2)
        return hypot(c1[0] - c2[0], c1[1] - c2[1])

    def update(self, detections):
        assigned = set()

        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_dist = 1e9

            for i, det in enumerate(detections):
                if i in assigned:
                    continue

                dist = self._distance(track["bbox"], det["bbox"])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

            if best_idx is not None and best_dist <= TRACK_MATCH_DISTANCE_PX:
                det = detections[best_idx]
                assigned.add(best_idx)

                track["bbox"] = det["bbox"]
                track["face_row"] = det["face_row"]
                track["det_score"] = det["score"]
                track["missing_frames"] = 0
                track["seen_frames"] += 1
                track["updated"] = True
            else:
                track["missing_frames"] += 1
                track["updated"] = False

        for i, det in enumerate(detections):
            if i in assigned:
                continue

            track_id = f"track_{self.next_id:04d}"
            self.next_id += 1

            self.tracks[track_id] = {
                "track_id": track_id,
                "bbox": det["bbox"],
                "face_row": det["face_row"],
                "det_score": det["score"],
                "missing_frames": 0,
                "seen_frames": 1,
                "updated": True,
                "identity": None,
                "identity_type": None,
                "identity_score": None,
                "stable_unknown_frames": 0,
                "pending_alert_sent": False,
                "visible_session_start_ts": None,
                "visible_session_start_str": None,
                "last_seen_ts": None,
                "last_logged_ts": 0.0,
                "access_session_open": False,
            }

        removed = []
        for track_id, track in list(self.tracks.items()):
            if track["missing_frames"] > TRACK_MAX_MISSING_FRAMES:
                removed.append(track)
                del self.tracks[track_id]

        return self.tracks, removed