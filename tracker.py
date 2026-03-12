from math import hypot
from time import time

from config import (
    TRACK_MATCH_DISTANCE_PX,
    TRACK_MAX_MISSING_FRAMES,
    TRACK_MIN_STABLE_FRAMES,
)


class MultiFaceTracker:
    def __init__(self):
        self.next_id = 1
        self.tracks = {}

    @staticmethod
    def _center(bbox):
        x, y, w, h = bbox
        return (x + w / 2.0, y + h / 2.0)

    @staticmethod
    def _distance(b1, b2):
        c1 = MultiFaceTracker._center(b1)
        c2 = MultiFaceTracker._center(b2)
        return hypot(c1[0] - c2[0], c1[1] - c2[1])

    @staticmethod
    def _iou(b1, b2):
        x1, y1, w1, h1 = b1
        x2, y2, w2, h2 = b2

        xa = max(x1, x2)
        ya = max(y1, y2)
        xb = min(x1 + w1, x2 + w2)
        yb = min(y1 + h1, y2 + h2)

        inter_w = max(0, xb - xa)
        inter_h = max(0, yb - ya)
        inter_area = inter_w * inter_h

        area1 = max(0, w1) * max(0, h1)
        area2 = max(0, w2) * max(0, h2)
        union = area1 + area2 - inter_area

        if union <= 0:
            return 0.0

        return inter_area / union

    def _match_score(self, track_bbox, det_bbox):
        dist = self._distance(track_bbox, det_bbox)
        iou = self._iou(track_bbox, det_bbox)

        if dist > TRACK_MATCH_DISTANCE_PX:
            return None

        dist_ratio = min(dist / TRACK_MATCH_DISTANCE_PX, 1.0)

        # distance is the main constraint, IoU helps stabilization
        score = (1.0 - dist_ratio) + (0.8 * iou)
        return score

    def _new_track(self, det):
        now_ts = time()
        track_id = f"track_{self.next_id:04d}"
        self.next_id += 1

        return {
            "track_id": track_id,

            # geometry / detection
            "bbox": det["bbox"],
            "prev_bbox": det["bbox"],
            "face_row": det["face_row"],
            "det_score": det["score"],
            "match_score": 1.0,

            # tracking lifecycle
            "missing_frames": 0,
            "seen_frames": 1,
            "updated": True,
            "track_state": "new",

            # identity decision
            "identity": None,
            "identity_type": None,
            "identity_score": None,

            # candidate identity stabilization
            "candidate_identity": None,
            "candidate_identity_type": None,
            "candidate_identity_score": None,
            "candidate_identity_hits": 0,
            "identity_locked": False,
            "identity_lock_score": None,

            # unknown handling
            "stable_unknown_frames": 0,
            "pending_alert_sent": False,
            "unknown_created": False,

            # timing / session bookkeeping
            "created_ts": now_ts,
            "first_seen_ts": now_ts,
            "last_seen_ts": now_ts,
            "last_recognition_ts": 0.0,
            "last_logged_ts": 0.0,

            # visible session bookkeeping
            "visible_session_start_ts": None,
            "visible_session_start_str": None,
            "appearance_count": 0,
            "total_visible_time_hint_sec": 0.0,

            # access session bookkeeping
            "access_session_open": False,
        }

    def update(self, detections):
        assigned = set()
        now_ts = time()

        # mark all tracks as not updated by default for this cycle
        for track in self.tracks.values():
            track["updated"] = False

        # match existing tracks to current detections
        for track_id, track in list(self.tracks.items()):
            best_idx = None
            best_score = -1.0

            for i, det in enumerate(detections):
                if i in assigned:
                    continue

                score = self._match_score(track["bbox"], det["bbox"])
                if score is None:
                    continue

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                det = detections[best_idx]
                assigned.add(best_idx)

                track["prev_bbox"] = track["bbox"]
                track["bbox"] = det["bbox"]
                track["face_row"] = det["face_row"]
                track["det_score"] = det["score"]
                track["missing_frames"] = 0
                track["seen_frames"] += 1
                track["updated"] = True
                track["match_score"] = best_score
                track["last_seen_ts"] = now_ts

                if track["seen_frames"] >= TRACK_MIN_STABLE_FRAMES:
                    track["track_state"] = "stable"
                else:
                    track["track_state"] = "new"
            else:
                track["missing_frames"] += 1
                track["updated"] = False
                track["match_score"] = 0.0

                if track["missing_frames"] > 0:
                    track["track_state"] = "lost"

        # create tracks for unmatched detections
        for i, det in enumerate(detections):
            if i in assigned:
                continue

            track = self._new_track(det)
            self.tracks[track["track_id"]] = track

        # remove expired tracks
        removed = []
        for track_id, track in list(self.tracks.items()):
            if track["missing_frames"] > TRACK_MAX_MISSING_FRAMES:
                track["removed_ts"] = now_ts
                removed.append(dict(track))
                del self.tracks[track_id]

        return self.tracks, removed