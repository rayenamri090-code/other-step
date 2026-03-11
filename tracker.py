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

        area1 = w1 * h1
        area2 = w2 * h2
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
        score = (1.0 - dist_ratio) + (0.8 * iou)
        return score

    def update(self, detections):
        assigned = set()

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
                track["track_state"] = "stable" if track["seen_frames"] >= 3 else "new"
            else:
                track["missing_frames"] += 1
                track["updated"] = False
                track["match_score"] = 0.0

                if track["missing_frames"] > 0:
                    track["track_state"] = "lost"

        for i, det in enumerate(detections):
            if i in assigned:
                continue

            track_id = f"track_{self.next_id:04d}"
            self.next_id += 1

            self.tracks[track_id] = {
                "track_id": track_id,
                "bbox": det["bbox"],
                "prev_bbox": det["bbox"],
                "face_row": det["face_row"],
                "det_score": det["score"],
                "missing_frames": 0,
                "seen_frames": 1,
                "updated": True,
                "match_score": 1.0,
                "track_state": "new",

                # current decided identity
                "identity": None,
                "identity_type": None,
                "identity_score": None,

                # stabilization fields for next recognition upgrade
                "candidate_identity": None,
                "candidate_identity_type": None,
                "candidate_identity_score": None,
                "candidate_identity_hits": 0,
                "identity_locked": False,
                "identity_lock_score": None,

                # unknown handling
                "stable_unknown_frames": 0,
                "pending_alert_sent": False,

                # session/runtime bookkeeping
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