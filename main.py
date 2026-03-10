import time
import json
import cv2
from datetime import datetime

from config import WINDOW_NAME, CAMERA_ID, LOG_COOLDOWN_SEC
from database import (
    init_db,
    create_demo_seed,
    get_camera_zone,
    add_access_event,
    add_alert,
    update_last_seen,
)
from camera_source import CameraSource
from detector import FaceDetector
from tracker import MultiFaceTracker
from recognizer import FaceRecognizer
from identity_service import IdentityService
from authorization_service import AuthorizationService
from session_service import SessionService
from mqtt_service import MQTTService


def color_for_state(identity_type, decision):
    if decision == "AUTHORIZED":
        return (0, 255, 0)
    if identity_type == "unknown":
        return (0, 0, 255)
    if decision == "ALERT_PENDING":
        return (0, 165, 255)
    return (0, 255, 255)


def draw_track(frame, track, decision=None):
    x, y, w, h = track["bbox"]
    identity = track.get("identity") or track["track_id"]
    identity_type = track.get("identity_type") or "unresolved"
    score = track.get("identity_score")

    color = color_for_state(identity_type, decision or "UNKNOWN")
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

    top = f"{identity} [{identity_type}]"
    if score is not None:
        top += f" {score:.3f}"

    cv2.putText(frame, top, (x, y - 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

    if decision:
        cv2.putText(frame, decision, (x, y - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)


def main():
    init_db()
    create_demo_seed()

    zone_info = get_camera_zone(CAMERA_ID)
    zone_name = zone_info["zone_name"]

    camera = CameraSource()
    detector = FaceDetector()
    tracker = MultiFaceTracker()
    recognizer = FaceRecognizer()
    identity_service = IdentityService(recognizer, CAMERA_ID)
    authorization_service = AuthorizationService()
    session_service = SessionService(zone_name)
    mqtt_service = MQTTService()

    mqtt_service.connect()
    camera.open()

    print("[INFO] Enterprise prototype started")
    print(f"[INFO] Camera ID: {CAMERA_ID}")
    print(f"[INFO] Zone: {zone_name}")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                break

            detections = detector.detect_all(frame)
            live_tracks, removed_tracks = tracker.update(detections)
            now_ts = time.time()

            for removed in removed_tracks:
                session_service.on_track_removed(removed)

            cv2.putText(
                frame,
                f"Cam:{CAMERA_ID} Zone:{zone_name} Faces:{len(detections)} Tracks:{len(live_tracks)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            for track_id, track in live_tracks.items():
                if not track["updated"]:
                    draw_track(frame, track)
                    continue

                try:
                    embedding = recognizer.extract_embedding(frame, track["face_row"])
                except Exception:
                    draw_track(frame, track)
                    continue

                identity_service.process_track_identity(track, embedding)

                if track["identity_type"] == "unknown_candidate":
                    created = identity_service.convert_unknown_candidate_if_stable(track, embedding)
                    if created:
                        add_access_event(
                            camera_id=CAMERA_ID,
                            track_id=track_id,
                            person_id=created,
                            action="UNKNOWN_AUTO_CREATED",
                            confidence=None,
                            extra="pending_validation"
                        )
                        mqtt_service.publish({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera_id": CAMERA_ID,
                            "track_id": track_id,
                            "person_id": created,
                            "action": "UNKNOWN_AUTO_CREATED",
                            "extra": "pending_validation"
                        })

                decision, reason = authorization_service.decide(
                    track.get("identity"),
                    track.get("identity_type"),
                    zone_name
                )

                person_id = track.get("identity")
                if person_id:
                    update_last_seen(person_id)
                    session_service.on_track_seen(track, person_id)

                if person_id:
                    since_last_log = now_ts - track["last_logged_ts"]
                    if since_last_log >= LOG_COOLDOWN_SEC:
                        add_access_event(
                            camera_id=CAMERA_ID,
                            track_id=track_id,
                            person_id=person_id,
                            action=decision,
                            confidence=track.get("identity_score"),
                            extra=reason
                        )
                        mqtt_service.publish({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera_id": CAMERA_ID,
                            "track_id": track_id,
                            "person_id": person_id,
                            "identity_type": track.get("identity_type"),
                            "action": decision,
                            "confidence": track.get("identity_score"),
                            "extra": reason
                        })
                        track["last_logged_ts"] = now_ts

                if decision == "ALERT_PENDING" and not track["pending_alert_sent"]:
                    add_alert(
                        camera_id=CAMERA_ID,
                        track_id=track_id,
                        person_id=person_id,
                        alert_type="UNKNOWN_PENDING_VALIDATION",
                        notes=reason,
                        status="open"
                    )
                    track["pending_alert_sent"] = True

                draw_track(frame, track, decision)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        mqtt_service.disconnect()


if __name__ == "__main__":
    main()