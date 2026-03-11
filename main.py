import time
import cv2
from datetime import datetime

from config import validate_config, WINDOW_NAME, CAMERA_ID, LOG_COOLDOWN_SEC

validate_config()

from database import (
    init_db,
    create_demo_seed,
    get_camera_zone,
    add_access_event,
    add_alert,
    update_last_seen,
    get_grouped_daily_report,
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

    cv2.putText(
        frame,
        top,
        (x, y - 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    if decision:
        cv2.putText(
            frame,
            decision,
            (x, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )


def _print_group_section(title: str, items: list, include_work_hours: bool):
    print(f"\n[{title}]")

    if not items:
        print("  No data.")
        return

    for item in items:
        total_sec = item.get("total_visible_seconds", 0.0) or 0.0
        total_min = total_sec / 60.0
        first_entry = item.get("first_entry")
        last_entry = item.get("last_entry")

        print(f"  - {item['person_id']}")
        print(f"      appearances: {item.get('appearances', 0)}")
        print(f"      total visible time: {total_sec:.1f} sec ({total_min:.2f} min)")
        print(f"      first entry: {first_entry}")
        print(f"      last entry:  {last_entry}")

        if include_work_hours:
            total_hours = item.get("total_visible_hours")
            if total_hours is None:
                total_hours = round(total_sec / 3600.0, 4)
            print(f"      work hours in front of camera: {total_hours:.4f} hour(s)")


def _build_live_active_summary(live_tracks: dict):
    now_ts = time.time()

    grouped = {
        "employee": [],
        "visitor": [],
        "unknown": [],
    }

    for track_id, track in live_tracks.items():
        person_id = track.get("identity")
        person_type = track.get("identity_type")

        if not person_id:
            continue

        if person_type not in ("employee", "visitor", "unknown"):
            continue

        start_ts = track.get("visible_session_start_ts")
        last_seen_ts = track.get("last_seen_ts")

        live_duration = 0.0
        if start_ts is not None:
            end_ts = last_seen_ts if last_seen_ts is not None else now_ts
            if end_ts < start_ts:
                end_ts = start_ts
            live_duration = max(0.0, end_ts - start_ts)

        grouped[person_type].append({
            "track_id": track_id,
            "person_id": person_id,
            "person_type": person_type,
            "live_visible_seconds": live_duration,
            "identity_score": track.get("identity_score"),
        })

    return grouped


def _print_live_active_section(live_tracks: dict):
    live = _build_live_active_summary(live_tracks)

    print("\n" + "-" * 70)
    print("[LIVE ACTIVE TRACKS]")
    print("-" * 70)

    any_data = False

    for group_name in ("employee", "visitor", "unknown"):
        items = live[group_name]
        print(f"\n[{group_name.upper()} - ACTIVE NOW]")

        if not items:
            print("  No active tracks.")
            continue

        any_data = True
        for item in items:
            print(f"  - {item['person_id']} (track={item['track_id']})")
            print(f"      active visible time now: {item['live_visible_seconds']:.1f} sec")
            print(f"      live identity score: {item['identity_score']}")

    if not any_data:
        print("\nNo active recognized tracks right now.")

    print("-" * 70 + "\n")


def print_daily_report(date_str: str, live_tracks: dict):
    report = get_grouped_daily_report(date_str)

    print("\n" + "=" * 70)
    print(f"[REPORT] Daily analytics for {report['date']}")
    print("=" * 70)

    _print_group_section("EMPLOYEES", report["employee"], include_work_hours=True)
    _print_group_section("VISITORS", report["visitor"], include_work_hours=False)
    _print_group_section("UNKNOWN", report["unknown"], include_work_hours=False)

    _print_live_active_section(live_tracks)

    print("=" * 70 + "\n")


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
    print("[INFO] Press Q to quit")
    print("[INFO] Press R to print today's analytics report + live active tracks")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                print("[WARN] Failed to read frame from camera")
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
                2,
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

                if track.get("identity_type") == "unknown_candidate":
                    created = identity_service.convert_unknown_candidate_if_stable(track, embedding)
                    if created:
                        add_access_event(
                            camera_id=CAMERA_ID,
                            track_id=track_id,
                            person_id=created,
                            action="UNKNOWN_AUTO_CREATED",
                            confidence=None,
                            extra="pending_validation",
                        )
                        mqtt_service.publish({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera_id": CAMERA_ID,
                            "track_id": track_id,
                            "person_id": created,
                            "action": "UNKNOWN_AUTO_CREATED",
                            "extra": "pending_validation",
                        })

                decision, reason = authorization_service.decide(
                    track.get("identity"),
                    track.get("identity_type"),
                    zone_name,
                )

                person_id = track.get("identity")
                if person_id:
                    update_last_seen(person_id)
                    session_service.on_track_seen(track, person_id)

                if person_id:
                    since_last_log = now_ts - track.get("last_logged_ts", 0)
                    if since_last_log >= LOG_COOLDOWN_SEC:
                        add_access_event(
                            camera_id=CAMERA_ID,
                            track_id=track_id,
                            person_id=person_id,
                            action=decision,
                            confidence=track.get("identity_score"),
                            extra=reason,
                        )
                        mqtt_service.publish({
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "camera_id": CAMERA_ID,
                            "track_id": track_id,
                            "person_id": person_id,
                            "identity_type": track.get("identity_type"),
                            "action": decision,
                            "confidence": track.get("identity_score"),
                            "extra": reason,
                        })
                        track["last_logged_ts"] = now_ts

                if decision == "ALERT_PENDING" and not track.get("pending_alert_sent", False):
                    add_alert(
                        camera_id=CAMERA_ID,
                        track_id=track_id,
                        person_id=person_id,
                        alert_type="UNKNOWN_PENDING_VALIDATION",
                        notes=reason,
                        status="open",
                    )
                    track["pending_alert_sent"] = True

                draw_track(frame, track, decision)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                today_str = datetime.now().strftime("%Y-%m-%d")
                print_daily_report(today_str, tracker.tracks)

    finally:
        for track in list(tracker.tracks.values()):
            session_service.on_track_removed(track)

        camera.release()
        cv2.destroyAllWindows()
        mqtt_service.disconnect()


if __name__ == "__main__":
    main()