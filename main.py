import time
from datetime import datetime
from collections import Counter

import cv2

from config import (
    validate_config,
    WINDOW_NAME,
    CAMERA_ID,
    LOG_COOLDOWN_SEC,
    ATTRIBUTE_UPDATE_COOLDOWN_SEC,
    GENDER_CONFIDENCE_MIN,
    MIN_ATTRIBUTE_FACE_WIDTH,
    MIN_ATTRIBUTE_FACE_HEIGHT,
)

validate_config()

from database import (
    init_db,
    create_demo_seed,
    get_camera_zone,
    add_access_event,
    add_alert,
    add_system_event,
    update_last_seen,
    get_grouped_daily_report,
    get_identity_attributes,
    save_identity_attributes,
)
from camera_source import CameraSource
from detector import FaceDetector
from tracker import MultiFaceTracker
from recognizer import FaceRecognizer
from identity_service import IdentityService
from authorization_service import AuthorizationService
from session_service import SessionService
from mqtt_service import MQTTService
from attribute_service import AttributeService


# =========================================================
# Attribute Stability Helpers
# =========================================================

GENDER_HISTORY_SIZE = 5
GENDER_MIN_VOTES = 3

AGE_HISTORY_SIZE = 7
AGE_MIN_VOTES = 4


def ensure_track_attribute_fields(track: dict):
    # Gender
    track.setdefault("gender_prediction", None)
    track.setdefault("gender_confidence", None)
    track.setdefault("gender_history", [])
    track.setdefault("stable_gender", None)

    # Age
    track.setdefault("age_prediction", None)
    track.setdefault("age_confidence", None)
    track.setdefault("age_history", [])
    track.setdefault("stable_age", None)

    # Shared attribute update timestamp
    track.setdefault("attribute_last_update_ts", 0.0)

    # DB-locked attributes state
    track.setdefault("attributes_loaded_from_db", False)
    track.setdefault("attributes_locked", False)


def ensure_track_runtime_fields(track: dict):
    # Access decision logging
    track.setdefault("last_logged_ts", 0.0)
    track.setdefault("last_logged_decision", None)
    track.setdefault("last_logged_reason", None)

    # Alert throttling
    track.setdefault("last_alert_ts", 0.0)
    track.setdefault("alert_person_id", None)


def should_update_attributes(track: dict) -> bool:
    last_ts = track.get("attribute_last_update_ts", 0.0)
    return (time.time() - last_ts) >= ATTRIBUTE_UPDATE_COOLDOWN_SEC


def face_large_enough_for_attributes(track: dict) -> bool:
    x, y, w, h = track["bbox"]
    return w >= MIN_ATTRIBUTE_FACE_WIDTH and h >= MIN_ATTRIBUTE_FACE_HEIGHT


def update_stable_gender(track: dict, predicted_gender, confidence):
    if predicted_gender is None:
        return

    if confidence is None or confidence < GENDER_CONFIDENCE_MIN:
        return

    history = track.setdefault("gender_history", [])
    history.append(predicted_gender)

    if len(history) > GENDER_HISTORY_SIZE:
        history.pop(0)

    counts = Counter(history)
    best_gender, best_count = counts.most_common(1)[0]

    current_stable = track.get("stable_gender")

    if current_stable is None:
        if best_count >= GENDER_MIN_VOTES:
            track["stable_gender"] = best_gender
    else:
        if best_gender == current_stable:
            track["stable_gender"] = best_gender
        elif best_count >= GENDER_MIN_VOTES:
            track["stable_gender"] = best_gender


def update_stable_age(track: dict, predicted_age, confidence):
    if predicted_age is None:
        return

    if confidence is None:
        return

    history = track.setdefault("age_history", [])
    history.append(predicted_age)

    if len(history) > AGE_HISTORY_SIZE:
        history.pop(0)

    counts = Counter(history)
    best_age, best_count = counts.most_common(1)[0]

    current_stable = track.get("stable_age")

    if current_stable is None:
        if best_count >= AGE_MIN_VOTES:
            track["stable_age"] = best_age
    else:
        if best_age == current_stable:
            track["stable_age"] = best_age
        elif best_count >= AGE_MIN_VOTES:
            track["stable_age"] = best_age


def load_locked_attributes_into_track(track: dict, person_id: str):
    attrs = get_identity_attributes(person_id)
    if not attrs:
        return

    if int(attrs.get("attributes_locked", 0)) != 1:
        return

    track["stable_gender"] = attrs.get("predicted_gender")
    track["stable_age"] = attrs.get("predicted_age_range")

    track["gender_prediction"] = attrs.get("predicted_gender")
    track["age_prediction"] = attrs.get("predicted_age_range")

    track["gender_confidence"] = None
    track["age_confidence"] = None

    track["attributes_locked"] = True
    track["attributes_loaded_from_db"] = True


def try_lock_attributes_for_identity(track: dict, person_id: str) -> bool:
    stable_gender = track.get("stable_gender")
    stable_age = track.get("stable_age")

    if not stable_gender or not stable_age:
        return False

    save_identity_attributes(
        person_id=person_id,
        predicted_gender=stable_gender,
        predicted_age_range=stable_age,
        lock_attributes=1,
    )

    track["gender_prediction"] = stable_gender
    track["age_prediction"] = stable_age
    track["gender_confidence"] = None
    track["age_confidence"] = None
    track["attributes_locked"] = True
    track["attributes_loaded_from_db"] = True

    return True


def should_log_access_decision(track: dict, decision: str, reason: str, now_ts: float) -> bool:
    last_ts = track.get("last_logged_ts", 0.0)
    last_decision = track.get("last_logged_decision")
    last_reason = track.get("last_logged_reason")

    decision_changed = decision != last_decision
    reason_changed = reason != last_reason
    cooldown_elapsed = (now_ts - last_ts) >= LOG_COOLDOWN_SEC

    return decision_changed or reason_changed or cooldown_elapsed


def should_create_alert(track: dict, person_id: str | None, now_ts: float) -> bool:
    last_alert_ts = track.get("last_alert_ts", 0.0)
    alert_person_id = track.get("alert_person_id")

    person_changed = person_id != alert_person_id
    cooldown_elapsed = (now_ts - last_alert_ts) >= LOG_COOLDOWN_SEC

    return person_changed or cooldown_elapsed


def reset_alert_state_if_needed(track: dict, decision: str, person_id: str | None):
    if decision != "ALERT_PENDING":
        track["alert_person_id"] = None
        return

    if track.get("alert_person_id") is not None and track.get("alert_person_id") != person_id:
        track["alert_person_id"] = None


# =========================================================
# Drawing Helpers
# =========================================================

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
        (x, max(20, y - 62)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
    )

    if decision:
        cv2.putText(
            frame,
            decision,
            (x, max(20, y - 42)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    stable_gender = track.get("stable_gender")
    raw_gender = track.get("gender_prediction")
    gender_confidence = track.get("gender_confidence")
    gender_to_show = stable_gender if stable_gender else raw_gender

    stable_age = track.get("stable_age")
    raw_age = track.get("age_prediction")
    age_confidence = track.get("age_confidence")
    age_to_show = stable_age if stable_age else raw_age

    y_line = max(20, y - 22)

    if gender_to_show:
        gender_line = f"Gender: {gender_to_show}"
        if gender_confidence is not None:
            gender_line += f" ({gender_confidence:.2f})"

        cv2.putText(
            frame,
            gender_line,
            (x, y_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )
        y_line += 20

    if age_to_show:
        age_line = f"Age: {age_to_show}"
        if age_confidence is not None:
            age_line += f" ({age_confidence:.2f})"

        cv2.putText(
            frame,
            age_line,
            (x, y_line),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            color,
            2,
        )


# =========================================================
# Console Reporting Helpers
# =========================================================

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
            "stable_gender": track.get("stable_gender"),
            "stable_age": track.get("stable_age"),
            "access_granted_count": track.get("access_granted_count", 0),
            "access_denied_count": track.get("access_denied_count", 0),
            "access_alert_count": track.get("access_alert_count", 0),
        })

    return grouped


def _merge_historical_and_live(report: dict, live_summary: dict):
    merged = {
        "employee": {},
        "visitor": {},
        "unknown": {},
    }

    for group_name in ("employee", "visitor", "unknown"):
        for item in report[group_name]:
            merged[group_name][item["person_id"]] = {
                "person_id": item["person_id"],
                "person_type": item["person_type"],
                "historical_appearances": item.get("appearances", 0),
                "historical_visible_seconds": item.get("total_visible_seconds", 0.0) or 0.0,
                "effective_visible_seconds": item.get("total_visible_seconds", 0.0) or 0.0,
                "live_visible_seconds": 0.0,
                "first_entry": item.get("first_entry"),
                "last_entry": item.get("last_entry"),
                "total_visible_hours": item.get("total_visible_hours"),
                "is_live_now": False,
                "live_track_id": None,
                "identity_score": None,
                "stable_gender": None,
                "stable_age": None,
                "access_granted_count": 0,
                "access_denied_count": 0,
                "access_alert_count": 0,
            }

        for live_item in live_summary[group_name]:
            person_id = live_item["person_id"]

            if person_id not in merged[group_name]:
                merged[group_name][person_id] = {
                    "person_id": person_id,
                    "person_type": live_item["person_type"],
                    "historical_appearances": 0,
                    "historical_visible_seconds": 0.0,
                    "effective_visible_seconds": 0.0,
                    "live_visible_seconds": 0.0,
                    "first_entry": None,
                    "last_entry": None,
                    "total_visible_hours": None,
                    "is_live_now": False,
                    "live_track_id": None,
                    "identity_score": None,
                    "stable_gender": None,
                    "stable_age": None,
                    "access_granted_count": 0,
                    "access_denied_count": 0,
                    "access_alert_count": 0,
                }

            merged[group_name][person_id]["live_visible_seconds"] += live_item.get("live_visible_seconds", 0.0) or 0.0
            merged[group_name][person_id]["effective_visible_seconds"] += live_item.get("live_visible_seconds", 0.0) or 0.0
            merged[group_name][person_id]["is_live_now"] = True
            merged[group_name][person_id]["live_track_id"] = live_item.get("track_id")
            merged[group_name][person_id]["identity_score"] = live_item.get("identity_score")
            merged[group_name][person_id]["stable_gender"] = live_item.get("stable_gender")
            merged[group_name][person_id]["stable_age"] = live_item.get("stable_age")
            merged[group_name][person_id]["access_granted_count"] += live_item.get("access_granted_count", 0)
            merged[group_name][person_id]["access_denied_count"] += live_item.get("access_denied_count", 0)
            merged[group_name][person_id]["access_alert_count"] += live_item.get("access_alert_count", 0)

        merged[group_name] = sorted(
            merged[group_name].values(),
            key=lambda x: (x.get("effective_visible_seconds", 0.0), x["person_id"]),
            reverse=True,
        )

    return merged


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
            print(f"      stable gender: {item['stable_gender']}")
            print(f"      stable age: {item['stable_age']}")
            print(f"      authorized count: {item['access_granted_count']}")
            print(f"      denied count:     {item['access_denied_count']}")
            print(f"      alert count:      {item['access_alert_count']}")

    if not any_data:
        print("\nNo active recognized tracks right now.")

    print("-" * 70 + "\n")


def _print_effective_group_section(title: str, items: list, include_work_hours: bool):
    print(f"\n[{title} - EFFECTIVE TOTAL NOW]")

    if not items:
        print("  No data.")
        return

    for item in items:
        hist_sec = item.get("historical_visible_seconds", 0.0) or 0.0
        live_sec = item.get("live_visible_seconds", 0.0) or 0.0
        eff_sec = item.get("effective_visible_seconds", 0.0) or 0.0

        print(f"  - {item['person_id']}")
        print(f"      historical visible time: {hist_sec:.1f} sec ({hist_sec / 60.0:.2f} min)")
        print(f"      live visible time now:   {live_sec:.1f} sec ({live_sec / 60.0:.2f} min)")
        print(f"      effective total now:     {eff_sec:.1f} sec ({eff_sec / 60.0:.2f} min)")
        print(f"      historical appearances:  {item.get('historical_appearances', 0)}")
        print(f"      live now:                {item.get('is_live_now', False)}")
        print(f"      first entry:             {item.get('first_entry')}")
        print(f"      last entry:              {item.get('last_entry')}")

        if item.get("is_live_now"):
            print(f"      active track:            {item.get('live_track_id')}")
            print(f"      live identity score:     {item.get('identity_score')}")
            print(f"      stable gender:           {item.get('stable_gender')}")
            print(f"      stable age:              {item.get('stable_age')}")
            print(f"      authorized count:        {item.get('access_granted_count', 0)}")
            print(f"      denied count:            {item.get('access_denied_count', 0)}")
            print(f"      alert count:             {item.get('access_alert_count', 0)}")

        if include_work_hours:
            total_hours = item.get("total_visible_hours")
            if total_hours is None:
                total_hours = round(eff_sec / 3600.0, 4)
            print(f"      work hours estimate now: {total_hours:.4f} hour(s)")


def print_daily_report(date_str: str, live_tracks: dict):
    report = get_grouped_daily_report(date_str)
    live_summary = _build_live_active_summary(live_tracks)
    effective = _merge_historical_and_live(report, live_summary)

    print("\n" + "=" * 70)
    print(f"[REPORT] Daily analytics for {report['date']}")
    print("=" * 70)

    print("\n[HISTORICAL FINALIZED DATA]")
    _print_group_section("EMPLOYEES", report["employee"], include_work_hours=True)
    _print_group_section("VISITORS", report["visitor"], include_work_hours=False)
    _print_group_section("UNKNOWN", report["unknown"], include_work_hours=False)

    _print_live_active_section(live_tracks)

    print("\n" + "-" * 70)
    print("[EFFECTIVE TOTALS = HISTORICAL + LIVE ACTIVE NOW]")
    print("-" * 70)
    _print_effective_group_section("EMPLOYEES", effective["employee"], include_work_hours=True)
    _print_effective_group_section("VISITORS", effective["visitor"], include_work_hours=False)
    _print_effective_group_section("UNKNOWN", effective["unknown"], include_work_hours=False)

    print("=" * 70 + "\n")


# =========================================================
# MQTT / Event Helpers
# =========================================================

def publish_event(mqtt_service, payload: dict, kind: str = "system"):
    try:
        if kind == "access":
            mqtt_service.publish_access(payload)
        elif kind == "alert":
            mqtt_service.publish_alert(payload)
        else:
            mqtt_service.publish_system(payload)
    except Exception:
        pass


def log_system_event(
    mqtt_service,
    event_type: str,
    camera_id: str,
    zone_id: str | None,
    track_id: str | None = None,
    person_id: str | None = None,
    person_type: str | None = None,
    confidence: float | None = None,
    payload: dict | None = None,
):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    add_system_event(
        event_type=event_type,
        camera_id=camera_id,
        zone_id=zone_id,
        track_id=track_id,
        person_id=person_id,
        person_type=person_type,
        confidence=confidence,
        payload=payload or {},
    )

    publish_event(
        mqtt_service,
        {
            "timestamp": timestamp,
            "event_type": event_type,
            "camera_id": camera_id,
            "zone_id": zone_id,
            "track_id": track_id,
            "person_id": person_id,
            "person_type": person_type,
            "confidence": confidence,
            "payload": payload or {},
        },
        kind="system",
    )


# =========================================================
# Main
# =========================================================

def main():
    init_db()
    create_demo_seed()

    zone_info = get_camera_zone(CAMERA_ID)
    zone_id = zone_info["zone_id"]
    zone_name = zone_info["zone_name"]

    camera = CameraSource()
    detector = FaceDetector()
    tracker = MultiFaceTracker()
    recognizer = FaceRecognizer()
    identity_service = IdentityService(recognizer, CAMERA_ID, zone_id=zone_id)
    authorization_service = AuthorizationService()
    session_service = SessionService(zone_id=zone_id, zone_name=zone_name)
    mqtt_service = MQTTService()
    attribute_service = AttributeService()

    mqtt_service.connect()
    camera.open()

    print("[INFO] Enterprise prototype started")
    print(f"[INFO] Camera ID: {CAMERA_ID}")
    print(f"[INFO] Zone ID: {zone_id}")
    print(f"[INFO] Zone Name: {zone_name}")
    print("[INFO] Press Q to quit")
    print("[INFO] Press R to print today's analytics report + live active tracks + effective totals")

    prev_time = time.time()
    fps = 0.0

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

                if removed.get("identity"):
                    log_system_event(
                        mqtt_service=mqtt_service,
                        event_type="track_removed",
                        camera_id=CAMERA_ID,
                        zone_id=zone_id,
                        track_id=removed.get("track_id"),
                        person_id=removed.get("identity"),
                        person_type=removed.get("identity_type"),
                        confidence=removed.get("identity_score"),
                        payload={
                            "reason": "track_timeout",
                            "seen_frames": removed.get("seen_frames"),
                            "missing_frames": removed.get("missing_frames"),
                        },
                    )

            current_time = time.time()
            dt = current_time - prev_time
            if dt > 0:
                fps = 1.0 / dt
            prev_time = current_time

            cv2.putText(
                frame,
                f"Cam:{CAMERA_ID} Zone:{zone_name} Faces:{len(detections)} Tracks:{len(live_tracks)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.putText(
                frame,
                f"FPS: {fps:.2f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

            for track_id, track in live_tracks.items():
                ensure_track_attribute_fields(track)
                ensure_track_runtime_fields(track)

                if not track["updated"]:
                    draw_track(frame, track)
                    continue

                try:
                    embedding = recognizer.extract_embedding(frame, track["face_row"])
                except Exception:
                    draw_track(frame, track)
                    continue

                previous_identity = track.get("identity")
                previous_identity_type = track.get("identity_type")

                identity_service.process_track_identity(track, embedding)

                if track.get("identity_type") == "unknown_candidate":
                    created_or_reused = identity_service.convert_unknown_candidate_if_stable(track, embedding)

                    if created_or_reused:
                        if track.get("unknown_just_created"):
                            add_access_event(
                                camera_id=CAMERA_ID,
                                zone_id=zone_id,
                                track_id=track_id,
                                person_id=created_or_reused,
                                person_type="unknown",
                                action="UNKNOWN_AUTO_CREATED",
                                confidence=None,
                                extra="pending_validation",
                            )

                            publish_event(
                                mqtt_service,
                                {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "camera_id": CAMERA_ID,
                                    "zone_id": zone_id,
                                    "track_id": track_id,
                                    "person_id": created_or_reused,
                                    "person_type": "unknown",
                                    "action": "UNKNOWN_AUTO_CREATED",
                                    "extra": "pending_validation",
                                },
                                kind="access",
                            )

                            log_system_event(
                                mqtt_service=mqtt_service,
                                event_type="unknown_created",
                                camera_id=CAMERA_ID,
                                zone_id=zone_id,
                                track_id=track_id,
                                person_id=created_or_reused,
                                person_type="unknown",
                                confidence=None,
                                payload={
                                    "status": "pending_validation",
                                    "source": "auto_creation",
                                },
                            )

                        elif track.get("unknown_just_reused"):
                            log_system_event(
                                mqtt_service=mqtt_service,
                                event_type="unknown_reused",
                                camera_id=CAMERA_ID,
                                zone_id=zone_id,
                                track_id=track_id,
                                person_id=created_or_reused,
                                person_type="unknown",
                                confidence=track.get("identity_score"),
                                payload={
                                    "source": "unknown_reidentification",
                                },
                            )

                if track.get("identity_just_confirmed"):
                    log_system_event(
                        mqtt_service=mqtt_service,
                        event_type="person_recognized",
                        camera_id=CAMERA_ID,
                        zone_id=zone_id,
                        track_id=track_id,
                        person_id=track.get("identity"),
                        person_type=track.get("identity_type"),
                        confidence=track.get("identity_score"),
                        payload={
                            "previous_identity": previous_identity,
                            "previous_identity_type": previous_identity_type,
                            "identity_switched": track.get("identity_just_switched", False),
                        },
                    )

                person_id = track.get("identity")
                person_type = track.get("identity_type")

                # ---------------------------------------------
                # Load locked attributes once from DB
                # ---------------------------------------------
                if person_id and person_type in ("employee", "visitor", "unknown"):
                    if not track.get("attributes_loaded_from_db", False):
                        load_locked_attributes_into_track(track, person_id)

                # ---------------------------------------------
                # Predict attributes only when NOT locked
                # ---------------------------------------------
                if (
                    person_id
                    and person_type in ("employee", "visitor", "unknown")
                    and not track.get("attributes_locked", False)
                    and should_update_attributes(track)
                    and face_large_enough_for_attributes(track)
                ):
                    attr_result = attribute_service.predict_attributes(frame, track["bbox"])

                    new_gender = attr_result.get("gender_prediction")
                    new_gender_conf = attr_result.get("gender_confidence")
                    new_age = attr_result.get("age_prediction")
                    new_age_conf = attr_result.get("age_confidence")

                    old_stable_gender = track.get("stable_gender")
                    old_raw_gender = track.get("gender_prediction")
                    old_raw_gender_conf = track.get("gender_confidence")

                    old_stable_age = track.get("stable_age")
                    old_raw_age = track.get("age_prediction")
                    old_raw_age_conf = track.get("age_confidence")

                    track["gender_prediction"] = new_gender
                    track["gender_confidence"] = new_gender_conf
                    track["age_prediction"] = new_age
                    track["age_confidence"] = new_age_conf
                    track["attribute_last_update_ts"] = now_ts

                    update_stable_gender(track, new_gender, new_gender_conf)
                    update_stable_age(track, new_age, new_age_conf)

                    stable_gender_changed = old_stable_gender != track.get("stable_gender")
                    stable_age_changed = old_stable_age != track.get("stable_age")

                    raw_gender_changed = (
                        old_raw_gender != new_gender
                        or round(old_raw_gender_conf or 0.0, 3) != round(new_gender_conf or 0.0, 3)
                    )

                    raw_age_changed = (
                        old_raw_age != new_age
                        or round(old_raw_age_conf or 0.0, 3) != round(new_age_conf or 0.0, 3)
                    )

                    if (
                        (new_gender or new_age)
                        and (
                            stable_gender_changed
                            or raw_gender_changed
                            or stable_age_changed
                            or raw_age_changed
                        )
                    ):
                        log_system_event(
                            mqtt_service=mqtt_service,
                            event_type="attribute_updated",
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            confidence=track.get("identity_score"),
                            payload={
                                "gender_prediction": new_gender,
                                "gender_confidence": new_gender_conf,
                                "stable_gender": track.get("stable_gender"),
                                "gender_history": track.get("gender_history", []),
                                "age_prediction": new_age,
                                "age_confidence": new_age_conf,
                                "stable_age": track.get("stable_age"),
                                "age_history": track.get("age_history", []),
                            },
                        )

                    # Lock permanently once both values are stable
                    if try_lock_attributes_for_identity(track, person_id):
                        log_system_event(
                            mqtt_service=mqtt_service,
                            event_type="attributes_locked",
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            confidence=track.get("identity_score"),
                            payload={
                                "predicted_gender": track.get("stable_gender"),
                                "predicted_age_range": track.get("stable_age"),
                            },
                        )

                decision, reason = authorization_service.decide(
                    person_id=person_id,
                    person_type=person_type,
                    zone_name=zone_name,
                    zone_id=zone_id,
                )

                reset_alert_state_if_needed(track, decision, person_id)

                if person_id and person_type in ("employee", "visitor", "unknown"):
                    update_last_seen(person_id)
                    session_service.on_track_seen(track, person_id)

                if person_id:
                    if should_log_access_decision(track, decision, reason, now_ts):
                        session_service.record_access_decision(track, decision)

                        add_access_event(
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            action=decision,
                            confidence=track.get("identity_score"),
                            extra=reason,
                        )

                        publish_event(
                            mqtt_service,
                            {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "camera_id": CAMERA_ID,
                                "zone_id": zone_id,
                                "track_id": track_id,
                                "person_id": person_id,
                                "person_type": person_type,
                                "action": decision,
                                "confidence": track.get("identity_score"),
                                "extra": reason,
                            },
                            kind="access",
                        )

                        log_system_event(
                            mqtt_service=mqtt_service,
                            event_type="access_decision",
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            confidence=track.get("identity_score"),
                            payload={
                                "decision": decision,
                                "reason": reason,
                            },
                        )

                        track["last_logged_ts"] = now_ts
                        track["last_logged_decision"] = decision
                        track["last_logged_reason"] = reason

                if decision == "ALERT_PENDING" and person_id:
                    if should_create_alert(track, person_id, now_ts):
                        add_alert(
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            alert_type="UNKNOWN_PENDING_VALIDATION",
                            notes=reason,
                            status="open",
                        )

                        publish_event(
                            mqtt_service,
                            {
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "camera_id": CAMERA_ID,
                                "zone_id": zone_id,
                                "track_id": track_id,
                                "person_id": person_id,
                                "person_type": person_type,
                                "alert_type": "UNKNOWN_PENDING_VALIDATION",
                                "reason": reason,
                                "status": "open",
                            },
                            kind="alert",
                        )

                        log_system_event(
                            mqtt_service=mqtt_service,
                            event_type="alert_created",
                            camera_id=CAMERA_ID,
                            zone_id=zone_id,
                            track_id=track_id,
                            person_id=person_id,
                            person_type=person_type,
                            confidence=track.get("identity_score"),
                            payload={
                                "alert_type": "UNKNOWN_PENDING_VALIDATION",
                                "reason": reason,
                            },
                        )

                        track["last_alert_ts"] = now_ts
                        track["alert_person_id"] = person_id

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