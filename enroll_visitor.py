import json
import sqlite3
from datetime import datetime

import cv2

from config import (
    DB_FILE,
    WINDOW_NAME,
    UNKNOWN_REUSE_THRESHOLD,
)
from camera_source import CameraSource
from detector import FaceDetector
from recognizer import FaceRecognizer
from database import (
    ensure_identity,
    add_embedding,
    get_identity_info,
    resolve_unknown_to_existing_identity,
)


CAPTURE_TARGET = 8
UNKNOWN_SUGGESTION_MIN_SCORE = UNKNOWN_REUSE_THRESHOLD


def insert_visitor_profile(
    person_id: str,
    display_name: str,
    host_person_id: str,
    visit_reason: str,
    valid_from: str,
    valid_to: str,
):
    ensure_identity(
        person_id=person_id,
        person_type="visitor",
        display_name=display_name,
        status="active",
        is_active=1,
    )

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO visitors (
        person_id,
        host_person_id,
        visit_reason,
        valid_from,
        valid_to
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        person_id,
        host_person_id,
        visit_reason,
        valid_from,
        valid_to,
    ))

    conn.commit()
    conn.close()


def draw_status(frame, text, y=30, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        (10, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        color,
        2,
    )


def ask_yes_no(prompt: str) -> bool:
    value = input(prompt).strip().lower()
    return value in ("y", "yes", "1", "true")


def suggest_unknown_matches(recognizer: FaceRecognizer, embedding, min_score=UNKNOWN_SUGGESTION_MIN_SCORE, top_k=5):
    candidates = recognizer.recognize_top_k(embedding, k=top_k)
    filtered = []

    for item in candidates:
        if item["person_type"] != "unknown":
            continue
        if item["score"] < min_score:
            continue

        info = get_identity_info(item["person_id"])
        if info is None:
            continue
        if info["status"] in ("blocked", "merged", "inactive"):
            continue

        filtered.append(item)

    return filtered


def choose_unknown_candidate(candidates):
    if not candidates:
        return None

    print("\n[INFO] Potential unknown matches found:")
    for idx, item in enumerate(candidates, start=1):
        print(f"  {idx}) {item['person_id']}  score={item['score']:.3f}")

    print("  0) None of these, continue as new visitor")

    while True:
        raw = input("Choose candidate number: ").strip()
        if raw == "":
            return None

        try:
            choice = int(raw)
        except ValueError:
            print("Invalid choice.")
            continue

        if choice == 0:
            return None

        if 1 <= choice <= len(candidates):
            return candidates[choice - 1]["person_id"]

        print("Choice out of range.")


def main():
    print("\n=== Visitor Enrollment ===\n")

    person_id = input("Visitor person_id (example visitor_001): ").strip()
    display_name = input("Display name: ").strip()
    host_person_id = input("Host person_id (example emp_001): ").strip()
    visit_reason = input("Visit reason: ").strip()
    valid_from = input("Valid from (YYYY-MM-DD HH:MM:SS): ").strip()
    valid_to = input("Valid to   (YYYY-MM-DD HH:MM:SS): ").strip()

    if not person_id:
        print("person_id is required.")
        return

    if not valid_from or not valid_to:
        print("valid_from and valid_to are required.")
        return

    try:
        datetime.strptime(valid_from, "%Y-%m-%d %H:%M:%S")
        datetime.strptime(valid_to, "%Y-%m-%d %H:%M:%S")
    except ValueError:
        print("Invalid datetime format. Use YYYY-MM-DD HH:MM:SS")
        return

    existing_target_info = get_identity_info(person_id)
    if existing_target_info is not None and existing_target_info["person_type"] != "visitor":
        print(f"[ERROR] person_id '{person_id}' already exists but is not a visitor.")
        return

    # Create/update visitor profile BEFORE camera flow.
    insert_visitor_profile(
        person_id=person_id,
        display_name=display_name or person_id,
        host_person_id=host_person_id,
        visit_reason=visit_reason,
        valid_from=valid_from,
        valid_to=valid_to,
    )

    print(f"\n[INFO] Visitor profile saved for {person_id}")
    print("[INFO] Camera will open now")
    print("[INFO] Press S to scan unknown suggestions from current face")
    print("[INFO] Press C to capture a sample")
    print("[INFO] Press Q to quit")
    print(f"[INFO] Need {CAPTURE_TARGET} good samples\n")

    camera = CameraSource()
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    saved_count = 0
    resolved_unknown_id = None
    merge_done = False

    try:
        camera.open()

        while True:
            ok, frame = camera.read()
            if not ok:
                print("[WARN] Failed to read frame")
                break

            detections = detector.detect_all(frame)

            best_det = None
            if detections:
                best_det = max(
                    detections,
                    key=lambda d: d["bbox"][2] * d["bbox"][3]
                )

            display = frame.copy()

            draw_status(display, f"Enroll visitor: {person_id}", y=30)
            draw_status(display, f"Saved samples: {saved_count}/{CAPTURE_TARGET}", y=60, color=(255, 255, 0))
            draw_status(display, "Press S suggest | C capture | Q quit", y=90, color=(200, 200, 200))

            if resolved_unknown_id:
                merge_color = (0, 255, 0) if merge_done else (0, 255, 255)
                merge_text = f"Resolved from: {resolved_unknown_id}"
                if merge_done:
                    merge_text += " [MERGED]"
                draw_status(display, merge_text, y=120, color=merge_color)
                face_status_y = 150
            else:
                face_status_y = 120

            if best_det is not None:
                x, y, w, h = best_det["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_status(display, "Face detected", y=face_status_y, color=(0, 255, 0))
            else:
                draw_status(display, "No face detected", y=face_status_y, color=(0, 0, 255))

            cv2.imshow(f"{WINDOW_NAME} - Visitor Enrollment", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

            if key in (ord("s"), ord("S")):
                if best_det is None:
                    print("[WARN] No face detected. Cannot scan suggestions.")
                    continue

                try:
                    embedding = recognizer.extract_embedding(frame, best_det["face_row"])
                except Exception as e:
                    print(f"[WARN] Failed to extract embedding: {e}")
                    continue

                candidates = suggest_unknown_matches(recognizer, embedding)
                chosen_unknown_id = choose_unknown_candidate(candidates)

                if not chosen_unknown_id:
                    print("[INFO] No unknown candidate selected. Will continue as fresh visitor.")
                    continue

                if merge_done and resolved_unknown_id == chosen_unknown_id:
                    print(f"[INFO] Already merged from {chosen_unknown_id}.")
                    continue

                should_merge_history = ask_yes_no(
                    "Reassign old unknown history to this visitor? [y/N]: "
                )

                note = f"Resolved {chosen_unknown_id} into visitor {person_id}"

                try:
                    result = resolve_unknown_to_existing_identity(
                        unknown_person_id=chosen_unknown_id,
                        target_person_id=person_id,
                        note=note,
                        copy_embeddings=True,
                        reassign_history=should_merge_history,
                    )
                    recognizer.reload_embeddings()

                    resolved_unknown_id = chosen_unknown_id
                    merge_done = True

                    print(f"[OK] Unknown resolved immediately into visitor: {result}")
                except Exception as e:
                    print(f"[ERROR] Failed to resolve unknown immediately: {e}")

                continue

            if key in (ord("c"), ord("C")):
                if best_det is None:
                    print("[WARN] No face detected. Cannot capture.")
                    continue

                try:
                    embedding = recognizer.extract_embedding(frame, best_det["face_row"])
                except Exception as e:
                    print(f"[WARN] Failed to extract embedding: {e}")
                    continue

                emb_json = json.dumps(embedding.astype(float).tolist())
                add_embedding(person_id, emb_json, quality_score=1.0)
                saved_count += 1

                print(f"[OK] Sample {saved_count}/{CAPTURE_TARGET} saved")

                if saved_count >= CAPTURE_TARGET:
                    print(f"\n[SUCCESS] Visitor enrollment completed for {person_id}")
                    print("[INFO] Restart your main app so recognizer reloads embeddings.")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()