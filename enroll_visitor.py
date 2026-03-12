import json
import sqlite3
from datetime import datetime

import cv2

from config import (
    DB_FILE,
    WINDOW_NAME,
)
from camera_source import CameraSource
from detector import FaceDetector
from recognizer import FaceRecognizer
from database import ensure_identity, add_embedding


CAPTURE_TARGET = 8


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


def main():
    print("\n=== Visitor Enrollment ===\n")

    person_id = input("Visitor person_id (example vis_001): ").strip()
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
    print("[INFO] Press C to capture a sample")
    print("[INFO] Press Q to quit")
    print(f"[INFO] Need {CAPTURE_TARGET} good samples\n")

    camera = CameraSource()
    detector = FaceDetector()
    recognizer = FaceRecognizer()

    saved_count = 0

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
            draw_status(display, "Press C to capture | Press Q to quit", y=90, color=(200, 200, 200))

            if best_det is not None:
                x, y, w, h = best_det["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_status(display, "Face detected", y=120, color=(0, 255, 0))
            else:
                draw_status(display, "No face detected", y=120, color=(0, 0, 255))

            cv2.imshow(f"{WINDOW_NAME} - Visitor Enrollment", display)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break

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