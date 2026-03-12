import json
import sqlite3
from datetime import datetime

import cv2

from config import (
    DB_FILE,
    CAMERA_ID,
    WINDOW_NAME,
)
from camera_source import CameraSource
from detector import FaceDetector
from recognizer import FaceRecognizer
from database import ensure_identity, add_embedding


CAPTURE_TARGET = 8


def insert_employee_profile(
    person_id: str,
    display_name: str,
    employee_code: str,
    department: str,
    role_name: str,
    schedule_id: str,
):
    ensure_identity(
        person_id=person_id,
        person_type="employee",
        display_name=display_name,
        status="active",
    )

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    c.execute("""
    INSERT OR REPLACE INTO employees (
        person_id,
        employee_code,
        department,
        role_name,
        schedule_id
    )
    VALUES (?, ?, ?, ?, ?)
    """, (
        person_id,
        employee_code,
        department,
        role_name,
        schedule_id,
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
    print("\n=== Employee Enrollment ===\n")

    person_id = input("Employee person_id (example emp_002): ").strip()
    display_name = input("Display name: ").strip()
    employee_code = input("Employee code (example E002): ").strip()
    department = input("Department: ").strip()
    role_name = input("Role name (example engineer): ").strip()
    schedule_id = input("Schedule id (example sched_office): ").strip()

    if not person_id:
        print("person_id is required.")
        return

    insert_employee_profile(
        person_id=person_id,
        display_name=display_name or person_id,
        employee_code=employee_code,
        department=department,
        role_name=role_name,
        schedule_id=schedule_id,
    )

    print(f"\n[INFO] Employee profile saved for {person_id}")
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
                # pick biggest face
                best_det = max(
                    detections,
                    key=lambda d: d["bbox"][2] * d["bbox"][3]
                )

            display = frame.copy()

            draw_status(display, f"Enroll employee: {person_id}", y=30)
            draw_status(display, f"Saved samples: {saved_count}/{CAPTURE_TARGET}", y=60, color=(255, 255, 0))
            draw_status(display, "Press C to capture | Press Q to quit", y=90, color=(200, 200, 200))

            if best_det is not None:
                x, y, w, h = best_det["bbox"]
                cv2.rectangle(display, (x, y), (x + w, y + h), (0, 255, 0), 2)
                draw_status(display, "Face detected", y=120, color=(0, 255, 0))
            else:
                draw_status(display, "No face detected", y=120, color=(0, 0, 255))

            cv2.imshow(f"{WINDOW_NAME} - Enrollment", display)

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
                    print(f"\n[SUCCESS] Enrollment completed for {person_id}")
                    print("[INFO] Restart your main app so recognizer reloads embeddings.")
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()