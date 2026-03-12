import cv2

from config import (
    CAMERA_SOURCE,
    FRAME_WIDTH,
    FRAME_HEIGHT,
)


class CameraSource:
    def __init__(self, source=CAMERA_SOURCE):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)

        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera source")

        # Apply preferred capture size for PC camera prototype
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    def read(self):
        if self.cap is None:
            return False, None

        ok, frame = self.cap.read()
        if not ok or frame is None:
            return False, None

        return True, frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None