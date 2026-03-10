import cv2
from config import CAMERA_SOURCE


class CameraSource:
    def __init__(self, source=CAMERA_SOURCE):
        self.source = source
        self.cap = None

    def open(self):
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open camera source")

    def read(self):
        if self.cap is None:
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None