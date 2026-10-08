"""Background camera thread: grabs frames, runs detection, serves MJPEG."""
import threading
import time

import cv2
import numpy as np

from app.detector import ObjectDetector, draw_detections
from config import Config


def _placeholder(message):
    img = np.full((Config.FRAME_HEIGHT, Config.FRAME_WIDTH, 3), 30, dtype=np.uint8)
    y = Config.FRAME_HEIGHT // 2
    for i, line in enumerate(message.split("\n")):
        cv2.putText(img, line, (24, y + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    ok, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


class VideoStream:
    def __init__(self):
        self._lock = threading.Lock()
        self._thread = None
        self._running = False
        self._jpeg = _placeholder("Starting camera...")
        self._detections = []
        self.error = None

    # -- lifecycle ---------------------------------------------------------
    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False

    # -- worker ------------------------------------------------------------
    def _loop(self):
        try:
            detector = ObjectDetector()
        except FileNotFoundError as exc:
            detector = None
            self.error = str(exc)

        cap = cv2.VideoCapture(Config.CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
        if not cap.isOpened():
            self.error = "Camera not available.\nCheck CAMERA_INDEX and permissions."
            with self._lock:
                self._jpeg = _placeholder(self.error)
            return

        frame_no, detections = 0, []
        while self._running:
            ok, frame = cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            if detector and frame_no % Config.DETECT_EVERY_N_FRAMES == 0:
                detections = detector.detect(frame)
            frame_no += 1

            annotated = draw_detections(frame.copy(), detections)
            ok, buf = cv2.imencode(
                ".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, Config.JPEG_QUALITY]
            )
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
                    self._detections = detections
        cap.release()

    # -- accessors ---------------------------------------------------------
    @property
    def detections(self):
        with self._lock:
            return list(self._detections)

    def frames(self):
        """Generator for the multipart MJPEG response."""
        self.start()
        while True:
            with self._lock:
                jpeg = self._jpeg
            yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            time.sleep(0.04)


stream = VideoStream()
