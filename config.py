"""Central configuration. Override any value with an environment variable."""
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-only-change-me")
    HOST = os.getenv("HOST", "127.0.0.1")
    PORT = int(os.getenv("PORT", 5000))
    DEBUG = os.getenv("FLASK_DEBUG", "1") == "1"

    # Paths
    MODEL_DIR = BASE_DIR / "models"
    AUDIO_DIR = BASE_DIR / "app" / "static" / "audio"

    # Sprint 2 – camera and detection
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.45))
    NMS_THRESHOLD = 0.4
    DETECT_EVERY_N_FRAMES = int(os.getenv("DETECT_EVERY_N_FRAMES", 3))
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    JPEG_QUALITY = 80
