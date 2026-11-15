"""
Configuration module for Automatic Multilanguage Communication System.
Department of Computer Science & Engineering, SKIT Jaipur.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"
AUDIO_CACHE_DIR = STATIC_DIR / "audio_cache"
AUDIO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def _env_flag(name, default=False):
    """Reads a boolean environment variable such as AMCS_DEBUG=1."""
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Project Information
APP_VERSION = "3.2.0"
PROJECT_TITLE = "Automatic Multilanguage Communication System"
COLLEGE_NAME = "Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur"
DEPARTMENT = "Department of Computer Science & Engineering"
TEAM_MEMBERS = [
    {"name": "Amit Tiwari", "roll_no": "23ESKCX010"},
    {"name": "Deepak Choudhary", "roll_no": "23ESKCX026"},
    {"name": "Deependra Singh", "roll_no": "23ESKCX027"}
]
SUPERVISOR = {"name": "Vikas", "designation": "Assistant Professor-2, Dept. of CSE"}

# Server Settings (override with environment variables)
SERVER_HOST = os.environ.get("AMCS_HOST", "0.0.0.0")
SERVER_PORT = int(os.environ.get("AMCS_PORT", "5000"))
DEBUG_MODE = _env_flag("AMCS_DEBUG", False)
USE_HTTPS = _env_flag("AMCS_HTTPS", False)   # self-signed HTTPS so phones can use the camera
MAX_UPLOAD_MB = 16                           # largest accepted request body

# Vision & Object Detection Settings
# "auto" times the candidate models at start-up and keeps the most accurate one that still runs in
# real time on this machine: YOLO26-Medium (53.1 COCO mAP) on a reasonably fast CPU or any GPU,
# otherwise YOLO26-Small (48.6 mAP). Set AMCS_YOLO_MODEL to a file name to force a specific model.
YOLO_MODEL_NAME = os.environ.get("AMCS_YOLO_MODEL", "auto")
YOLO_AUTO_MODELS = ["yolo26m.pt", "yolo26s.pt"]      # most accurate first
YOLO_AUTO_BUDGET_MS = float(os.environ.get("AMCS_YOLO_BUDGET_MS", "330"))   # per frame, at 640 px
# With "auto", the next lighter model is also loaded and answers "fast" requests (phones use fast by
# default), so a phone over Wi-Fi gets quicker answers while a computer keeps the accurate model.
FAST_QUALITY_LIGHT_MODEL = os.environ.get("AMCS_FAST_LIGHT_MODEL", "1") != "0"
# Tried in order if the preferred weights are missing and cannot be downloaded
YOLO_FALLBACK_MODELS = ["yolo26s.pt", "yolo11s.pt", "yolov8s.pt", "yolov8n.pt"]
YOLO_IMAGE_SIZE = 640           # inference resolution (long side) of the "balanced" profile
CONFIDENCE_THRESHOLD = 0.35     # per-frame threshold; speech additionally requires stable tracks
# Two-threshold tracking: a detection between the weak and the normal threshold is kept only when
# it continues an object that was seen clearly within the last HYSTERESIS_SECONDS.
WEAK_CONFIDENCE_THRESHOLD = 0.20
HYSTERESIS_SECONDS = 3.0
CROSS_CLASS_IOU = 0.8           # same box reported as two classes: keep the more confident one
LOW_LIGHT_LUMA = 70.0           # mean brightness (0-255) below which a frame counts as dim (UI hint)
# Brightening dim frames in software (gamma + CLAHE) was benchmarked on darkened COCO images and
# reduced accuracy (it amplifies sensor noise), so it stays off. Real light - the torch - helps.
LOW_LIGHT_ENHANCE = False
IOU_THRESHOLD = 0.45
CAMERA_HFOV_DEGREES = 70.0      # typical laptop / phone camera horizontal field of view
DEFAULT_DETECTION_QUALITY = "balanced"   # "fast" (512 px) | "mobile" (light model, 640 px) | "balanced" (640 px) | "high" (960 px)

# Extended vocabulary (YOLOE, open-vocabulary): doors, stairs, switches, cabinets, keys, wallets...
# Built by tools/build_extended_model.py from modules/extended_vocabulary.py. Optional: if the file
# is missing the app simply runs with the main detector only. Set AMCS_EXTENDED=0 to disable.
EXTENDED_MODEL_NAME = os.environ.get("AMCS_EXTENDED_MODEL", "yoloe-26s-amcs.pt")
EXTENDED_ENABLED = os.environ.get("AMCS_EXTENDED", "1").strip().lower() not in ("0", "false", "no", "off")
EXTENDED_CONFIDENCE = 0.40
EXTENDED_EVERY_N_FRAMES = 3     # the extended model is slower, so the live loop runs it on every 3rd frame
                                # (every 6th at "fast" quality)
EXTENDED_CACHE_SECONDS = 2.5    # its results are reused on the frames in between, while the view stays the same
# Classes the extended model confuses easily need more confidence (checked by eye on 500 COCO photos:
# stones reported as coins at 0.93, turbans as helmets, oven doors as towels below 0.6)
EXTENDED_CLASS_CONFIDENCE = {
    "coin": 0.95,
    "helmet": 0.6,
    "towel": 0.6,
    "stool": 0.6,
    "trash can": 0.55,
    "mirror": 0.5,
    "window": 0.5,
}
# Mean grey-level change (0-255, exposure removed) of a small thumbnail above which the camera is
# looking at a new scene. Measured: a hand-held view stays below ~21-29, different scenes start at ~32.
EXTENDED_SCENE_CHANGE = 26.0

# Proximity Estimation Configuration (based on typical object height in pixels vs frame height)
# High risk obstacle classes for visually impaired users
PRIORITY_OBSTACLES = {
    "person", "bicycle", "car", "motorcycle", "bus", "truck", "traffic light",
    "stop sign", "fire hydrant", "dog", "cat", "chair", "couch", "potted plant",
    "bed", "dining table", "stairs", "door", "bench", "pole"
}

# Only these can trigger "careful" proximity warnings. A cup or phone held near the camera is
# described, not treated as an obstacle. (Extended-vocabulary obstacles are added in vision_detector.)
OBSTACLE_CLASSES = PRIORITY_OBSTACLES | {
    "suitcase", "refrigerator", "toilet", "parking meter", "horse", "sheep", "cow", "elephant",
    "bear", "train", "boat", "airplane", "motorcycle", "tv", "oven", "sink",
}

# Distance Thresholds (Estimated in meters)
DISTANCE_HAZARD_METERS = 1.3    # Immediate danger / proximity warning
DISTANCE_CAUTION_METERS = 2.8   # Close proximity alert
DISTANCE_NOTICE_METERS = 5.0    # Approaching awareness

# Multilingual Translation & Speech Settings
# Supported ISO-639-1 language codes, native names and flags
SUPPORTED_LANGUAGES = {
    "en": {"name": "English", "native": "English", "flag": "🇬🇧", "speech_code": "en"},
    "hi": {"name": "Hindi", "native": "हिन्दी", "flag": "🇮🇳", "speech_code": "hi"},
    "es": {"name": "Spanish", "native": "Español", "flag": "🇪🇸", "speech_code": "es"},
    "fr": {"name": "French", "native": "Français", "flag": "🇫🇷", "speech_code": "fr"},
    "de": {"name": "German", "native": "Deutsch", "flag": "🇩🇪", "speech_code": "de"},
    "zh-CN": {"name": "Chinese", "native": "中文 (简体)", "flag": "🇨🇳", "speech_code": "zh-CN"},
    "ja": {"name": "Japanese", "native": "日本語", "flag": "🇯🇵", "speech_code": "ja"},
    "ar": {"name": "Arabic", "native": "العربية", "flag": "🇸🇦", "speech_code": "ar"},
    "ru": {"name": "Russian", "native": "Русский", "flag": "🇷🇺", "speech_code": "ru"},
    "bn": {"name": "Bengali", "native": "বাংলা", "flag": "🇧🇩", "speech_code": "bn"},
    "ta": {"name": "Tamil", "native": "தமிழ்", "flag": "🇮🇳", "speech_code": "ta"},
    "te": {"name": "Telugu", "native": "తెలుగు", "flag": "🇮🇳", "speech_code": "te"},
    "mr": {"name": "Marathi", "native": "मराठी", "flag": "🇮🇳", "speech_code": "mr"},
    "gu": {"name": "Gujarati", "native": "ગુજરાતી", "flag": "🇮🇳", "speech_code": "gu"}
}

DEFAULT_LANGUAGE = "en"

# Speech Synthesis Engine
SPEECH_ENGINE = "gTTS"  # 'gTTS' or 'pyttsx3' fallback
SPEECH_COOLDOWN_SECONDS = 8.0  # Generous interval so voice never feels spammy
AUDIO_CACHE_MAX_FILES = 400    # Oldest cached MP3s are pruned beyond this
GTTS_TIMEOUT_SECONDS = 6       # Give up on Google TTS quickly and use browser speech
GTTS_RETRY_AFTER_SECONDS = 60  # After a network failure, skip gTTS for this long

# Location & GPS Settings
# Default fallback coordinate: SKIT Jaipur campus
DEFAULT_LATITUDE = 26.8222
DEFAULT_LONGITUDE = 75.8653
DEFAULT_LOCATION_LABEL = "SKIT Campus, Jagatpura, Jaipur"
DEFAULT_MAP_ZOOM = 16
GEOFENCE_RADIUS_METERS = 500  # Alert caregiver if outside boundary

# Indian Currency Recognition Settings
SUPPORTED_CURRENCIES = [10, 20, 50, 100, 200, 500]

# Caregiver Emergency Dispatch Channels
CAREGIVER_PHONE = "+919876543210"  # Configurable emergency contact
EMERGENCY_SMS_TEMPLATE = "🚨 EMERGENCY ALERT from AMCS User! Location: {label} (Lat: {lat}, Lon: {lon}). Google Maps: {maps_url}"

