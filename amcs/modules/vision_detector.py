"""
Vision Detector Module using OpenCV & YOLO for real-time object & obstacle detection.

Two detectors work together:
  * the main YOLO26 model (80 COCO classes: people, vehicles, furniture, cups, phones...) on every frame
  * an optional open-vocabulary YOLOE model with an extended everyday vocabulary (doors, stairs,
    switches, cabinets, keys, wallets...) on every few frames, see modules/extended_vocabulary.py

Accuracy measures on top of the raw model output:
  * the largest model the machine can run in real time is picked automatically at start-up
  * dim frames are flagged so the user can be told to switch on a light or the torch
    (brightening the picture in software was measured and made detection worse, so it is off)
  * two-threshold tracking: a faint detection is kept when it continues an object that was seen
    clearly a moment ago, so objects do not flicker in and out; a faint one-off is ignored
  * one object reported as two different classes is reduced to the more confident one

Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import base64
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import config
from modules.extended_vocabulary import EXTENDED_HEIGHTS, EXTENDED_OBSTACLES

# Typical real-world heights (metres) used for monocular distance estimation
STANDARD_OBJECT_HEIGHTS = {
    "person": 1.70, "bicycle": 1.00, "car": 1.50, "motorcycle": 1.15, "bus": 3.20,
    "truck": 3.00, "traffic light": 1.00, "stop sign": 0.80, "fire hydrant": 0.75,
    "bench": 0.80, "dog": 0.55, "cat": 0.30, "chair": 0.85, "couch": 0.85,
    "potted plant": 0.60, "bed": 0.65, "dining table": 0.75, "laptop": 0.25,
    "bottle": 0.26, "cup": 0.12, "backpack": 0.45, "handbag": 0.30, "suitcase": 0.60,
    "cell phone": 0.15, "tv": 0.65, "refrigerator": 1.75, "book": 0.22, "clock": 0.30,
    "toilet": 0.75, "sink": 0.25, "oven": 0.85, "microwave": 0.30, "umbrella": 1.00,
    "keyboard": 0.04, "mouse": 0.04, "remote": 0.15, "vase": 0.30, "bowl": 0.08,
    "wine glass": 0.18, "scissors": 0.18, "toothbrush": 0.19, "teddy bear": 0.35,
    "banana": 0.04, "apple": 0.08, "orange": 0.08, "spoon": 0.03, "fork": 0.03, "knife": 0.03,
}
for _name, _h in EXTENDED_HEIGHTS.items():
    STANDARD_OBJECT_HEIGHTS.setdefault(_name, _h)

# Typical widths, used when an object is cut off by the frame edge (e.g. a person close to the camera)
STANDARD_OBJECT_WIDTHS = {
    "person": 0.45, "bicycle": 1.70, "car": 1.80, "motorcycle": 2.00, "bus": 2.50,
    "truck": 2.50, "bench": 1.50, "dog": 0.70, "cat": 0.40, "chair": 0.50, "couch": 1.90,
    "bed": 1.50, "dining table": 1.20, "laptop": 0.33, "tv": 1.10, "refrigerator": 0.75,
    "suitcase": 0.45, "backpack": 0.32, "toilet": 0.40, "potted plant": 0.40,
    "door": 0.90, "cabinet": 0.80, "wardrobe": 1.00, "desk": 1.20, "shelf": 0.90, "window": 1.00,
}
DEFAULT_OBJECT_HEIGHT = 0.80
OBSTACLES = set(config.OBSTACLE_CLASSES) | EXTENDED_OBSTACLES

# Things people wear: when they sit inside a person's box they are clothing, not separate objects
WORN_ITEMS = {"glasses", "watch", "cap", "helmet", "shoe", "slipper", "earphones", "headphones",
              "towel", "blanket", "bag", "tie", "backpack", "handbag"}
# Extended names that duplicate a main-model class
EXTENDED_DUPLICATES = {"desk": "dining table", "bag": "handbag", "box": "suitcase"}

# Inference resolution per quality profile (the browser captures at the same size).
# "mobile" (phones) runs the lighter model at 640 px: on 500 COCO photos it found more than the main
# model at 512 px (recall .52 vs .52, small objects .29 vs .27) in 60% of the time.
QUALITY_IMAGE_SIZES = {"fast": 512, "mobile": 640, "balanced": 640, "high": 960}
LIGHT_MODEL_QUALITIES = {"fast", "mobile"}

PROJECT_DIR = Path(__file__).resolve().parent.parent


def focal_length_px(width, height):
    """
    Focal length in pixels. The camera's field of view (CAMERA_HFOV_DEGREES) spans the long side of
    the sensor, which is the picture's height when a phone is held upright (portrait). Using the width
    there made every distance about 25-30% too short.
    """
    long_side = max(width, height)
    return (long_side / 2.0) / math.tan(math.radians(getattr(config, "CAMERA_HFOV_DEGREES", 70.0) / 2.0))


def _box_area(b):
    return max(0, b[2] - b[0]) * max(0, b[3] - b[1])


def _intersection(a, b):
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    return max(0, w) * max(0, h)


def _iou(a, b):
    inter = _intersection(a, b)
    return inter / max(1, _box_area(a) + _box_area(b) - inter)


def _resolve_weights(name):
    local = PROJECT_DIR / name
    return str(local) if local.exists() else name   # a bare name lets Ultralytics download it


class VisionDetector:
    # Defaults for the accuracy stages. They live on the class so that a detector built without
    # __init__ (unit-test stubs) behaves like the plain single-threshold detector.
    low_conf = None                 # weak-detection threshold; None = no two-threshold tracking
    hysteresis_seconds = 3.0        # how long after a clear sighting a faint one is still trusted
    cross_class_iou = 0.8           # two classes on (almost) the same box: keep the confident one
    enhance_low_light = False
    low_light_luma = 70.0           # mean brightness (0-255) below which a frame counts as dim
    _memory = ()                    # objects seen clearly a moment ago: [name, box, last_clear_time]
    _view = None
    _active_imgsz = None
    fast_model = None               # lighter model for "fast" / "mobile" requests (see FAST_QUALITY_LIGHT_MODEL)
    ext_class_conf = {}             # per-class minimum confidence for the extended model
    _ext_signature = None           # thumbnail of the view the cached extended results belong to
    _ext_due = False                # the view changed: run the extended model on the next frame
    fast_model_name = None
    _clahe = None
    _lock = threading.RLock()       # the models are not safe to call from two requests at once
    infer_ms = None

    def __init__(self, model_name=config.YOLO_MODEL_NAME, conf_thresh=config.CONFIDENCE_THRESHOLD,
                 extended_model_name=None):
        self.conf_thresh = conf_thresh
        weak = getattr(config, "WEAK_CONFIDENCE_THRESHOLD", None)
        self.low_conf = weak if weak and weak < conf_thresh else None
        self.hysteresis_seconds = getattr(config, "HYSTERESIS_SECONDS", 3.0)
        self.cross_class_iou = getattr(config, "CROSS_CLASS_IOU", 0.8)
        self.enhance_low_light = getattr(config, "LOW_LIGHT_ENHANCE", False)
        self.low_light_luma = getattr(config, "LOW_LIGHT_LUMA", 70.0)
        self._memory = []
        self.iou_thresh = config.IOU_THRESHOLD
        self.imgsz = getattr(config, "YOLO_IMAGE_SIZE", 640)
        self.model_path = model_name
        self.model_name = None
        self.model = None
        self._load_model()

        self.ext_model = None
        self.ext_model_name = None
        self.ext_conf = getattr(config, "EXTENDED_CONFIDENCE", 0.40)
        self.ext_every = max(1, int(getattr(config, "EXTENDED_EVERY_N_FRAMES", 3)))
        self.ext_cache_s = getattr(config, "EXTENDED_CACHE_SECONDS", 2.5)
        self.ext_class_conf = dict(getattr(config, "EXTENDED_CLASS_CONFIDENCE", {}))
        self._frame_counter = 0
        self._ext_cache = (0.0, [])        # (timestamp, raw extended detections)
        self._ext_previous = []            # previous extended run, for cross-run confirmation
        if getattr(config, "EXTENDED_ENABLED", True):
            self._load_extended(extended_model_name or getattr(config, "EXTENDED_MODEL_NAME", None))

    # ------------------------------------------------------------------ model loading
    def _benchmark(self, model):
        """Milliseconds for one frame at the default resolution (best of three, after warm-up)."""
        frame = np.random.default_rng(0).integers(0, 255, (360, 640, 3), dtype=np.uint8)
        model.predict(frame, imgsz=self.imgsz, verbose=False)      # warm-up: the first call is slow
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            model.predict(frame, imgsz=self.imgsz, verbose=False)
            best = min(best, (time.perf_counter() - t0) * 1000.0)
        return best

    def _load_model(self):
        """
        Loads the detector weights. With "auto" the most accurate model that keeps up with the live
        camera on this machine is chosen: each candidate is timed and the first one within
        YOLO_AUTO_BUDGET_MS wins. A named model is loaded as-is, with fallbacks if it is unavailable.
        """
        auto = str(self.model_path).lower() == "auto"
        fallbacks = list(getattr(config, "YOLO_FALLBACK_MODELS", []))
        if auto:
            candidates = list(getattr(config, "YOLO_AUTO_MODELS", [])) or fallbacks
            candidates += [m for m in fallbacks if m not in candidates]
        else:
            candidates = [self.model_path] + [m for m in fallbacks if m != self.model_path]
        budget = float(getattr(config, "YOLO_AUTO_BUDGET_MS", 330))
        loaded = None                                      # (model, name, ms) of the best usable one
        for name in candidates:
            try:
                print(f"[VisionDetector] Loading YOLO model: {name}...")
                model = YOLO(_resolve_weights(name))
                ms = self._benchmark(model)
            except Exception as e:
                print(f"[VisionDetector] Warning: could not load {name}: {e}")
                continue
            if not auto or ms <= budget:
                loaded = (model, name, ms)
                break
            if loaded is None or ms < loaded[2]:
                loaded = (model, name, ms)                 # over budget: remember the fastest so far
            print(f"[VisionDetector] {name} needs {ms:.0f} ms per frame (budget {budget:.0f} ms); trying a lighter model.")
        if loaded is None:
            self.model = None
            print("[VisionDetector] No YOLO model could be loaded; object detection is disabled.")
            return
        self.model, self.model_name, self.infer_ms = loaded[0], loaded[1], round(loaded[2], 1)
        print(f"[VisionDetector] Model {self.model_name} loaded successfully ({self.infer_ms:.0f} ms per frame).")
        if auto and getattr(config, "FAST_QUALITY_LIGHT_MODEL", False):
            self._load_fast_model(list(getattr(config, "YOLO_AUTO_MODELS", [])))

    def _load_fast_model(self, auto_models):
        """Loads the auto model one step lighter than the chosen one, for "fast" and "mobile" requests."""
        if self.model_name not in auto_models:
            return
        lighter = auto_models[auto_models.index(self.model_name) + 1:]
        if not lighter:
            return
        try:
            model = YOLO(_resolve_weights(lighter[0]))
            model.predict(np.zeros((360, 640, 3), dtype=np.uint8), imgsz=self.imgsz, verbose=False)
        except Exception as e:
            print(f"[VisionDetector] Warning: could not load {lighter[0]} for fast quality: {e}")
            return
        self.fast_model, self.fast_model_name = model, lighter[0]
        print(f"[VisionDetector] {self.fast_model_name} loaded for fast quality (phones).")

    def _load_extended(self, name):
        """The extended-vocabulary model is optional; it is only used if its file is present."""
        if not name or not (PROJECT_DIR / name).exists():
            print(f"[VisionDetector] Extended vocabulary model '{name}' not found; using the main detector only.")
            return
        try:
            self.ext_model = YOLO(str(PROJECT_DIR / name))
            self.ext_model.predict(np.zeros((360, 640, 3), dtype=np.uint8), imgsz=self.imgsz, verbose=False)
            self.ext_model_name = name
            print(f"[VisionDetector] Extended vocabulary model {name} loaded ({len(self.ext_model.names)} classes).")
        except Exception as e:
            print(f"[VisionDetector] Warning: could not load extended model {name}: {e}")
            self.ext_model = None

    # ------------------------------------------------------------------ geometry
    def estimate_distance(self, class_name, bbox_height, frame_height, bbox=None, frame_width=None):
        """
        Monocular distance estimate: distance = real_size * focal_length / pixel_size.
        The focal length comes from the camera's field of view across the long side of the picture
        (a phone held upright films in portrait, so that is its height). When the box is cut off by
        the top or bottom of the frame its height is unreliable, so a width-based estimate is used as
        well and the nearer of the two is kept. Clamped to 0.4 m - 15 m.
        """
        name = class_name.lower()
        if bbox_height <= 5:
            return 10.0
        fw = frame_width or int(frame_height * 4 / 3)
        focal = focal_length_px(fw, frame_height)

        real_h = STANDARD_OBJECT_HEIGHTS.get(name, DEFAULT_OBJECT_HEIGHT)
        dist = (real_h * focal) / bbox_height

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            # Boxes of cut-off objects usually stop a few pixels short of the edge
            margin_y, margin_x = max(3, frame_height * 0.025), max(3, fw * 0.025)
            cut_vertically = y1 <= margin_y or y2 >= frame_height - margin_y
            cut_horizontally = x1 <= margin_x or x2 >= fw - margin_x
            bw = max(1, x2 - x1)
            if cut_vertically and not cut_horizontally and name in STANDARD_OBJECT_WIDTHS:
                dist = min(dist, (STANDARD_OBJECT_WIDTHS[name] * focal) / bw)
            # Something person-sized or smaller that fills most of the view is very close.
            # (Buses, trucks etc. fill the view from several metres away, so they are exempt.)
            if real_h <= 2.0 and (bbox_height >= frame_height * 0.85 or bw >= fw * 0.85):
                dist = min(dist, 0.8)

        return max(0.4, min(15.0, round(float(dist), 1)))

    def get_spatial_position(self, cx, frame_width):
        """Determines if object is on user's Left, Center (Ahead), or Right."""
        if cx < frame_width * 0.33:
            return "left"
        if cx > frame_width * 0.67:
            return "right"
        return "ahead"

    @staticmethod
    def _merge_fragments(raw):
        """
        Drops low-confidence boxes that are fragments of a stronger box of the same class
        (e.g. a raised knee detected as a second 'person'). raw: list of (name, conf, [x1,y1,x2,y2]).
        A fragment is clearly smaller than the object it belongs to, or lies mostly inside it; two
        similar-sized people standing shoulder to shoulder are both kept.
        """
        raw = sorted(raw, key=lambda r: -r[1])
        kept = []
        for name, conf, box in raw:
            fragment = False
            if conf < 0.6:
                area = _box_area(box)
                for kname, _kconf, kbox in kept:
                    if kname != name:
                        continue
                    karea = _box_area(kbox)
                    overlap = _intersection(box, kbox) / max(1, min(area, karea))
                    if overlap >= 0.6 or (overlap >= 0.3 and area <= 0.5 * karea):
                        fragment = True
                        break
            if not fragment:
                kept.append((name, conf, box))
        return kept

    def _suppress_cross_class(self, raw):
        """One object reported twice under different names (sofa/bed, cup/bowl): keep the surer one."""
        limit = self.cross_class_iou
        if not limit:
            return raw
        kept = []
        for name, conf, box in sorted(raw, key=lambda r: -r[1]):
            if any(kname != name and _iou(box, kbox) >= limit for kname, _kc, kbox in kept):
                continue
            kept.append((name, conf, box))
        return kept

    def _apply_hysteresis(self, raw, now):
        """
        Two-threshold tracking. raw was predicted at the weak threshold. Detections at or above the
        normal threshold always pass. A weaker one passes only if it continues an object that passed
        within the last few seconds (same class, overlapping box), which keeps a half-hidden or
        badly lit object from blinking out; a weak detection with no history is dropped.
        """
        strong = [r for r in raw if r[1] >= self.conf_thresh]
        if self.low_conf is None:
            return strong
        recent = [m for m in self._memory if now - m[2] <= self.hysteresis_seconds]
        memory = [[n, b, now] for n, _c, b in strong]
        out = list(strong)
        for name, conf, box in raw:
            if conf >= self.conf_thresh:
                continue
            if any(n == name and _iou(box, b) >= 0.5 for n, _c, b in strong):
                continue                                   # a weaker duplicate of a clear detection
            match = next((m for m in recent if m[0] == name and _iou(box, m[1]) >= 0.45), None)
            if match is not None:
                out.append((name, conf, box))
                memory.append([name, box, match[2]])       # follows the object, but the clock is not reset
                recent.remove(match)
        # Objects seen clearly a moment ago stay remembered through a frame where they are missed
        memory += [m for m in recent if not any(n == m[0] and _iou(m[1], b) >= 0.45 for n, b, _t in memory)]
        self._memory = memory
        return out

    def _prepare_frame(self, frame):
        """
        Returns (frame for the detector, low_light flag). The flag is always measured, so the app
        can suggest more light. Brightening the frame (gamma + CLAHE) is optional and off by default:
        on darkened COCO images it amplified sensor noise and lowered recall instead of raising it.
        """
        thumb = cv2.resize(frame, (64, 36), interpolation=cv2.INTER_AREA)
        luma = float(cv2.cvtColor(thumb, cv2.COLOR_BGR2GRAY).mean())
        if luma >= self.low_light_luma:
            return frame, False
        if not self.enhance_low_light:
            return frame, True
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        gamma = math.log(115 / 255.0) / math.log(max(luma, 4.0) / 255.0)     # lifts the mean to ~115
        gamma = max(0.35, min(1.0, gamma))
        lut = (np.power(np.arange(256) / 255.0, gamma) * 255.0).astype(np.uint8)
        light = cv2.LUT(lab[:, :, 0], lut)
        if self._clahe is None:
            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = self._clahe.apply(light)
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR), True

    def classify_severity(self, class_name, distance_m, spatial, is_user=False):
        """'hazard' | 'caution' | 'safe' for an object of this class at this distance and side."""
        if is_user or class_name not in OBSTACLES:
            return "safe"
        if distance_m <= config.DISTANCE_HAZARD_METERS and (spatial == "ahead" or class_name in config.PRIORITY_OBSTACLES):
            return "hazard"
        if distance_m <= config.DISTANCE_CAUTION_METERS:
            return "caution"
        return "safe"

    def refresh_severity(self, detections):
        """Re-derives severity after the tracker has smoothed the distances; returns the new counts."""
        for d in detections:
            if "is_obstacle" in d:
                d["severity"] = self.classify_severity(d["class"], d["distance_m"], d["spatial"], d.get("is_user", False))
        detections.sort(key=lambda d: d["distance_m"])
        return {
            "hazard_count": sum(1 for d in detections if d["severity"] == "hazard"),
            "caution_count": sum(1 for d in detections if d["severity"] == "caution"),
            "nearest": detections[0] if detections else None,
        }

    # ------------------------------------------------------------------ inference
    def _predict(self, model, frame, conf):
        results = model.predict(source=frame, imgsz=self._active_imgsz or self.imgsz, conf=conf,
                                iou=self.iou_thresh, max_det=60, verbose=False)
        raw = []
        if len(results) > 0 and results[0].boxes is not None:
            for box in results[0].boxes:
                cls_id = int(box.cls[0].item())
                x1, y1, x2, y2 = (int(v) for v in box.xyxy[0].cpu().numpy())
                raw.append((model.names[cls_id], float(box.conf[0].item()), [x1, y1, x2, y2]))
        return raw

    def _run_extended(self, frame, primary):
        """Runs the extended model and cleans its output against the main detections."""
        raw = self._merge_fragments(self._predict(self.ext_model, frame, self.ext_conf))
        people = [b for n, _c, b in primary if n == "person"]
        cleaned = []
        for name, conf, box in raw:
            if conf < self.ext_class_conf.get(name, 0.0):
                continue
            # Clothing / accessories on a person are not separate objects in the room
            if name in WORN_ITEMS and any(_intersection(box, p) >= 0.6 * max(1, _box_area(box)) for p in people):
                continue
            # Same object already found by the main detector
            twin = EXTENDED_DUPLICATES.get(name)
            if any(_iou(box, b) > 0.5 and (twin is None or n == twin or _iou(box, b) > 0.7) for n, _c, b in primary):
                continue
            # Confirmed when seen in the previous extended run too, or confident on its own
            confirmed = conf >= 0.6 or any(n == name and _iou(box, b) > 0.3 for n, _c, b in self._ext_previous)
            cleaned.append((name, conf, box, confirmed))
        self._ext_previous = [(n, c, b) for n, c, b, _ok in cleaned]
        return cleaned

    @staticmethod
    def _scene_signature(frame):
        """
        A 16x12 grey thumbnail: enough to tell 'same view' from 'the camera turned to something else'.
        Its average is removed, so the camera's automatic exposure does not count as a new view.
        """
        small = cv2.resize(frame, (16, 12), interpolation=cv2.INTER_AREA)
        grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
        return grey - grey.mean()

    def detect(self, frame, extended=None, camera_facing=None, quality=None):
        """
        Runs object detection on a BGR OpenCV image frame.
        extended: True = always run the extended model, False = never,
                  None = every Nth frame (results are reused on the frames in between).
        camera_facing: "user" for a front/laptop camera. Left/right are then reported from the
                  user's point of view (the picture is mirrored), and the large person closest to the
                  camera is marked as the user.
        quality: "fast" | "mobile" | "balanced" | "high" inference resolution (see QUALITY_IMAGE_SIZES).
        Returns:
            annotated_frame: np.ndarray with bounding boxes and proximity labels
            detections: list of dicts with detected item details (nearest first)
            summary_stats: summary metrics (hazard count, nearest obstacle, etc.)
        """
        empty_stats = {"total_detections": 0, "hazard_count": 0, "caution_count": 0, "nearest": None}
        if frame is None or self.model is None:
            return frame, [], empty_stats
        with self._lock:
            return self._detect(frame, extended, camera_facing, quality)

    def _detect(self, frame, extended, camera_facing, quality):
        h, w = frame.shape[:2]
        now = time.time()
        # A new camera or frame size is a new view: what was seen before does not carry over
        view = (w, h, camera_facing)
        if view != self._view:
            self._view, self._memory = view, []
        self._active_imgsz = QUALITY_IMAGE_SIZES.get(quality)
        model_frame, low_light = self._prepare_frame(frame)

        use_fast = quality in LIGHT_MODEL_QUALITIES and self.fast_model is not None
        model, model_name = (self.fast_model, self.fast_model_name) if use_fast else (self.model, self.model_name)
        raw = self._predict(model, model_frame, self.low_conf or self.conf_thresh)
        primary = self._merge_fragments(self._suppress_cross_class(self._apply_hysteresis(raw, now)))
        # Ties, bags etc. carried by a person are part of that person, not separate things to announce
        people_boxes = [b for n, _c, b in primary if n == "person"]
        primary = [(n, c, b) for n, c, b in primary
                   if not (n in WORN_ITEMS and any(_intersection(b, p) >= 0.6 * max(1, _box_area(b)) for p in people_boxes))]

        # Extended vocabulary: run now, or reuse the recent result while the camera still sees the
        # same view (after a turn, objects from the old view must not be announced)
        ext = []
        self._frame_counter += 1
        if self.ext_model is not None and extended is not False:
            every = self.ext_every * (2 if quality in LIGHT_MODEL_QUALITIES else 1)
            signature = self._scene_signature(model_frame)
            same_view = (self._ext_signature is not None and self._ext_signature.shape == signature.shape
                         and float(np.abs(signature - self._ext_signature).mean())
                         <= getattr(config, "EXTENDED_SCENE_CHANGE", 26.0))
            if extended or self._ext_due or self._frame_counter % every == 1 or every == 1:
                ext = self._run_extended(model_frame, primary)
                self._ext_cache = (now, ext)
                self._ext_signature, self._ext_due = signature, False
            elif same_view and now - self._ext_cache[0] <= self.ext_cache_s:
                ext = self._ext_cache[1]
            elif not same_view:
                self._ext_due = True

        mirrored = camera_facing == "user"
        focal = focal_length_px(w, h)
        user_box = None
        if mirrored:
            people = [(_box_area(b), b) for n, c, b in primary if n == "person" and c >= 0.5]
            if people:
                area, box = max(people, key=lambda p: p[0])
                if area >= 0.12 * w * h:
                    user_box = box
                    # Weaker "person" boxes overlapping the user, or headless boxes cut off by the
                    # bottom edge right in front of the lens, are the user's own arms or knees
                    def _own_body_part(b):
                        overlaps = _intersection(b, user_box) >= 0.25 * max(1, _box_area(b))
                        headless = b[3] >= h * 0.975 and b[1] >= h * 0.5
                        return overlaps or headless
                    primary = [(n, c, b) for n, c, b in primary
                               if b == user_box or n != "person" or c >= 0.6 or not _own_body_part(b)]

        items = [(n, c, b, "main", True) for n, c, b in primary] + [(n, c, b, "extended", ok) for n, c, b, ok in ext]
        detections = []
        annotated_frame = frame.copy()
        for class_name, conf, (x1, y1, x2, y2), source, confirmed in items:
            bbox_h = max(1, y2 - y1)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)

            spatial = self.get_spatial_position(cx, w)
            if mirrored and spatial != "ahead":
                spatial = "right" if spatial == "left" else "left"
            # Direction from straight ahead in degrees, from the user's point of view (+ = right)
            bearing = math.degrees(math.atan((cx - w / 2.0) / focal))
            if mirrored:
                bearing = -bearing
            distance_m = self.estimate_distance(class_name, bbox_h, h, bbox=(x1, y1, x2, y2), frame_width=w)
            is_user = user_box is not None and [x1, y1, x2, y2] == user_box

            is_priority = class_name in config.PRIORITY_OBSTACLES
            is_obstacle = class_name in OBSTACLES and not is_user
            severity = self.classify_severity(class_name, distance_m, spatial, is_user)
            color, bg_color = {"hazard": ((0, 0, 220), (0, 0, 180)),
                               "caution": ((0, 165, 255), (0, 140, 220))}.get(severity, ((0, 220, 100), (0, 180, 80)))

            det = {
                "class": class_name,
                "confidence": round(conf, 2),
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "center": [cx, cy],
                "spatial": spatial,
                "bearing_deg": round(bearing, 1),
                "distance_m": distance_m,
                "severity": severity,
                "is_priority": is_priority,
                "is_obstacle": is_obstacle,
                "source": source,
            }
            if source == "extended":
                det["confirmed"] = bool(confirmed)
            if is_user:
                det["is_user"] = True
            detections.append(det)

            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
            label = f"{'You' if is_user else class_name.capitalize()} {conf:.0%} | {spatial.capitalize()} ~{distance_m}m"
            font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
            (lbl_w, lbl_h), _ = cv2.getTextSize(label, font, font_scale, font_thick)
            tag_y1 = max(0, y1 - lbl_h - 8)
            cv2.rectangle(annotated_frame, (x1, tag_y1), (min(w, x1 + lbl_w + 12), y1), bg_color, -1)
            cv2.putText(annotated_frame, label, (x1 + 6, y1 - 5), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        detections.sort(key=lambda d: d["distance_m"])
        summary_stats = {
            "total_detections": len(detections),
            "hazard_count": sum(1 for d in detections if d["severity"] == "hazard"),
            "caution_count": sum(1 for d in detections if d["severity"] == "caution"),
            "nearest": detections[0] if detections else None,
            "model": model_name,
            "extended_model": self.ext_model_name,
            "low_light": low_light,
            "image_size": self._active_imgsz or self.imgsz,
        }
        self._draw_hud(annotated_frame, summary_stats)
        return annotated_frame, detections, summary_stats

    def _draw_hud(self, frame, stats):
        """Draws top HUD status banner with system indicator and hazard alerts."""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 36), (25, 30, 36), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, "AMCS VISION", (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 240, 255), 2, cv2.LINE_AA)
        if stats["hazard_count"] > 0:
            cv2.putText(frame, f"ALERT: {stats['hazard_count']} IMMINENT HAZARD(S)!", (160, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        elif stats["nearest"]:
            n = stats["nearest"]
            cv2.putText(frame, f"Nearest: {n['class']} ({n['spatial']}) at {n['distance_m']}m", (160, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (200, 230, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "Path Clear - No Obstacles Detected", (160, 23),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1, cv2.LINE_AA)

    def decode_base64_image(self, base64_data):
        """Decodes base64 data string (from HTML canvas/webcam) to OpenCV BGR image."""
        try:
            if "," in base64_data:
                base64_data = base64_data.split(",")[1]
            np_arr = np.frombuffer(base64.b64decode(base64_data), np.uint8)
            return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        except Exception as e:
            print(f"[VisionDetector] Error decoding base64 image: {e}")
            return None

    def encode_base64_image(self, frame):
        """Encodes an OpenCV BGR image to base64 JPEG string for browser display."""
        try:
            ret, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            if not ret:
                return None
            return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"[VisionDetector] Error encoding frame to base64: {e}")
            return None
