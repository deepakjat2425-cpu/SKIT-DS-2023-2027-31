"""
Item Finder & Acoustic Audio Radar Engine ("Find My Item").
Functions like a visual & acoustic Geiger counter to guide visually impaired users
to misplaced personal belongings (keys, phone, bottle, backpack, cane, etc.).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import math
import numpy as np
import cv2

# Universal item aliases and category mapping
ITEM_CATEGORY_MAP = {
    "phone": ["cell phone", "phone", "mobile", "smartphone", "iphone", "android"],
    "bottle": ["bottle", "water bottle", "flask", "thermos"],
    "cup": ["cup", "mug", "glass", "wine glass", "chai cup", "tumbler"],
    "backpack": ["backpack", "bag", "school bag", "satchel"],
    "handbag": ["handbag", "purse"],
    "wallet": ["wallet", "money purse", "batua"],
    "laptop": ["laptop", "computer"],
    "book": ["book", "notebook", "register"],
    "chair": ["chair", "seat", "stool"],
    "spectacles": ["spectacles", "glasses", "sunglasses", "chashma", "specs"],
    "keys": ["keys", "key", "keychain", "chabi"],
    "pen": ["pen", "pencil", "marker"],
    "remote": ["remote", "remote control", "tv remote"],
    "charger": ["charger", "phone charger", "adapter"],
    "medicine": ["medicine", "tablets", "pills", "dawai", "dawa"],
    "watch": ["watch", "wrist watch", "ghadi"],
    "headphones": ["headphones", "earphones", "earbuds", "headset"],
    "shoe": ["shoe", "shoes", "slipper", "slippers", "chappal", "sandal"],
    "umbrella": ["umbrella", "chhata"],
}

class ItemFinderEngine:
    def __init__(self):
        self.active_target = None
        self.consecutive_sightings = 0
        # Shiny-contour guess for keys/glasses. Turned off by the app when the extended-vocabulary
        # detector is available, because that model recognises them properly.
        self.use_heuristic_fallback = True

    def resolve_target_class(self, query):
        """Resolves user search query to standard detector classes."""
        q = " " + " ".join(query.strip().lower().replace("?", " ").split()) + " "
        # Longest whole-word alias wins, so "glasses" is spectacles rather than a drinking glass
        best = None
        for primary_cls, aliases in ITEM_CATEGORY_MAP.items():
            for a in [primary_cls] + aliases:
                if f" {a} " in q and (best is None or len(a) > best[0]):
                    best = (len(a), primary_cls, aliases)
        if best:
            return best[1], best[2]
        q = q.strip()
        return q, [q]

    def search_frame_for_item(self, frame, detections, target_query):
        """
        Scans detections and frame features for the requested item.
        Calculates:
            - found: bool
            - alignment_score: 0.0 (edge) to 1.0 (dead center in crosshair)
            - proximity_score: 0.0 (far) to 1.0 (within arm's reach <= 0.6m)
            - ping_frequency_hz: Audio radar frequency for client
            - spatial_pan: -1.0 (full left) to +1.0 (full right)
            - spoken_guidance: Natural navigational steering cue
        """
        if not target_query:
            return {"found": False, "target": None, "message": "No item specified."}

        primary_cls, aliases = self.resolve_target_class(target_query)
        self.active_target = primary_cls

        if frame is None:
            return {"found": False, "target": primary_cls, "message": "Frame not available."}

        h, w = frame.shape[:2]
        center_x, center_y = w / 2.0, h / 2.0

        # 1. Check detections from YOLOv8
        matching_detections = []
        for d in detections:
            cls_lower = d.get("class", "").lower()
            if cls_lower == primary_cls or cls_lower in aliases:
                matching_detections.append(d)

        # 2. Heuristic fallback for keys / small reflective items if target is 'keys'
        if not matching_detections and self.use_heuristic_fallback and primary_cls in ["keys", "spectacles"]:
            keys_det = self._detect_small_metallic_contours(frame, center_x, center_y)
            if keys_det:
                matching_detections.append(keys_det)

        if not matching_detections:
            self.consecutive_sightings = 0
            return {
                "found": False,
                "target": primary_cls,
                "radar_active": True,
                "ping_rate_hz": 0.8,         # Slow searching tick
                "ping_pitch_hz": 380,
                "spatial_pan": 0.0,
                "alignment_score": 0.0,
                "proximity_score": 0.0,
                "guidance": f"Scanning for {primary_cls}. Sweep camera slowly around the room."
            }

        # Select closest / most central matching object
        matching_detections.sort(key=lambda d: d.get("distance_m", 99))
        best_match = matching_detections[0]
        self.consecutive_sightings += 1

        bx1, by1, bx2, by2 = best_match["bbox"]
        obj_cx = (bx1 + bx2) / 2.0
        obj_cy = (by1 + by2) / 2.0
        dist_m = best_match.get("distance_m", 1.5)

        # Stereo Pan: -1.0 (left) to +1.0 (right)
        spatial_pan = round((obj_cx - center_x) / (center_x + 1e-5), 2)
        spatial_pan = max(-1.0, min(1.0, spatial_pan))

        # Alignment score: how close to dead center
        dist_from_center = math.hypot(obj_cx - center_x, obj_cy - center_y)
        max_dist = math.hypot(center_x, center_y)
        alignment_score = round(max(0.0, 1.0 - (dist_from_center / max_dist)), 2)

        # Proximity score: <= 0.5m is 1.0, >= 3.0m is 0.0
        proximity_score = round(max(0.0, min(1.0, (3.0 - dist_m) / 2.5)), 2)

        # Radar Audio Modulation:
        # Rate increases from 1.5 Hz to 9 Hz as user centers on item
        ping_rate = round(1.5 + (alignment_score * 4.5) + (proximity_score * 3.0), 1)
        # Pitch increases from 550 Hz to 1350 Hz
        ping_pitch = int(550 + (alignment_score * 500) + (proximity_score * 300))

        # Steering Verbal Direction
        if alignment_score >= 0.82:
            direction_cue = "Straight ahead in your center!"
        elif spatial_pan < -0.3:
            direction_cue = "Turn camera slightly left."
        elif spatial_pan > 0.3:
            direction_cue = "Turn camera slightly right."
        elif obj_cy < center_y - 80:
            direction_cue = "Tilt camera slightly up."
        else:
            direction_cue = "Tilt camera slightly down."

        guidance = f"{primary_cls.capitalize()} located ~{dist_m}m. {direction_cue}"

        return {
            "found": True,
            "target": primary_cls,
            "target_bbox": best_match["bbox"],
            "distance_m": dist_m,
            "spatial": best_match.get("spatial", "ahead"),
            "spatial_pan": spatial_pan,
            "alignment_score": alignment_score,
            "proximity_score": proximity_score,
            "ping_rate_hz": ping_rate,
            "ping_pitch_hz": ping_pitch,
            "is_centered": alignment_score >= 0.82,
            "guidance": guidance
        }

    def _detect_small_metallic_contours(self, frame, center_x, center_y):
        """Heuristic detector for small high-contrast shiny items like keys or glasses on surfaces."""
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 60, 180)

        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            area = cv2.contourArea(c)
            if 150 <= area <= 2800:
                x, y, bw, bh = cv2.boundingRect(c)
                aspect = bw / float(bh) if bh > 0 else 0
                if 0.3 <= aspect <= 3.0:
                    candidates.append((x, y, bw, bh, area))

        if candidates:
            # Pick candidate closest to center
            candidates.sort(key=lambda item: math.hypot((item[0] + item[2]/2) - center_x,
                                                       (item[1] + item[3]/2) - center_y))
            x, y, bw, bh, _ = candidates[0]
            cx = x + bw // 2
            cy = y + bh // 2
            spatial = "left" if cx < w * 0.35 else ("right" if cx > w * 0.65 else "ahead")
            return {
                "class": "keys",
                "confidence": 0.72,
                "bbox": [x, y, x + bw, y + bh],
                "center": [cx, cy],
                "spatial": spatial,
                "distance_m": 0.8,
                "severity": "safe"
            }
        return None
