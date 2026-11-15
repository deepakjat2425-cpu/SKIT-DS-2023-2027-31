"""
Traffic Light & Pedestrian Signal Navigation Assistant.
Empowers visually impaired individuals to safely navigate street intersections
by detecting pedestrian traffic signals (Red / Green / Yellow).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np

class TrafficCrossingAssistant:
    def __init__(self):
        pass

    def analyze_traffic_light(self, tl_crop):
        """
        Determines active light state of a cropped traffic light box.
        Returns:
            dict containing:
                - state: 'RED' | 'GREEN' | 'YELLOW' | 'UNKNOWN'
                - safe_to_cross: bool
                - confidence: float
                - label: str
        """
        if tl_crop is None or tl_crop.size == 0 or tl_crop.shape[0] < 12 or tl_crop.shape[1] < 6:
            return {"state": "UNKNOWN", "safe_to_cross": False, "confidence": 0.0, "label": "Traffic Signal"}

        hsv = cv2.cvtColor(tl_crop, cv2.COLOR_BGR2HSV)
        h, w = tl_crop.shape[:2]

        # Standard vertical traffic signal division:
        # Top third = Red, Middle third = Yellow, Bottom third = Green
        top_crop = hsv[0:int(h * 0.38), :]
        mid_crop = hsv[int(h * 0.32):int(h * 0.68), :]
        bot_crop = hsv[int(h * 0.62):, :]

        # 1. Red Mask (Hue 0-10 or 160-180)
        mask_r1 = cv2.inRange(top_crop, np.array([0, 100, 100]), np.array([12, 255, 255]))
        mask_r2 = cv2.inRange(top_crop, np.array([160, 100, 100]), np.array([180, 255, 255]))
        red_count = np.count_nonzero(mask_r1) + np.count_nonzero(mask_r2)

        # 2. Green Mask (Hue 40-90)
        mask_g = cv2.inRange(bot_crop, np.array([40, 90, 90]), np.array([92, 255, 255]))
        green_count = np.count_nonzero(mask_g)

        # 3. Yellow/Amber Mask (Hue 15-38)
        mask_y = cv2.inRange(mid_crop, np.array([15, 110, 110]), np.array([38, 255, 255]))
        yellow_count = np.count_nonzero(mask_y)

        # Determine dominant lighted color
        min_pixel_threshold = max(6, int((h * w) * 0.04))

        if red_count > green_count and red_count > yellow_count and red_count >= min_pixel_threshold:
            return {
                "state": "RED",
                "safe_to_cross": False,
                "confidence": 0.92,
                "label": "RED LIGHT (STOP)"
            }
        elif green_count > red_count and green_count > yellow_count and green_count >= min_pixel_threshold:
            return {
                "state": "GREEN",
                "safe_to_cross": True,
                "confidence": 0.92,
                "label": "GREEN LIGHT (WALK)"
            }
        elif yellow_count >= min_pixel_threshold and yellow_count > red_count:
            return {
                "state": "YELLOW",
                "safe_to_cross": False,
                "confidence": 0.85,
                "label": "YELLOW LIGHT (CAUTION)"
            }

        return {"state": "UNKNOWN", "safe_to_cross": False, "confidence": 0.0, "label": "Traffic Signal"}

    def analyze_intersection(self, frame, detections):
        """
        Analyzes detected traffic lights to generate pedestrian crossing advice.
        Returns:
            dict containing:
                - is_intersection: bool
                - signal_state: str ('RED', 'GREEN', 'YELLOW', 'NONE')
                - safe_to_cross: bool
                - advisory_en: str
                - advisory_hi: str
        """
        signal_state = "NONE"
        safe_to_cross = False
        tl_conf = 0.0

        # Check for traffic light detection
        for d in detections:
            if d["class"].lower() in ["traffic light", "traffic signal"]:
                x1, y1, x2, y2 = d["bbox"]
                crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
                res = self.analyze_traffic_light(crop)
                signal_state = res["state"]
                safe_to_cross = res["safe_to_cross"]
                tl_conf = res["confidence"]
                # Update detection label with signal state
                if signal_state in ["RED", "GREEN", "YELLOW"]:
                    d["class"] = f"Traffic Signal ({signal_state})"
                    d["severity"] = "hazard" if signal_state in ["RED", "YELLOW"] else "safe"
                break

        is_intersection = (signal_state != "NONE")
        advisory_en = ""
        advisory_hi = ""

        if signal_state == "RED":
            safe_to_cross = False
            advisory_en = "Pedestrian signal is RED. Please wait at the curb. Do not cross."
            advisory_hi = "पैदल यात्री सिग्नल लाल है। कृपया सड़क पार न करें, रुकें।"
        elif signal_state == "GREEN":
            safe_to_cross = True
            advisory_en = "Signal is GREEN. Safe to cross street now."
            advisory_hi = "सिग्नल हरा है। अब सड़क पार कर सकते हैं।"
        elif signal_state == "YELLOW":
            safe_to_cross = False
            advisory_en = "Signal is YELLOW. Caution: Prepare to stop."
            advisory_hi = "सिग्नल पीला है। सावधान: रुकने के लिए तैयार रहें।"

        return {
            "is_intersection": is_intersection,
            "signal_state": signal_state,
            "safe_to_cross": safe_to_cross,
            "advisory_en": advisory_en,
            "advisory_hi": advisory_hi
        }
