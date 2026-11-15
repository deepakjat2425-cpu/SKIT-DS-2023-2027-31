"""
Indian Currency Recognition Module for Visually Impaired Users.
Identifies Mahatma Gandhi New & Standard Series Banknotes (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
using HSV color histogram profiling and OCR denomination verification.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import re
import config

# Color profiles for Indian Banknotes (Mahatma Gandhi New Series) in HSV color space
# Hue: 0-180, Sat: 0-255, Val: 0-255
NOTE_PROFILES = {
    500: {
        "name": "Five Hundred Rupees",
        "hindi_name": "पाँच सौ रुपये",
        "color_name": "Stone Grey",
        # Greyish with subtle olive/green undertones: low saturation, medium-high value
        "hsv_ranges": [
            ((40, 20, 60), (95, 120, 200)),
            ((0, 0, 80), (180, 50, 210))
        ],
        "keywords": ["500", "५००", "five hundred", "paanch sau"]
    },
    200: {
        "name": "Two Hundred Rupees",
        "hindi_name": "दो सौ रुपये",
        "color_name": "Bright Yellow-Orange",
        # Warm golden yellow / orange: Hue 18 to 36, high saturation
        "hsv_ranges": [
            ((16, 120, 150), (34, 255, 255))
        ],
        "keywords": ["200", "२००", "two hundred", "do sau"]
    },
    100: {
        "name": "One Hundred Rupees",
        "hindi_name": "एक सौ रुपये",
        "color_name": "Lavender Purple",
        # Distinctive lavender / violet: Hue 125 to 160
        "hsv_ranges": [
            ((120, 40, 90), (160, 220, 240))
        ],
        "keywords": ["100", "१००", "one hundred", "sau", "ek sau"]
    },
    50: {
        "name": "Fifty Rupees",
        "hindi_name": "पचास रुपये",
        "color_name": "Fluorescent Cyan Blue",
        # Vivid cyan / turquoise: Hue 85 to 118, medium-high saturation
        "hsv_ranges": [
            ((85, 80, 100), (118, 255, 255))
        ],
        "keywords": ["50", "५०", "fifty", "pachaas"]
    },
    20: {
        "name": "Twenty Rupees",
        "hindi_name": "बीस रुपये",
        "color_name": "Greenish Yellow",
        # Yellowish-green: Hue 35 to 65
        "hsv_ranges": [
            ((35, 70, 90), (68, 240, 240))
        ],
        "keywords": ["20", "२०", "twenty", "bees"]
    },
    10: {
        "name": "Ten Rupees",
        "hindi_name": "दस रुपये",
        "color_name": "Chocolate Brown",
        # Deep terracotta brown: Hue 8 to 22, medium-high saturation, darker value
        "hsv_ranges": [
            ((6, 70, 40), (22, 230, 160))
        ],
        "keywords": ["10", "१०", "ten", "das"]
    }
}

class CurrencyRecognizer:
    def __init__(self):
        pass

    def analyze_colors(self, hsv_crop):
        """Calculates color match ratios for each banknote profile."""
        total_pixels = max(1, hsv_crop.shape[0] * hsv_crop.shape[1])
        scores = {}

        for denom, profile in NOTE_PROFILES.items():
            total_mask = np.zeros(hsv_crop.shape[:2], dtype=np.uint8)
            for (lower, upper) in profile["hsv_ranges"]:
                mask = cv2.inRange(hsv_crop, np.array(lower), np.array(upper))
                total_mask = cv2.bitwise_or(total_mask, mask)

            match_ratio = np.count_nonzero(total_mask) / float(total_pixels)
            scores[denom] = match_ratio

        return scores

    def extract_denomination_text(self, text):
        """Scans extracted OCR text for explicit numerical denomination markers."""
        if not text:
            return None, 0.0

        cleaned = text.lower().replace("₹", " ").replace("rs", " ")

        # Look for explicit numbers
        for denom, profile in NOTE_PROFILES.items():
            for kw in profile["keywords"]:
                pattern = r'(?<!\d)' + re.escape(kw) + r'(?!\d)'
                if re.search(pattern, cleaned):
                    return denom, 0.95

        # Check for numbers in string tokens
        numbers = re.findall(r'\b(500|200|100|50|20|10)\b', cleaned)
        if numbers:
            denom = int(numbers[0])
            return denom, 0.90

        return None, 0.0

    def detect(self, frame, ocr_text=""):
        """
        Detects Indian Currency note from image frame and optional OCR text.
        Returns:
            dict containing:
                - detected: bool
                - denomination: int (10, 20, 50, 100, 200, 500)
                - label: str ("Five Hundred Rupees")
                - hindi_label: str ("पाँच सौ रुपये")
                - confidence: float (0.0 to 1.0)
                - bbox: list [x, y, w, h]
                - description: str
        """
        if frame is None:
            return {"detected": False, "denomination": None, "confidence": 0.0}

        h, w = frame.shape[:2]

        # Focus analysis on central region where currency note is held (center 70%)
        cy_start, cy_end = int(h * 0.15), int(h * 0.85)
        cx_start, cx_end = int(w * 0.15), int(w * 0.85)
        central_crop = frame[cy_start:cy_end, cx_start:cx_end]

        hsv_crop = cv2.cvtColor(central_crop, cv2.COLOR_BGR2HSV)
        color_scores = self.analyze_colors(hsv_crop)

        # 1. Check OCR text for explicit denomination
        text_denom, text_conf = self.extract_denomination_text(ocr_text)

        # 2. Get best color candidate
        best_denom_color = max(color_scores, key=color_scores.get)
        best_color_ratio = color_scores[best_denom_color]

        # Combine text verification and color matching
        detected_denom = None
        confidence = 0.0

        if text_denom is not None:
            # Text was found! Check if color also supports it
            color_support = color_scores.get(text_denom, 0.0)
            detected_denom = text_denom
            confidence = min(0.98, text_conf + (0.05 if color_support > 0.10 else 0.0))
        elif best_color_ratio >= 0.20:
            # Color is strongly dominant in center
            detected_denom = best_denom_color
            confidence = round(min(0.85, 0.50 + (best_color_ratio * 0.7)), 2)

        if detected_denom is not None and confidence >= 0.50:
            meta = NOTE_PROFILES[detected_denom]
            desc_en = f"Detected {meta['name']} banknote (₹{detected_denom})."
            desc_hi = f"₹{detected_denom} का नोट पहचाना गया ({meta['hindi_name']})।"

            return {
                "detected": True,
                "denomination": detected_denom,
                "label": meta["name"],
                "hindi_label": meta["hindi_name"],
                "color_name": meta["color_name"],
                "confidence": confidence,
                "description_en": desc_en,
                "description_hi": desc_hi,
                "bbox": [cx_start, cy_start, cx_end - cx_start, cy_end - cy_start]
            }

        return {
            "detected": False,
            "denomination": None,
            "confidence": 0.0,
            "description_en": "No Indian currency banknote clearly recognized.",
            "description_hi": "कोई भारतीय नोट स्पष्ट रूप से नहीं पहचाना गया।"
        }
