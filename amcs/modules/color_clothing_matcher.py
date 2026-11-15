"""
Color & Clothing Wardrobe Matcher Engine.
Enables independent dressing, garment color identification, pattern analysis,
and outfit styling compatibility for visually impaired individuals.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import cv2
import numpy as np
import math

# Reference named color palette in Lab space for perceptual distance calculation
COLOR_PALETTE = {
    "Black": (0, 0, 0),
    "Charcoal Grey": (54, 54, 54),
    "Light Grey": (180, 180, 180),
    "White": (245, 245, 245),
    "Navy Blue": (15, 30, 80),
    "Royal Blue": (30, 70, 190),
    "Sky Blue": (135, 206, 235),
    "Teal": (0, 128, 128),
    "Forest Green": (34, 110, 50),
    "Olive Green": (107, 120, 45),
    "Crimson Red": (180, 20, 30),
    "Maroon / Burgundy": (90, 15, 25),
    "Pink": (255, 140, 175),
    "Yellow": (240, 215, 30),
    "Orange": (240, 120, 20),
    "Beige": (225, 210, 185),
    "Khaki / Tan": (175, 155, 115),
    "Brown": (100, 60, 30),
    "Purple": (115, 45, 140),
    "Lavender": (190, 160, 220)
}

# Outfit Pairing Compatibility Matrix
PAIRING_ADVICE = {
    "Navy Blue": "Navy blue is classic and versatile. Pairs wonderfully with beige, khaki, light grey, or crisp white.",
    "White": "White is universally compatible. Matches seamlessly with any dark, neutral, or vibrant colored garment.",
    "Black": "Black provides a sharp, sleek contrast. Pairs effortlessly with grey, white, red, beige, or denim.",
    "Charcoal Grey": "Charcoal grey is modern and formal. Pairs best with navy blue, black, white, or burgundy.",
    "Light Grey": "Light grey pairs nicely with dark trousers like navy blue, black, or deep forest green.",
    "Sky Blue": "Sky blue looks fresh when paired with navy blue, khaki, beige, or charcoal grey.",
    "Beige": "Beige is warm and neutral. Matches exceptionally well with navy blue, brown, forest green, or white.",
    "Khaki / Tan": "Khaki pairs ideally with dark navy blue, white, black, or olive green.",
    "Forest Green": "Forest green pairs smartly with khaki, beige, navy, or black.",
    "Crimson Red": "Crimson red makes a bold statement. Pair with neutral black, grey, or white to balance the outfit.",
    "Maroon / Burgundy": "Burgundy pairs sophisticatedly with grey, black, khaki, or navy.",
    "Brown": "Brown coordinates naturally with beige, cream, white, or light blue.",
    "Yellow": "Yellow pairs best with navy blue, charcoal grey, or black.",
    "Purple": "Purple matches well with grey, black, or white neutrals."
}
DEFAULT_PAIRING = "This garment pairs best with solid neutral bottoms such as black, grey, or navy."

class ColorClothingMatcher:
    def __init__(self):
        pass

    def _color_distance_sq(self, c1, c2):
        """Euclidean distance squared between two RGB colors."""
        return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2 + (c1[2] - c2[2])**2

    def identify_closest_color(self, rgb):
        """Finds the closest human named color from RGB."""
        best_name = "Black"
        min_dist = float("inf")
        for name, ref_rgb in COLOR_PALETTE.items():
            dist = self._color_distance_sq(rgb, ref_rgb)
            if dist < min_dist:
                min_dist = dist
                best_name = name
        return best_name

    def detect_pattern(self, gray_roi):
        """Detects whether fabric is solid, striped, checkered, or printed."""
        # Sobel gradients in X and Y
        sobel_x = cv2.Sobel(gray_roi, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray_roi, cv2.CV_64F, 0, 1, ksize=3)

        std_x = np.std(sobel_x)
        std_y = np.std(sobel_y)
        edge_variance = (std_x + std_y) / 2.0

        if edge_variance < 14.0:
            return "Solid (Plain)"
        elif abs(std_x - std_y) > 12.0:
            return "Striped"
        elif edge_variance > 32.0 and abs(std_x - std_y) < 10.0:
            return "Checkered / Plaid"
        else:
            return "Patterned / Textured"

    def analyze_clothing(self, frame):
        """
        Analyzes central clothing region for dominant colors, patterns, and styling advice.
        """
        if frame is None:
            return None

        h, w = frame.shape[:2]
        # Crop central 50% ROI where user holds the garment
        cy1, cy2 = int(h * 0.25), int(h * 0.75)
        cx1, cx2 = int(w * 0.25), int(w * 0.75)
        roi = frame[cy1:cy2, cx1:cx2]

        # Convert to RGB and Grayscale
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Average color in central patch (excluding extreme shadows/specular highlights)
        mid_h, mid_w = roi_rgb.shape[:2]
        center_patch = roi_rgb[int(mid_h*0.2):int(mid_h*0.8), int(mid_w*0.2):int(mid_w*0.8)]
        avg_r = int(np.median(center_patch[:, :, 0]))
        avg_g = int(np.median(center_patch[:, :, 1]))
        avg_b = int(np.median(center_patch[:, :, 2]))

        primary_color = self.identify_closest_color((avg_r, avg_g, avg_b))
        hex_code = f"#{avg_r:02x}{avg_g:02x}{avg_b:02x}"

        pattern = self.detect_pattern(roi_gray)
        pairing_advice = PAIRING_ADVICE.get(primary_color, DEFAULT_PAIRING)

        summary_en = f"Garment is {primary_color} with {pattern} fabric. {pairing_advice}"

        return {
            "primary_color": primary_color,
            "hex_code": hex_code,
            "rgb": [avg_r, avg_g, avg_b],
            "pattern": pattern,
            "pairing_advice_en": pairing_advice,
            "summary_en": summary_en
        }
