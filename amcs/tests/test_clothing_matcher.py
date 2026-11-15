"""
Unit Tests for ColorClothingMatcher (Color & Clothing Wardrobe Matcher).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
import numpy as np
from modules.color_clothing_matcher import ColorClothingMatcher

class TestColorClothingMatcher(unittest.TestCase):
    def setUp(self):
        self.matcher = ColorClothingMatcher()

    def test_solid_navy_blue(self):
        # Create synthetic Navy Blue patch (BGR: 70, 25, 15)
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[:, :] = (70, 25, 15)  # RGB is (15, 25, 70)

        res = self.matcher.analyze_clothing(frame)
        self.assertIsNotNone(res)
        self.assertEqual(res["primary_color"], "Navy Blue")
        self.assertEqual(res["pattern"], "Solid (Plain)")
        self.assertIn("beige", res["pairing_advice_en"].lower())

    def test_solid_crimson_red(self):
        frame = np.zeros((300, 300, 3), dtype=np.uint8)
        frame[:, :] = (20, 20, 180)  # RGB is (180, 20, 20)

        res = self.matcher.analyze_clothing(frame)
        self.assertIsNotNone(res)
        self.assertEqual(res["primary_color"], "Crimson Red")

    def test_striped_pattern(self):
        # Create synthetic vertical striped pattern
        frame = np.ones((300, 300, 3), dtype=np.uint8) * 240
        # Draw high-contrast dark vertical stripes every 20 pixels
        for x in range(0, 300, 20):
            frame[:, x:x+8] = (20, 20, 20)

        res = self.matcher.analyze_clothing(frame)
        self.assertIsNotNone(res)
        self.assertEqual(res["pattern"], "Striped")


class TestClothingAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_match_clothing(self):
        import cv2, base64
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:, :] = (70, 25, 15)  # Navy blue BGR
        _, buf = cv2.imencode('.jpg', img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        res = self.client.post("/api/match_clothing", json={"image": b64})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Navy Blue", data["analysis"]["primary_color"])


if __name__ == "__main__":
    unittest.main()
