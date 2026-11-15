"""
Unit Tests for ItemFinderEngine ("Find My Item" Audio Radar).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
import numpy as np
import cv2
from modules.item_finder_engine import ItemFinderEngine

class TestItemFinderEngine(unittest.TestCase):
    def setUp(self):
        self.finder = ItemFinderEngine()

    def test_alias_resolution(self):
        cls, aliases = self.finder.resolve_target_class("water bottle")
        self.assertEqual(cls, "bottle")
        self.assertIn("bottle", aliases)

        cls2, _ = self.finder.resolve_target_class("mobile")
        self.assertEqual(cls2, "phone")

        cls3, _ = self.finder.resolve_target_class("chabi")
        self.assertEqual(cls3, "keys")

    def test_item_found_and_centered(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Bounding box dead center (cx = 320, cy = 240)
        detections = [{
            "class": "bottle",
            "bbox": [290, 200, 350, 280],
            "distance_m": 0.7,
            "spatial": "ahead"
        }]

        result = self.finder.search_frame_for_item(frame, detections, "bottle")
        self.assertTrue(result["found"])
        self.assertEqual(result["target"], "bottle")
        self.assertAlmostEqual(result["spatial_pan"], 0.0, delta=0.1)
        self.assertGreaterEqual(result["alignment_score"], 0.85)
        self.assertTrue(result["is_centered"])
        # High ping rate when centered and close
        self.assertGreater(result["ping_rate_hz"], 5.0)
        self.assertGreater(result["ping_pitch_hz"], 900)

    def test_item_on_left(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Bounding box on far left
        detections = [{
            "class": "cell phone",
            "bbox": [20, 200, 80, 280],
            "distance_m": 1.2,
            "spatial": "left"
        }]

        result = self.finder.search_frame_for_item(frame, detections, "phone")
        self.assertTrue(result["found"])
        self.assertLess(result["spatial_pan"], -0.4)
        self.assertIn("left", result["guidance"].lower())

    def test_item_not_found_sweep_guidance(self):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = []

        result = self.finder.search_frame_for_item(frame, detections, "bottle")
        self.assertFalse(result["found"])
        self.assertEqual(result["target"], "bottle")
        self.assertIn("sweep camera", result["guidance"].lower())
        self.assertLessEqual(result["ping_rate_hz"], 1.0)


class TestFindItemAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_find_item(self):
        # Create a small valid base64 test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import base64
        _, buf = cv2.imencode('.jpg', img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        res = self.client.post("/api/find_item", json={
            "target": "bottle",
            "image": b64
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("radar", data)
        self.assertEqual(data["radar"]["target"], "bottle")


if __name__ == "__main__":
    unittest.main()
