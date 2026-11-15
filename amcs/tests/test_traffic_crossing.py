"""
Unit tests for Traffic Light & Pedestrian Signal Assistant.
"""

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.traffic_crossing_assistant import TrafficCrossingAssistant

class TestTrafficCrossingAssistant(unittest.TestCase):
    def setUp(self):
        self.assistant = TrafficCrossingAssistant()

    def test_red_traffic_light(self):
        """Simulate a red traffic signal and verify detection."""
        tl = np.zeros((90, 30, 3), dtype=np.uint8)
        # Top third: Red circle
        cv2.circle(tl, (15, 15), 10, (0, 0, 240), -1)
        res = self.assistant.analyze_traffic_light(tl)
        self.assertEqual(res["state"], "RED")
        self.assertFalse(res["safe_to_cross"])

    def test_green_traffic_light(self):
        """Simulate a green traffic signal and verify detection."""
        tl = np.zeros((90, 30, 3), dtype=np.uint8)
        # Bottom third: Green circle
        cv2.circle(tl, (15, 75), 10, (0, 240, 0), -1)
        res = self.assistant.analyze_traffic_light(tl)
        self.assertEqual(res["state"], "GREEN")
        self.assertTrue(res["safe_to_cross"])

    def test_yellow_traffic_light(self):
        """Simulate a yellow traffic signal and verify caution detection."""
        tl = np.zeros((90, 30, 3), dtype=np.uint8)
        # Middle third: Yellow circle
        cv2.circle(tl, (15, 45), 10, (0, 220, 220), -1)
        res = self.assistant.analyze_traffic_light(tl)
        self.assertEqual(res["state"], "YELLOW")
        self.assertFalse(res["safe_to_cross"])

    def test_intersection_guidance_synthesis(self):
        """Verify intersection guidance synthesis."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add traffic light in frame
        cv2.circle(frame, (100, 100), 15, (0, 0, 240), -1)
        detections = [{
            "class": "traffic light",
            "bbox": [80, 70, 120, 160],
            "distance_m": 8.0,
            "spatial": "ahead"
        }]
        res = self.assistant.analyze_intersection(frame, detections)
        self.assertTrue(res["is_intersection"])
        self.assertIn("RED", res["advisory_en"])
        self.assertFalse(res["safe_to_cross"])

if __name__ == '__main__':
    unittest.main()
