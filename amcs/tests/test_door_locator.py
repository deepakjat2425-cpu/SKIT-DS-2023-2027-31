"""
Unit Tests for IndoorDoorLocator (Smart Door, Handle & Room Sign Locator).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
import numpy as np
import cv2
from modules.indoor_door_locator import IndoorDoorLocator

class TestIndoorDoorLocator(unittest.TestCase):
    def setUp(self):
        self.locator = IndoorDoorLocator()

    def test_closed_door_and_handle_detection(self):
        # Create a synthetic image of a closed door with handle
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 210
        # Draw door panel: width=180, height=360 (aspect = 2.0)
        cv2.rectangle(frame, (230, 60), (410, 420), (120, 80, 50), -1)
        # Door frame edge
        cv2.rectangle(frame, (230, 60), (410, 420), (40, 30, 20), 4)
        # Handle on right side (x=385, y=240)
        cv2.circle(frame, (385, 240), 10, (240, 240, 240), -1)

        annotated, doors = self.locator.analyze_frame_for_doors(frame)
        self.assertIsNotNone(annotated)
        self.assertTrue(len(doors) > 0)
        d = doors[0]
        self.assertEqual(d["state"], "closed")
        self.assertEqual(d["handle_side"], "right")
        self.assertIn("handle is on the right", d["guidance"].lower())
        self.assertGreater(d["distance_m"], 0.5)

    def test_open_doorway_detection(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
        # Open dark corridor through doorway
        cv2.rectangle(frame, (220, 70), (420, 430), (15, 15, 20), -1)
        cv2.rectangle(frame, (220, 70), (420, 430), (50, 50, 50), 4)

        annotated, doors = self.locator.analyze_frame_for_doors(frame)
        self.assertTrue(len(doors) > 0)
        d = doors[0]
        self.assertEqual(d["state"], "open")
        self.assertIn("open doorway", d["guidance"].lower())

    def test_door_with_room_sign(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 210
        cv2.rectangle(frame, (230, 100), (410, 440), (100, 70, 40), -1)
        cv2.rectangle(frame, (230, 100), (410, 440), (30, 20, 10), 4)

        # Mock OCR items above the door
        ocr_items = [
            {
                "text": "Room 204 CSE Lab",
                "bbox": [[250, 60], [390, 60], [390, 85], [250, 85]]
            }
        ]

        annotated, doors = self.locator.analyze_frame_for_doors(frame, ocr_items=ocr_items)
        self.assertTrue(len(doors) > 0)
        d = doors[0]
        self.assertEqual(d["sign_text"], "Room 204 CSE Lab")
        self.assertIn("room 204 cse lab", d["guidance"].lower())


class TestDoorAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_scan_door(self):
        # Create a small valid base64 test image
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        import base64
        _, buf = cv2.imencode('.jpg', img)
        b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode('utf-8')

        res = self.client.post("/api/scan_door", json={
            "image": b64
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("doors", data)


if __name__ == "__main__":
    unittest.main()
