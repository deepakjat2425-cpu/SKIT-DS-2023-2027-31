"""
Unit & Integration Tests for New Features:
- Indian Currency Note Recognition (₹10, ₹20, ₹50, ₹100, ₹200, ₹500)
- Caregiver WhatsApp / Google Maps Emergency Dispatch
- Caregiver Daily Mobility & Safety Analytics Export
"""

import unittest
import json
import base64
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from modules.currency_recognizer import CurrencyRecognizer
from modules.location_tracker import LocationTracker

class TestNewFeatures(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.currency = CurrencyRecognizer()
        self.tracker = LocationTracker()

        # Generate a test stone-grey image (simulating ₹500 note)
        dummy_note = np.full((300, 500, 3), (120, 125, 130), dtype=np.uint8)
        _, buffer = cv2.imencode('.jpg', dummy_note)
        self.test_note_b64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    def test_currency_text_extraction(self):
        """Test recognition of denomination from tokens."""
        denom, conf = self.currency.extract_denomination_text("Reserve Bank of India 500 Rupees")
        self.assertEqual(denom, 500)
        self.assertGreater(conf, 0.8)

        denom200, _ = self.currency.extract_denomination_text("RBI ₹200 DO SAU RUPEES")
        self.assertEqual(denom200, 200)

        denom100, _ = self.currency.extract_denomination_text("Mahatma Gandhi 100")
        self.assertEqual(denom100, 100)

    def test_api_currency_detect(self):
        """Verify dedicated /api/currency_detect endpoint."""
        res = self.app.post("/api/currency_detect",
                            data=json.dumps({
                                "image": self.test_note_b64,
                                "lang": "hi"
                            }),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("result", data)
        self.assertIn("spoken_text", data)

    def test_caregiver_emergency_dispatch(self):
        """Verify SOS trigger returns Google Maps and WhatsApp URLs."""
        res = self.app.post("/api/sos/trigger",
                            data=json.dumps({"message": "Viva Test Emergency"}),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("maps_url", data)
        self.assertIn("whatsapp_url", data)
        self.assertIn("maps.google.com", data["maps_url"])
        self.assertIn("whatsapp.com", data["whatsapp_url"])

    def test_caregiver_mobility_report(self):
        """Verify daily mobility report generation (HTML and JSON)."""
        # 1. HTML printable report
        res_html = self.app.get("/api/caregiver/export_report")
        self.assertEqual(res_html.status_code, 200)
        self.assertIn("Daily Mobility", res_html.get_data(as_text=True))

        # 2. JSON data
        res_json = self.app.get("/api/caregiver/export_report?format=json")
        self.assertEqual(res_json.status_code, 200)
        data = json.loads(res_json.data)
        self.assertIn("total_distance_m", data)
        self.assertIn("duration_minutes", data)
        self.assertIn("waypoints_recorded", data)

    def test_tracker_distance_calculation(self):
        """Verify Haversine accumulated distance calculation."""
        # Add two points ~111 meters apart
        self.tracker.update_location(26.8222, 75.8653)
        self.tracker.update_location(26.8232, 75.8653)
        dist = self.tracker.calculate_total_distance_m()
        self.assertGreater(dist, 50)
        analytics = self.tracker.get_safety_analytics()
        self.assertIn("total_distance_m", analytics)

if __name__ == "__main__":
    unittest.main()
