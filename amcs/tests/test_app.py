"""
Integration & End-to-End API Tests for Automatic Multilanguage Communication System.
SKIT Jaipur - Final Year Project.
"""

import unittest
import json
import base64
import cv2
import numpy as np
import os
import sys

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
import config

class TestFlaskEndpoints(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True

        # Generate a test JPEG base64 image
        dummy_img = np.zeros((300, 400, 3), dtype=np.uint8)
        # Draw a white rectangle to simulate an object
        cv2.rectangle(dummy_img, (100, 100), (200, 250), (255, 255, 255), -1)
        _, buffer = cv2.imencode('.jpg', dummy_img)
        self.test_base64 = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

    def test_pages_render(self):
        """Verify all HTML template pages render with 200 OK."""
        pages = ["/", "/caregiver", "/ocr", "/settings", "/about"]
        for p in pages:
            res = self.app.get(p)
            self.assertEqual(res.status_code, 200, f"Page {p} returned status {res.status_code}")
            self.assertIn("Automatic Multilanguage Communication System", res.get_data(as_text=True))

    def test_api_languages(self):
        """Verify languages endpoint returns supported languages."""
        res = self.app.get("/api/languages")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertIn("en", data)
        self.assertIn("hi", data)
        self.assertIn("es", data)

    def test_api_detect_vision(self):
        """Verify real-time vision-to-speech detection endpoint."""
        payload = {
            "image": self.test_base64,
            "lang": "hi",
            "mode": "vision",
            "force_speech": True
        }
        res = self.app.post("/api/detect",
                            data=json.dumps(payload),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("annotated_image", data)
        self.assertIn("description_en", data)
        self.assertIn("description_translated", data)

    def test_api_ocr_translate(self):
        """Verify signboard OCR translation endpoint."""
        payload = {
            "image": self.test_base64,
            "target_langs": ["en", "hi", "fr", "es"]
        }
        res = self.app.post("/api/ocr_translate",
                            data=json.dumps(payload),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")

    def test_api_location_flow(self):
        """Verify GPS update, geofence, and Folium map rendering."""
        # 1. Update location to SKIT Jaipur coordinates
        update_payload = {
            "latitude": 26.8222,
            "longitude": 75.8653,
            "accuracy": 8.5,
            "location_name": "SKIT Campus Jagatpura"
        }
        res_update = self.app.post("/api/location/update",
                                  data=json.dumps(update_payload),
                                  content_type="application/json")
        self.assertEqual(res_update.status_code, 200)

        # 2. Get status
        res_status = self.app.get("/api/location/status")
        self.assertEqual(res_status.status_code, 200)
        status_data = json.loads(res_status.data)
        self.assertEqual(status_data["latitude"], 26.8222)
        self.assertFalse(status_data["geofence"]["breached"])

        # 3. Get Folium Map HTML
        res_map = self.app.get("/api/map")
        self.assertEqual(res_map.status_code, 200)
        self.assertIn("leaflet", res_map.get_data(as_text=True).lower())

    def test_api_sos_flow(self):
        """Verify emergency SOS trigger and resolve cycle."""
        # Trigger
        res_trig = self.app.post("/api/sos/trigger",
                                 data=json.dumps({"message": "Test Emergency"}),
                                 content_type="application/json")
        self.assertEqual(res_trig.status_code, 200)

        # Verify active
        res_status = self.app.get("/api/location/status")
        status_data = json.loads(res_status.data)
        self.assertTrue(status_data["sos_active"])

        # Resolve
        res_res = self.app.post("/api/sos/resolve")
        self.assertEqual(res_res.status_code, 200)

        # Verify inactive
        res_status2 = self.app.get("/api/location/status")
        status_data2 = json.loads(res_status2.data)
        self.assertFalse(status_data2["sos_active"])

    def test_api_currency_detect(self):
        """Verify dedicated currency detection endpoint."""
        payload = {
            "image": self.test_base64,
            "lang": "en"
        }
        res = self.app.post("/api/currency_detect",
                            data=json.dumps(payload),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("spoken_text", data)

    def test_api_medicine_detect(self):
        """Verify dedicated medicine detection endpoint."""
        payload = {
            "image": self.test_base64,
            "lang": "en"
        }
        res = self.app.post("/api/medicine_detect",
                            data=json.dumps(payload),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("spoken_text", data)

    def test_api_detect_medicine_mode(self):
        """Verify /api/detect with medicine mode."""
        payload = {
            "image": self.test_base64,
            "lang": "hi",
            "mode": "medicine",
            "force_speech": True
        }
        res = self.app.post("/api/detect",
                            data=json.dumps(payload),
                            content_type="application/json")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertIn("description_translated", data)

    def test_api_contacts_and_report(self):
        """Verify contacts list and caregiver mobility report."""
        res_contacts = self.app.get("/api/contacts")
        self.assertEqual(res_contacts.status_code, 200)
        data_c = json.loads(res_contacts.data)
        self.assertEqual(data_c["status"], "success")
        self.assertTrue(len(data_c["contacts"]) > 0)

        res_rep = self.app.get("/api/caregiver/export_report?format=json")
        self.assertEqual(res_rep.status_code, 200)
        data_r = json.loads(res_rep.data)
        self.assertIn("safety_score", data_r)

if __name__ == "__main__":
    unittest.main()
