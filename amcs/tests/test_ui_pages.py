"""
UI shell, page rendering and API hardening tests.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app, teleassist_engine
import config


class TestPagesRender(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()

    def test_every_page_renders_with_shared_shell(self):
        for path in ["/", "/ocr", "/caregiver", "/sandbox", "/about", "/settings"]:
            with self.subTest(path=path):
                res = self.client.get(path)
                self.assertEqual(res.status_code, 200)
                html = res.get_data(as_text=True)
                # Shared shell: skip link, toast region, dialog and UI helper script
                self.assertIn('class="skip-link"', html)
                self.assertIn('id="toast-region"', html)
                self.assertIn('id="app-dialog"', html)
                self.assertIn("js/ui.js", html)
                self.assertIn(config.PROJECT_TITLE, html)

    def test_current_page_is_marked_in_navigation(self):
        html = self.client.get("/ocr").get_data(as_text=True)
        self.assertIn('href="/ocr" class="nav-link" aria-current="page"', html)

    def test_assist_page_has_live_region_and_controls(self):
        html = self.client.get("/").get_data(as_text=True)
        self.assertIn('id="speech-translated-text"', html)
        self.assertIn('aria-live="polite"', html)
        for element_id in ["start-auto-btn", "mic-btn", "sos-btn", "lang-select", "find-item-btn"]:
            self.assertIn(f'id="{element_id}"', html)
        # The conflicting legacy voice script must not be loaded
        self.assertNotIn("speech_recognition.js", html)

    def test_about_lists_team_from_config(self):
        html = self.client.get("/about").get_data(as_text=True)
        for member in config.TEAM_MEMBERS:
            self.assertIn(member["name"], html)
            self.assertIn(member["roll_no"], html)

    def test_service_worker_served_from_root(self):
        res = self.client.get("/sw.js")
        self.assertEqual(res.status_code, 200)
        self.assertIn("javascript", res.content_type)
        self.assertIn("no-cache", res.headers.get("Cache-Control", ""))
        res.close()

    def test_manifest_icons_exist(self):
        for name in ["icon.svg", "icon-192.png", "icon-512.png"]:
            self.assertTrue((config.STATIC_DIR / "icons" / name).exists(), name)


class TestApiHardening(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        teleassist_engine.end_call()

    def tearDown(self):
        teleassist_engine.end_call()

    def test_frames_ignored_without_active_call(self):
        res = self.client.post("/api/teleassist/frame", json={"image": "data:image/jpeg;base64,abc"})
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.get_json()["status"], "frame_ignored")
        self.assertFalse(self.client.get("/api/teleassist/frame").get_json()["has_frame"])

    def test_frames_must_be_data_urls(self):
        self.client.post("/api/teleassist/call", json={"action": "start"})
        res = self.client.post("/api/teleassist/frame", json={"image": "javascript:alert(1)"})
        self.assertEqual(res.status_code, 400)

    def test_intercom_reports_call_state_and_rejects_empty(self):
        self.assertEqual(self.client.post("/api/teleassist/intercom", json={"message": "   "}).status_code, 400)
        poll = self.client.get("/api/teleassist/intercom").get_json()
        self.assertIn("call_active", poll)
        self.assertFalse(poll["call_active"])

    def test_sos_uses_coordinates_sent_with_alert(self):
        res = self.client.post("/api/sos/trigger", json={"message": "help", "latitude": 26.9, "longitude": 75.8})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertAlmostEqual(data["latitude"], 26.9)
        self.assertAlmostEqual(data["longitude"], 75.8)
        self.client.post("/api/sos/resolve")

    def test_unknown_api_route_returns_json(self):
        res = self.client.get("/api/does-not-exist")
        self.assertEqual(res.status_code, 404)
        self.assertEqual(res.get_json()["status"], "error")


if __name__ == "__main__":
    unittest.main()
