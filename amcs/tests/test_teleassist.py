"""
Unit Tests for TeleAssistEngine (Caregiver "See What I See" Live Tele-Assistance).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
from modules.teleassist_engine import TeleAssistEngine

class TestTeleAssistEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TeleAssistEngine()

    def test_call_lifecycle(self):
        start = self.engine.start_call("session_123")
        self.assertEqual(start["status"], "call_started")
        self.assertTrue(self.engine.call_active)

        # Update frame
        self.engine.update_frame("data:image/jpeg;base64,mockframe")
        view = self.engine.get_latest_frame()
        self.assertTrue(view["call_active"])
        self.assertTrue(view["has_frame"])
        self.assertEqual(view["frame"], "data:image/jpeg;base64,mockframe")

        # End call
        end = self.engine.end_call()
        self.assertEqual(end["status"], "call_ended")
        self.assertFalse(self.engine.call_active)

    def test_caregiver_intercom_guidance(self):
        self.engine.start_call()
        self.engine.post_caregiver_guidance("Take two steps to your right, there is a curb.")

        msgs = self.engine.fetch_pending_guidance()
        self.assertEqual(len(msgs), 1)
        self.assertIn("curb", msgs[0]["message"])

        # Subsequent fetch is empty
        self.assertEqual(len(self.engine.fetch_pending_guidance()), 0)

    def test_webrtc_signals(self):
        self.engine.set_signal("offer", {"sdp": "mock_offer_sdp"})
        self.engine.set_signal("answer", {"sdp": "mock_answer_sdp"})
        signals = self.engine.get_signals()
        self.assertEqual(signals["offer"]["sdp"], "mock_offer_sdp")
        self.assertEqual(signals["answer"]["sdp"], "mock_answer_sdp")


class TestTeleAssistAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_teleassist_call_and_frame(self):
        # Start call
        res_call = self.client.post("/api/teleassist/call", json={"action": "start"})
        self.assertEqual(res_call.status_code, 200)

        # Upload frame
        res_frame = self.client.post("/api/teleassist/frame", json={"image": "data:image/jpeg;base64,sample"})
        self.assertEqual(res_frame.status_code, 200)

        # Get frame
        res_get = self.client.get("/api/teleassist/frame")
        self.assertEqual(res_get.status_code, 200)
        self.assertTrue(res_get.get_json()["has_frame"])

        # Post caregiver guidance intercom
        res_msg = self.client.post("/api/teleassist/intercom", json={"message": "All clear ahead."})
        self.assertEqual(res_msg.status_code, 200)

        # Poll intercom
        res_poll = self.client.get("/api/teleassist/intercom")
        self.assertEqual(res_poll.status_code, 200)
        self.assertEqual(len(res_poll.get_json()["messages"]), 1)


if __name__ == "__main__":
    unittest.main()
