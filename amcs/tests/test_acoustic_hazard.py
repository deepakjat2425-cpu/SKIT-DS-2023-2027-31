"""
Unit Tests for AcousticHazardDetector (Acoustic Siren & Car Horn AI Detector).
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
from modules.acoustic_hazard_detector import AcousticHazardDetector

class TestAcousticHazardDetector(unittest.TestCase):
    def setUp(self):
        self.detector = AcousticHazardDetector()

    def test_vehicle_horn_detection(self):
        horn_signal = self.detector.simulate_test_signal(hazard_type="horn")
        res = self.detector.analyze_audio_spectrum(horn_signal)
        self.assertTrue(res["detected"])
        self.assertEqual(res["hazard_type"], "vehicle_horn")
        self.assertEqual(res["severity"], "hazard")
        self.assertIn("horn", res["advisory_en"].lower())

    def test_emergency_siren_detection(self):
        siren_signal = self.detector.simulate_test_signal(hazard_type="siren")
        res = self.detector.analyze_audio_spectrum(siren_signal)
        self.assertTrue(res["detected"])
        self.assertEqual(res["hazard_type"], "emergency_siren")
        self.assertEqual(res["severity"], "hazard")
        self.assertIn("siren", res["advisory_en"].lower())

    def test_ambient_silence_or_speech(self):
        # Uniform ambient low noise
        ambient = [0.04] * 128
        res = self.detector.analyze_audio_spectrum(ambient)
        self.assertFalse(res["detected"])
        self.assertEqual(res["hazard_type"], "none")


class TestAcousticAPI(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_audio_hazard_detect(self):
        detector = AcousticHazardDetector()
        test_bins = detector.simulate_test_signal(hazard_type="horn")

        res = self.client.post("/api/audio_hazard_detect", json={
            "frequency_bins": test_bins
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(data["result"]["detected"])
        self.assertEqual(data["result"]["hazard_type"], "vehicle_horn")


if __name__ == "__main__":
    unittest.main()
