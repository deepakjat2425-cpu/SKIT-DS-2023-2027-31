"""
Unit Tests for Outdoor Assistive Suite:
- GroundHazardDetector (Elevation drops, curbs, potholes, stairs)
- NavigationAssistant (Turn-by-turn wayfinding)
- SceneQAEngine (Conversational Scene Q&A)
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
import numpy as np
import cv2
from modules.ground_hazard_detector import GroundHazardDetector
from modules.navigation_assistant import NavigationAssistant, SKIT_WAYPOINTS
from modules.scene_qa_engine import SceneQAEngine

class TestGroundHazardDetector(unittest.TestCase):
    def setUp(self):
        self.detector = GroundHazardDetector()

    def test_distance_estimation(self):
        # Frame height 480: bottom edge (y=480) is ~0.6m
        dist_bottom = self.detector.estimate_ground_distance(480, 480)
        self.assertAlmostEqual(dist_bottom, 0.6, delta=0.2)

        # Upper ground ROI (y=312) is further (~3.5m)
        dist_top = self.detector.estimate_ground_distance(312, 480)
        self.assertGreater(dist_top, 2.5)

    def test_curb_detection(self):
        # Create a synthetic image with a strong horizontal curb edge in bottom third
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 180
        # Darker asphalt below curb line
        frame[380:480, :] = 40
        # White curb border
        frame[376:380, :] = 255

        annotated, hazards = self.detector.analyze_ground_plane(frame)
        self.assertIsNotNone(annotated)
        self.assertTrue(len(hazards) > 0)
        curb_hazard = next((h for h in hazards if h["category"] == "curb"), None)
        self.assertIsNotNone(curb_hazard)
        self.assertIn("distance_m", curb_hazard)

    def test_pothole_detection(self):
        # Create an image with an enclosed dark pothole region in lower center
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 160
        # Draw dark pothole ellipse in bottom ground plane
        cv2.ellipse(frame, (320, 420), (60, 25), 0, 0, 360, (20, 20, 20), -1)

        annotated, hazards = self.detector.analyze_ground_plane(frame)
        pothole_hazard = next((h for h in hazards if h["category"] == "pothole"), None)
        self.assertIsNotNone(pothole_hazard)
        self.assertEqual(pothole_hazard["category"], "pothole")


class TestNavigationAssistant(unittest.TestCase):
    def setUp(self):
        # Offline: guided straight to the destination, no map services called from tests
        self.nav = NavigationAssistant(online=False)

    def test_destination_resolution(self):
        key, wp = self.nav.resolve_destination("library")
        self.assertEqual(key, "central_library")
        self.assertIsNotNone(wp)

        key2, wp2 = self.nav.resolve_destination("canteen")
        self.assertEqual(key2, "cafeteria")

        key3, wp3 = self.nav.resolve_destination("nonexistent_place_xyz")
        self.assertIsNone(key3)

    def test_haversine_and_bearing(self):
        # Main Gate to Library
        gate = SKIT_WAYPOINTS["main_gate"]
        lib = SKIT_WAYPOINTS["central_library"]
        dist = self.nav.haversine_distance(gate["lat"], gate["lon"], lib["lat"], lib["lon"])
        self.assertGreater(dist, 50.0)
        self.assertLess(dist, 500.0)

        bearing = self.nav.calculate_bearing(gate["lat"], gate["lon"], lib["lat"], lib["lon"])
        self.assertGreaterEqual(bearing, 0.0)
        self.assertLessEqual(bearing, 360.0)

    def test_route_planning_and_progression(self):
        origin_lat = 26.82285
        origin_lon = 75.86475
        plan = self.nav.plan_pedestrian_route(origin_lat, origin_lon, "library")

        self.assertEqual(plan["status"], "success")
        self.assertGreater(len(plan["steps"]), 1)
        self.assertIn("instruction", plan["steps"][0])

        # Test progress update along route
        update = self.nav.update_navigation_progress(26.82350, 75.86540)
        self.assertEqual(update["status"], "navigating")
        self.assertFalse(update["is_completed"])

        # Test arrival at destination coords
        lib = SKIT_WAYPOINTS["central_library"]
        arrival = self.nav.update_navigation_progress(lib["lat"], lib["lon"])
        self.assertEqual(arrival["status"], "arrived")
        self.assertTrue(arrival["is_completed"])


class TestSceneQAEngine(unittest.TestCase):
    def setUp(self):
        self.qa = SceneQAEngine()

    def test_seating_question(self):
        detections = [
            {"class": "chair", "spatial": "ahead", "distance_m": 1.4, "severity": "safe"}
        ]
        ans = self.qa.answer_question("Is there any place to sit?", detections=detections)
        self.assertIn("chair", ans.lower())
        self.assertIn("1.4", ans)

    def test_sign_reading_question(self):
        ans = self.qa.answer_question("What does the sign say?", ocr_text="Emergency Exit Only")
        self.assertIn("emergency exit only", ans.lower())

    def test_path_clear_question(self):
        # Empty detections
        ans_clear = self.qa.answer_question("Is the path clear?", detections=[])
        self.assertIn("clear", ans_clear.lower())

        # Hazard detection
        hazards = [{"class": "car", "spatial": "ahead", "distance_m": 1.1, "severity": "hazard"}]
        ans_hazard = self.qa.answer_question("Can I walk?", detections=hazards)
        self.assertIn("hazard", ans_hazard.lower())

    def test_comprehensive_describe(self):
        detections = [
            {"class": "person", "spatial": "ahead", "distance_m": 2.0},
            {"class": "bench", "spatial": "left", "distance_m": 3.1}
        ]
        ans = self.qa.answer_question("What is in front of me?", detections=detections, ocr_text="Library Block")
        self.assertIn("person", ans.lower())
        self.assertIn("bench", ans.lower())


class TestOutdoorAPIRoutes(unittest.TestCase):
    def setUp(self):
        from app import app
        self.client = app.test_client()

    def test_api_landmarks(self):
        res = self.client.get("/api/navigate/landmarks")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertTrue(len(data["landmarks"]) > 0)

    def test_api_navigation_plan_and_update(self):
        from unittest import mock
        import app as amcs
        patcher = mock.patch.object(amcs.navigation_assistant, "online", False)
        patcher.start()
        self.addCleanup(patcher.stop)
        res = self.client.post("/api/navigate/plan", json={
            "destination": "library",
            "origin_lat": 26.82285,
            "origin_lon": 75.86475
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("Central Library", data["destination"])

        # Update progress
        res_up = self.client.post("/api/navigate/update", json={
            "lat": 26.82350,
            "lon": 75.86540
        })
        self.assertEqual(res_up.status_code, 200)

        # Cancel navigation
        res_cancel = self.client.post("/api/navigate/cancel")
        self.assertEqual(res_cancel.status_code, 200)

    def test_api_scene_qa(self):
        res = self.client.post("/api/scene_qa", json={
            "question": "Is the path clear?"
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("clear", data["answer_en"].lower())

    def test_api_fall_detected(self):
        res = self.client.post("/api/fall_detected", json={
            "latitude": 26.82285,
            "longitude": 75.86475,
            "battery": 88
        })
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "fall_alert_dispatched")
        self.assertIn("whatsapp_url", data)


if __name__ == "__main__":
    unittest.main()
