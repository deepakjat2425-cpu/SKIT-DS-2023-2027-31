"""
Regression tests for recognition accuracy fixes:
- a failing helper stage never breaks /api/detect
- ground hazards ignore areas covered by detected objects
- heuristic findings need several frames before they are reported
- distance jitter does not create false collision alarms
- duplicate fragments of one object are merged; cut-off objects get sensible distances
"""

import base64
import os
import sys
import unittest
from unittest import mock

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.ground_hazard_detector import GroundHazardDetector
from modules.scene_stabilizer import TemporalConfirmer
from modules.trajectory_tracker import TrajectoryTracker
from modules.vision_detector import VisionDetector
from modules.nlp_describer import NLPDescriber


def _jpeg_data_url(img):
    ok, buf = cv2.imencode(".jpg", img)
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


class TestGroundHazardMasking(unittest.TestCase):
    def test_dark_clothing_inside_object_box_is_not_a_pothole(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 160
        cv2.ellipse(frame, (320, 420), (60, 25), 0, 0, 360, (20, 20, 20), -1)
        det = GroundHazardDetector()
        _, unmasked = det.analyze_ground_plane(frame)
        self.assertTrue(any(h["category"] == "pothole" for h in unmasked))
        # The same dark blob, but it lies inside a detected person's box
        _, masked = det.analyze_ground_plane(frame, object_boxes=[[220, 250, 420, 480]])
        self.assertFalse(any(h["category"] == "pothole" for h in masked))

    def test_pothole_needs_lit_floor(self):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 50      # dark floor everywhere
        cv2.ellipse(frame, (320, 420), (60, 25), 0, 0, 360, (5, 5, 5), -1)
        _, hazards = GroundHazardDetector().analyze_ground_plane(frame)
        self.assertFalse(any(h["category"] == "pothole" for h in hazards))

    def test_hough_lines_any_shape(self):
        # Stair-like stripes exercise the HoughLinesP path that used to crash on newer OpenCV
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 170
        for y in range(310, 480, 22):
            frame[y:y + 4, 40:600] = 30
        GroundHazardDetector().analyze_ground_plane(frame)   # must not raise


class TestTemporalConfirmer(unittest.TestCase):
    def test_needs_three_of_four_frames(self):
        c = TemporalConfirmer(window=4, need=3)
        self.assertEqual(c.update({"curb"}, timestamp=1.0), set())
        self.assertEqual(c.update(set(), timestamp=1.2), set())
        self.assertEqual(c.update({"curb"}, timestamp=1.4), set())
        self.assertEqual(c.update({"curb"}, timestamp=1.6), {"curb"})

    def test_long_gap_resets(self):
        c = TemporalConfirmer(window=4, need=2, max_gap_s=2.0)
        c.update({"door"}, timestamp=1.0)
        self.assertEqual(c.update({"door"}, timestamp=10.0), set())


class TestTrackerJitter(unittest.TestCase):
    def test_noisy_stationary_person_has_no_collision_alarm(self):
        tracker = TrajectoryTracker()
        t = 100.0
        for dist in [2.0, 1.7, 2.1, 1.6, 2.0, 1.7, 2.1]:     # +-25 cm noise, 5 frames per second
            out, hazard = tracker.update_tracks(
                [{"class": "person", "bbox": [200, 80, 320, 360], "distance_m": dist, "spatial": "ahead"}], timestamp=t)
            t += 0.2
            self.assertIsNone(hazard)
            self.assertNotEqual(out[0]["collision_risk"], "IMMINENT_HAZARD")
        self.assertGreaterEqual(out[0]["hits"], 7)

    def test_steady_approach_is_detected(self):
        tracker = TrajectoryTracker()
        t = 200.0
        out = None
        for dist in [6.0, 5.4, 4.8, 4.2, 3.6, 3.0]:          # 3 m/s towards the user
            out, hazard = tracker.update_tracks(
                [{"class": "bicycle", "bbox": [250, 100, 350, 300], "distance_m": dist, "spatial": "ahead"}], timestamp=t)
            t += 0.2
        self.assertEqual(out[0]["collision_risk"], "IMMINENT_HAZARD")
        self.assertAlmostEqual(out[0]["radial_velocity_mps"], 3.0, delta=0.4)

    def test_scene_cut_is_not_an_approach(self):
        # A small far person, then a different, much bigger person in a new view: not one object rushing in
        tracker = TrajectoryTracker()
        tracker.update_tracks([{"class": "person", "bbox": [300, 150, 340, 250], "distance_m": 6.0, "spatial": "ahead"}],
                              timestamp=300.0)
        out, hazard = tracker.update_tracks(
            [{"class": "person", "bbox": [260, 40, 420, 360], "distance_m": 1.5, "spatial": "ahead"}], timestamp=300.6)
        self.assertIsNone(hazard)
        self.assertEqual(out[0]["collision_risk"], "SAFE")
        self.assertEqual(out[0]["hits"], 1)

    def test_camera_switch_forgets_old_tracks(self):
        # The user's knee (wide box, front camera), then a standing person in a new rear-camera view
        tracker = TrajectoryTracker()
        tracker.set_view((640, 360, "user"))
        tracker.update_tracks([{"class": "person", "bbox": [465, 209, 639, 355], "distance_m": 5.3, "spatial": "left"}],
                              timestamp=400.0)
        out, hazard = tracker.update_tracks(
            [{"class": "person", "bbox": [395, 234, 479, 521], "distance_m": 2.0, "spatial": "right"}], timestamp=400.6)
        self.assertIsNone(hazard)                          # rejected by shape even without the view change
        self.assertEqual(out[0]["hits"], 1)
        tracker.set_view((480, 640, "environment"))
        self.assertEqual(tracker.tracks, {})


class TestDetectorGeometry(unittest.TestCase):
    def test_fragments_of_one_person_are_merged(self):
        raw = [("person", 0.72, [146, 70, 532, 355]),
               ("person", 0.38, [465, 210, 639, 355]),      # a knee of the same person
               ("cup", 0.40, [330, 64, 346, 85])]
        kept = VisionDetector._merge_fragments(raw)
        self.assertEqual(sorted(k[0] for k in kept), ["cup", "person"])

    def test_two_confident_people_are_kept(self):
        raw = [("person", 0.9, [100, 50, 300, 360]), ("person", 0.85, [250, 60, 450, 360])]
        self.assertEqual(len(VisionDetector._merge_fragments(raw)), 2)

    def test_cut_off_person_close_to_camera(self):
        vd = VisionDetector.__new__(VisionDetector)     # geometry only, no model needed
        # Upper body filling the lower frame, cut off at the bottom edge
        d = vd.estimate_distance("person", 285, 360, bbox=(146, 70, 532, 355), frame_width=640)
        self.assertLess(d, 1.0)
        # Whole standing person, small in frame -> far away
        d_far = vd.estimate_distance("person", 120, 360, bbox=(300, 120, 340, 240), frame_width=640)
        self.assertGreater(d_far, 4.0)

    def test_bus_filling_view_is_not_clamped_close(self):
        vd = VisionDetector.__new__(VisionDetector)
        d = vd.estimate_distance("bus", 300, 360, bbox=(0, 40, 640, 340), frame_width=640)
        self.assertGreater(d, 2.0)


class TestSpokenNames(unittest.TestCase):
    def test_friendly_names_and_static_wording(self):
        nlp = NLPDescriber()
        res = nlp.describe_scene([{"class": "dining table", "spatial": "left", "distance_m": 2.0,
                                   "severity": "caution", "is_priority": False}])
        self.assertIn("table", res["text"])
        self.assertNotIn("dining table", res["text"])
        self.assertNotIn("approaching", res["text"])


class TestDetectApiResilience(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.amcs = amcs
        cls.client = amcs.app.test_client()

    def test_failing_helper_stage_does_not_break_detection(self):
        frame = np.ones((360, 640, 3), dtype=np.uint8) * 128
        with mock.patch.object(self.amcs.ground_hazard_detector, "analyze_ground_plane", side_effect=TypeError("boom")), \
             mock.patch.object(self.amcs.indoor_door_locator, "analyze_frame_for_doors", side_effect=ValueError("boom")):
            res = self.client.post("/api/detect", json={"image": _jpeg_data_url(frame), "lang": "en", "mode": "vision"})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["frame_size"], [640, 360])


if __name__ == "__main__":
    unittest.main()
