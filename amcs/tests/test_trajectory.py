"""
Unit tests for Autonomous Vehicle-Grade Time-to-Collision (TTC) & Dynamic Trajectory Tracker.
"""

import unittest
import time
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.trajectory_tracker import TrajectoryTracker
from modules.nlp_describer import NLPDescriber

class TestTrajectoryTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = TrajectoryTracker()
        self.nlp = NLPDescriber()

    def test_approaching_obstacle_ttc(self):
        """Simulate an obstacle moving towards user and verify TTC computation."""
        t0 = 1000.0
        # Frame 1: Bicycle at 5.0m
        det_f1 = [{
            "class": "bicycle",
            "bbox": [200, 100, 300, 300],
            "distance_m": 5.0,
            "spatial": "ahead"
        }]
        d_out1, h1 = self.tracker.update_tracks(det_f1, timestamp=t0)
        self.assertEqual(len(d_out1), 1)

        # Frame 2: Bicycle closes in to 2.2m over 0.8 seconds (v = 2.8m / 0.8s = 3.5 m/s ~= 12.6 km/h)
        t1 = t0 + 0.8
        det_f2 = [{
            "class": "bicycle",
            "bbox": [200, 80, 320, 350],
            "distance_m": 2.2,
            "spatial": "ahead"
        }]
        d_out2, h2 = self.tracker.update_tracks(det_f2, timestamp=t1)
        self.assertEqual(len(d_out2), 1)

        target = d_out2[0]
        # Radial velocity should be positive (approaching)
        self.assertGreater(target["radial_velocity_mps"], 0.5)
        self.assertIsNotNone(target["ttc_seconds"])
        self.assertLess(target["ttc_seconds"], 2.8)
        self.assertEqual(target["collision_risk"], "IMMINENT_HAZARD")

        # Verify NLP responds with urgent dynamic evasive command
        desc = self.nlp.describe_scene(d_out2)
        self.assertEqual(desc["priority"], "CRITICAL")
        self.assertTrue(desc["is_urgent"])
        self.assertIn("Collision in", desc["text"])
        self.assertIn("bicycle", desc["text"].lower())

    def test_stationary_obstacle_no_collision(self):
        """Verify stationary obstacle has SAFE collision risk and no false TTC alarms."""
        t0 = 2000.0
        det1 = [{
            "class": "chair",
            "bbox": [100, 100, 200, 250],
            "distance_m": 2.5,
            "spatial": "left"
        }]
        self.tracker.update_tracks(det1, timestamp=t0)

        # Same distance 1 second later
        t1 = t0 + 1.0
        det2 = [{
            "class": "chair",
            "bbox": [100, 100, 200, 250],
            "distance_m": 2.5,
            "spatial": "left"
        }]
        d_out, h = self.tracker.update_tracks(det2, timestamp=t1)
        self.assertEqual(d_out[0]["collision_risk"], "SAFE")
        self.assertIsNone(d_out[0]["ttc_seconds"])

if __name__ == '__main__':
    unittest.main()
