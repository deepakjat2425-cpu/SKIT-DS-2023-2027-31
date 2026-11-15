"""
Unit tests for Face & Social Interaction Engine.
"""

import unittest
import numpy as np
import cv2
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app
from modules.face_social_engine import FaceSocialEngine

class TestFaceSocialEngine(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.engine = FaceSocialEngine()

    def test_embedding_extraction(self):
        """Verify 128-d unit-normalized embedding extraction."""
        face = np.full((80, 80, 3), (150, 160, 180), dtype=np.uint8)
        emb = self.engine.extract_embedding(face)
        self.assertIsNotNone(emb)
        self.assertEqual(len(emb), 128)
        # Verify L2 norm is ~1.0
        self.assertAlmostEqual(float(np.linalg.norm(emb)), 1.0, places=3)

    def test_expression_estimation(self):
        """Verify expression estimation returns valid string."""
        face = np.zeros((100, 100, 3), dtype=np.uint8)
        expr = self.engine.estimate_expression(face)
        self.assertIn(expr, ["smiling", "neutral", "attentive"])

    def test_register_and_identify(self):
        """Test enrolling a new contact and identifying them."""
        # Create a distinctive colored face pattern
        test_face = np.zeros((100, 100, 3), dtype=np.uint8)
        test_face[:, :] = (20, 180, 240) # Bright gold/orange pattern

        # Register
        ok, msg = self.engine.register_contact("Dr. Sharma", "Physician", test_face)
        self.assertTrue(ok)

        # Identify from full image frame
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Place person in frame
        frame[100:350, 200:350] = (20, 180, 240)
        res = self.engine.identify_person(frame, [200, 100, 350, 350])
        self.assertIsNotNone(res)
        self.assertEqual(res["name"], "Dr. Sharma")
        self.assertEqual(res["role"], "Physician")
        self.assertTrue(res["recognized"])

    def test_api_contacts_endpoint(self):
        """Verify /api/contacts endpoint."""
        res = self.app.get("/api/contacts")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["status"], "success")
        self.assertGreater(len(data["contacts"]), 0)
        # Check Vikas Sir is pre-enrolled
        names = [c["name"] for c in data["contacts"]]
        self.assertIn("Vikas Sir", names)

if __name__ == '__main__':
    unittest.main()
