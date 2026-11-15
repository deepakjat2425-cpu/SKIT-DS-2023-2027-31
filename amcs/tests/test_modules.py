"""
Unit tests for Automatic Multilanguage Communication System modules.
"""

import unittest
import numpy as np
import cv2
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vision_detector import VisionDetector
from modules.nlp_describer import NLPDescriber
from modules.ocr_translator import OCRTranslator
from modules.speech_synthesizer import SpeechSynthesizer
from modules.location_tracker import LocationTracker
import config

class TestSystemModules(unittest.TestCase):
    def setUp(self):
        self.vision = VisionDetector()
        self.nlp = NLPDescriber()
        self.ocr = OCRTranslator()
        self.speech = SpeechSynthesizer()
        self.tracker = LocationTracker()

    def test_vision_detector(self):
        # Create a blank test image
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated, detections, stats = self.vision.detect(dummy_img)
        self.assertIsNotNone(annotated)
        self.assertIsInstance(detections, list)
        self.assertIn("total_detections", stats)

    def test_nlp_describer(self):
        # Test empty scene
        res_empty = self.nlp.describe_scene([])
        self.assertIn("clear", res_empty["text"].lower())

        # Test simulated hazard
        fake_hazard = [{
            "class": "chair",
            "spatial": "ahead",
            "distance_m": 1.1,
            "severity": "hazard",
            "is_priority": True
        }]
        res_hazard = self.nlp.describe_scene(fake_hazard)
        self.assertEqual(res_hazard["priority"], "HIGH")
        self.assertTrue(res_hazard["is_urgent"])
        self.assertIn("chair", res_hazard["text"].lower())

    def test_ocr_translator(self):
        # Test translation fallback
        res_hi = self.ocr.translate_text("Stop", target_language="hi")
        self.assertTrue(len(res_hi) > 0)
        
        # Test railway station translation from PPT
        res_station = self.ocr.translate_text("Railway Station", target_language="hi")
        self.assertIn("रेलवे", res_station)

    def test_speech_synthesizer(self):
        speech_res = self.speech.synthesize("Path is clear", lang="en")
        self.assertIsNotNone(speech_res["text"])
        # Either URL generated, cached, or browser fallback flag
        self.assertTrue(speech_res["audio_url"] is not None or speech_res["fallback_browser"] is True)

    def test_location_tracker(self):
        # Test GPS update
        self.tracker.update_location(26.8225, 75.8655, accuracy=8.0)
        self.assertEqual(self.tracker.current_lat, 26.8225)
        # Test Folium Map generation
        map_html = self.tracker.generate_folium_map()
        self.assertIn("leaflet", map_html.lower())

if __name__ == '__main__':
    unittest.main()
