"""
Unit tests for Medicine Blister Strip & Expiry Date Reader.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import unittest
import numpy as np
from datetime import datetime
from modules.medicine_reader import MedicineReader

class TestMedicineReader(unittest.TestCase):
    def setUp(self):
        self.reader = MedicineReader()

    def test_preprocess_foil(self):
        """Tests CLAHE & bilateral filter foil glare reduction."""
        dummy_foil = np.random.randint(80, 240, (200, 200, 3), dtype=np.uint8)
        processed = self.reader.preprocess_foil(dummy_foil)
        self.assertIsNotNone(processed)
        self.assertEqual(processed.shape, (200, 200, 3))

    def test_parse_dolo_valid_expiry(self):
        """Tests recognizing Dolo 650 with future expiry date."""
        sample_ocr = "MICRO LABS LIMITED DOLO 650 B.NO: ML24A01 MFG: 09/2024 EXP: 08/2028"
        res = self.reader.analyze_strip(raw_text=sample_ocr)

        self.assertTrue(res["detected"])
        self.assertEqual(res["medicine_name"], "Dolo")
        self.assertEqual(res["dosage"], "650mg")
        self.assertEqual(res["expiry_str"], "08/2028")
        self.assertFalse(res["is_expired"])
        self.assertIn("Dolo 650mg", res["advisory_en"])
        self.assertIn("Safe to use", res["advisory_en"])
        self.assertIn("सुरक्षित", res["advisory_hi"])

    def test_parse_expired_medicine(self):
        """Tests expired medicine detection and safety warning."""
        sample_ocr = "AZITHROMYCIN 500 MG TABLETS IP BATCH NO: 7721 EXP: 03/2023"
        res = self.reader.analyze_strip(raw_text=sample_ocr)

        self.assertTrue(res["detected"])
        self.assertEqual(res["medicine_name"], "Azithromycin")
        self.assertEqual(res["dosage"], "500mg")
        self.assertEqual(res["expiry_str"], "03/2023")
        self.assertTrue(res["is_expired"])
        self.assertIn("DANGER", res["advisory_en"])
        self.assertIn("EXPIRED", res["advisory_en"])
        self.assertIn("सावधान", res["advisory_hi"])

    def test_text_month_expiry_format(self):
        """Tests parsing month names like 'EXP: NOV 2027'."""
        sample_ocr = "PAN 40 Pantoprazole Gastro-resistant Tablets EXP: NOV 2027"
        res = self.reader.analyze_strip(raw_text=sample_ocr)

        self.assertTrue(res["detected"])
        self.assertIn(res["medicine_name"].lower(), ["pan", "pantoprazole"])
        self.assertEqual(res["expiry_str"], "11/2027")
        self.assertEqual(res["expiry_month"], 11)
        self.assertEqual(res["expiry_year"], 2027)

    def test_no_medicine_text(self):
        """Tests handling empty or irrelevant text."""
        res = self.reader.analyze_strip(raw_text="Random street signboard text")
        self.assertFalse(res["detected"])
        self.assertIn("No medicine label", res["advisory_en"])

if __name__ == "__main__":
    unittest.main()
