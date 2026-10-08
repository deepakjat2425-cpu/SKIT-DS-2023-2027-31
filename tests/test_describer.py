import unittest

from app.describer import describe


def det(label, direction="ahead", distance="far"):
    return {"label": label, "direction": direction, "distance": distance, "confidence": 0.9}


class DescribeTests(unittest.TestCase):
    def test_empty(self):
        self.assertIn("can't see", describe([])["text"])

    def test_single_object(self):
        self.assertEqual(describe([det("chair", "left")])["text"], "I can see a chair on your left.")

    def test_article_and_plural(self):
        text = describe([det("person"), det("person"), det("apple", "right")])["text"]
        self.assertIn("two people ahead", text)
        self.assertIn("an apple on your right", text)

    def test_alert_for_very_close(self):
        result = describe([det("car", "right", "very close"), det("chair")])
        self.assertEqual(len(result["alerts"]), 1)
        self.assertTrue(result["text"].startswith("Caution, a car is very close, on your right."))


if __name__ == "__main__":
    unittest.main()
