import unittest
from unittest.mock import patch

from app import create_app

class LanguageApiTests(unittest.TestCase):
    def setUp(self):
        self.app = create_app().test_client()

    @patch("app.routes.describe.stream")
    @patch("app.routes.describe.speak", return_value="test.mp3")
    def test_supported_language_returns_audio_url(self, speak_mock, stream_mock):
        stream_mock.detections = []
        response = self.app.post("/api/describe", json={"lang": "hi"})
        self.assertEqual(response.status_code, 200)
        payload = response.get_json()
        self.assertEqual(payload["lang"], "hi")
        self.assertEqual(payload["language_name"], "Hindi")
        self.assertIn("audio_url", payload)
        speak_mock.assert_called_once()

    def test_unsupported_language_returns_400(self):
        response = self.app.post("/api/describe", json={"lang": "xx"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["error"], "Unsupported language")

    def test_missing_language_defaults_to_english(self):
        with patch("app.routes.describe.stream") as stream_mock, patch(
            "app.routes.describe.speak", return_value="test.mp3"
        ):
            stream_mock.detections = []
            response = self.app.post("/api/describe", json={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_json()["lang"], "en")

if __name__ == "__main__":
    unittest.main()
