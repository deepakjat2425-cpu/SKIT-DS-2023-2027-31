"""
Tests for the real-time environment guidance:
- ground checks: real steps / curbs are found even with camera noise and blur, while grass,
  floor tiles and painted lines (which used to raise false alarms) stay silent
- the extended model's cached results are dropped when the camera turns to a new view, and
  easily confused classes need more confidence
- /api/detect translates and voices a message only when the browser is going to speak it
- a failed online translation is not remembered, and the services are rested for a while
- spoken wording: many of one kind summed up, honest "pass on either side", nearest distance
Automatic Multilanguage Communication System - SKIT Jaipur.
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
from modules.nlp_describer import NLPDescriber
from modules.ocr_translator import OCRTranslator
from modules.vision_detector import VisionDetector

RNG = np.random.default_rng(7)


def _noisy(frame, sigma=8):
    return np.clip(frame.astype(np.float32) + RNG.normal(0, sigma, frame.shape), 0, 255).astype(np.uint8)


def _stairs(light=175, dark=120, blur=0):
    """Steps seen from above: alternating tread / riser shades, closer steps taller (perspective)."""
    frame = np.full((480, 640, 3), 150, np.uint8)
    y, step, k = 480, 34, 0
    while y > 300:
        frame[max(y - step, 0):y] = light if k % 2 == 0 else dark
        y, step, k = y - step, max(10, int(step * 0.85)), k + 1
    return cv2.GaussianBlur(frame, (0, 0), blur) if blur else frame


class TestGroundChecks(unittest.TestCase):
    def setUp(self):
        self.det = GroundHazardDetector()

    def categories(self, frame, boxes=None):
        return {h["category"] for h in self.det.analyze_ground_plane(frame, object_boxes=boxes)[1]}

    def test_steps_found_with_noise_blur_and_low_contrast(self):
        for frame in (_noisy(_stairs()), _noisy(_stairs(), 14), _noisy(_stairs(blur=1.5)),
                      _noisy(_stairs(light=160, dark=135))):
            self.assertIn("stairs", self.categories(frame))

    def test_steps_found_with_a_person_standing_on_them(self):
        self.assertIn("stairs", self.categories(_noisy(_stairs()), boxes=[[250, 200, 390, 480]]))

    def test_steps_wording_is_short_and_honest(self):
        hazard = self.det.analyze_ground_plane(_noisy(_stairs()))[1][0]
        self.assertTrue(hazard["advice"].startswith("Careful, steps ahead"))
        self.assertNotIn("Down", hazard["advice"])       # up or down cannot be told from one picture

    def test_curb_found_with_noise(self):
        frame = np.full((480, 640, 3), 170, np.uint8)
        frame[400:] = 95
        self.assertIn("curb", self.categories(_noisy(frame)))

    def test_textured_ground_raises_no_alarm(self):
        grass = cv2.GaussianBlur(RNG.integers(40, 200, (480, 640, 3)).astype(np.uint8), (0, 0), 1.0)
        self.assertEqual(self.categories(grass), set())

    def test_floor_tiles_are_not_steps(self):
        tiles = np.full((480, 640, 3), 190, np.uint8)
        y, gap = 480, 50
        while y > 280:
            tiles[y - 4:y] = 60
            y, gap = y - gap, max(14, int(gap * 0.85))
        for x in range(0, 640, 80):
            tiles[:, x:x + 4] = 60
        self.assertEqual(self.categories(_noisy(tiles, 6)), set())

    def test_painted_line_is_not_a_curb(self):
        frame = np.full((480, 640, 3), 140, np.uint8)
        frame[396:402] = 230
        self.assertEqual(self.categories(_noisy(frame)), set())


def _stub_with_extended(primary, extended):
    vd = VisionDetector.__new__(VisionDetector)
    vd.conf_thresh, vd.iou_thresh, vd.imgsz = 0.35, 0.45, 640
    vd.model, vd.model_name = object(), "stub.pt"
    vd.ext_model, vd.ext_model_name = object(), "stub-ext.pt"
    vd.ext_conf, vd.ext_every, vd.ext_cache_s = 0.4, 10, 60.0
    vd._frame_counter, vd._ext_cache, vd._ext_previous = 0, (0.0, []), []
    vd.low_conf = None
    vd._memory = []
    vd._predict = lambda model, frame, conf: [o for o in (primary if model is vd.model else extended) if o[1] >= conf]
    return vd


class TestExtendedModelCache(unittest.TestCase):
    def setUp(self):
        # Two different views: a bright wall on the left / a bright floor at the bottom
        self.view_a = np.full((360, 640, 3), 60, np.uint8)
        self.view_a[:, :280] = 210
        self.view_b = np.full((360, 640, 3), 60, np.uint8)
        self.view_b[220:] = 210

    def test_cached_objects_are_dropped_after_the_camera_turns(self):
        vd = _stub_with_extended([], [("door", 0.8, [40, 40, 200, 340])])
        runs = []
        real_run = vd._run_extended
        vd._run_extended = lambda frame, primary: (runs.append(1), real_run(frame, primary))[1]
        shifted_a = np.roll(self.view_a, 6, axis=1)

        names = lambda frame: [d["class"] for d in vd.detect(frame)[1]]
        self.assertEqual(names(self.view_a), ["door"])     # frame 1: the extended model runs
        self.assertEqual(names(shifted_a), ["door"])       # same view, slightly moved: cached result reused
        self.assertEqual(len(runs), 1)
        self.assertEqual(names(self.view_b), [])           # new view: the old door is not announced
        names(self.view_b)                                 # and the model runs again on the next frame
        self.assertEqual(len(runs), 2)

    def test_easily_confused_classes_need_more_confidence(self):
        vd = _stub_with_extended([], [("coin", 0.9, [10, 10, 60, 60]), ("door", 0.5, [100, 20, 260, 340])])
        vd.ext_class_conf = {"coin": 0.95}
        self.assertEqual([n for n, _c, _b, _ok in vd._run_extended(self.view_a, [])], ["door"])


SCENE = {"text": "I can see a chair ahead.", "speech_text": "I can see a chair ahead.", "priority": "NORMAL",
         "is_urgent": False, "changed": False, "scene_key": "chair:ahead", "hazard_key": "", "hazard_distance": None}
HAZARD = {"text": "Careful, a chair right in front of you, about 1 meter away.",
          "speech_text": "Careful, a chair right in front of you, about 1 meter away.", "priority": "HIGH",
          "is_urgent": True, "changed": False, "scene_key": "chair:ahead", "hazard_key": "chair:ahead",
          "hazard_distance": 1.0}


class TestSpeechOnlyWhenSpoken(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.amcs = amcs
        cls.client = amcs.app.test_client()
        ok, buf = cv2.imencode(".jpg", np.full((360, 640, 3), 128, np.uint8))
        cls.image = "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    def post(self, scene, **fields):
        body = dict({"image": self.image, "lang": "hi", "tts": False, "mode": "vision"}, **fields)
        with mock.patch.object(self.amcs.nlp_describer, "describe_scene", return_value=dict(scene)):
            return self.client.post("/api/detect", json=body).get_json()

    def test_unchanged_scene_reuses_the_last_translation(self):
        with mock.patch.object(self.amcs.ocr_translator, "translate_text", side_effect=lambda t, **k: "HI:" + t) as tr:
            first = self.post(SCENE, announced_key="")
            self.assertEqual(first["description_translated"], "HI:I can see a chair ahead.")
            self.assertFalse(first["translation_stale"])
            calls = tr.call_count
            again = self.post(SCENE, announced_key="chair:ahead")      # the user was already told
            self.assertEqual(tr.call_count, calls)
            self.assertTrue(again["translation_stale"])
            self.assertEqual(again["description_translated"], "HI:I can see a chair ahead.")

    def test_hazard_is_translated_when_new_closer_or_due(self):
        told = {"key": "chair:ahead", "distance": 1.0, "due": False}
        with mock.patch.object(self.amcs.ocr_translator, "translate_text", side_effect=lambda t, **k: "HI:" + t) as tr:
            self.post(HAZARD, announced_key="")                        # the warning is given once
            tr.reset_mock()
            self.post(HAZARD, announced_key="chair:ahead", announced_hazard=told)
            self.assertEqual(tr.call_count, 0)                         # same warning, already given
            self.post(HAZARD, announced_key="chair:ahead", announced_hazard=dict(told, distance=2.0))
            self.assertGreater(tr.call_count, 0)                       # 1 m closer than last time
            calls = tr.call_count
            self.post(HAZARD, announced_key="chair:ahead", announced_hazard=dict(told, due=True))
            self.assertGreater(tr.call_count, calls)                   # time to repeat it

    def test_server_voice_only_for_spoken_frames(self):
        with mock.patch.object(self.amcs.speech_synthesizer, "synthesize_later",
                               return_value="/api/speech/x.mp3") as tts:
            data = self.post(SCENE, lang="en", tts=True, announced_key="")
            self.assertEqual(tts.call_count, 1)                        # one message, not two
            self.assertEqual(data["speech_audio_url"], "/api/speech/x.mp3")
            self.post(SCENE, lang="en", tts=True, announced_key="chair:ahead")
            self.assertEqual(tts.call_count, 1)                        # nothing new: no audio made

    def test_frame_does_not_wait_for_server_speech(self):
        synth = self.amcs.speech_synthesizer
        name = synth._get_cache_filename("Background speech test sentence.", "en")
        (synth.cache_dir / name).unlink(missing_ok=True)
        started = __import__("threading").Event()

        def slow_gtts(text, lang="en", force=False):                  # stands in for a 2 s gTTS call
            started.set()
            __import__("time").sleep(0.4)
            (synth.cache_dir / name).write_bytes(b"ID3fake-mp3")
            return {"audio_url": "/static/audio_cache/" + name}

        with mock.patch.object(synth, "synthesize", side_effect=slow_gtts), \
             mock.patch.object(synth, "_gtts_disabled_until", 0.0):
            t0 = __import__("time").perf_counter()
            url = synth.synthesize_later("Background speech test sentence.", lang="en")
            self.assertLess(__import__("time").perf_counter() - t0, 0.2)
            self.assertEqual(url, "/api/speech/" + name)
            self.assertTrue(started.wait(2))
            res = self.client.get(url)                                 # waits for the job, then serves it
            self.assertEqual(res.status_code, 200)
            self.assertEqual(res.data, b"ID3fake-mp3")
            res.close()                                                # Windows: release the file first
        (synth.cache_dir / name).unlink(missing_ok=True)
        self.assertEqual(self.client.get("/api/speech/../../config.py").status_code, 404)
        self.assertEqual(self.client.get("/api/speech/speech_" + "0" * 32 + ".mp3").status_code, 404)


class TestPhoneCertificate(unittest.TestCase):
    def test_certificate_is_made_once_and_reused(self):
        import tempfile
        import app as amcs
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(amcs.config, "BASE_DIR", __import__("pathlib").Path(tmp)):
            cert, key = amcs.dev_certificate()
            self.assertTrue(os.path.exists(cert) and os.path.exists(key))
            first = open(cert, "rb").read()
            self.assertEqual(amcs.dev_certificate(), (cert, key))
            self.assertEqual(open(cert, "rb").read(), first)        # the phone keeps trusting it


class TestTranslationOutage(unittest.TestCase):
    def test_failure_is_not_cached_and_services_rest(self):
        tr = OCRTranslator.__new__(OCRTranslator)          # no OCR model needed
        tr._translation_cache = {}
        down = mock.Mock(side_effect=ConnectionError("offline"))
        with mock.patch.object(OCRTranslator, "_translate_google", down),              mock.patch.object(OCRTranslator, "_translate_mymemory", down):
            self.assertEqual(tr.translate_text("A chair ahead.", target_language="hi", source_language="en"),
                             "A chair ahead .")
            self.assertEqual(down.call_count, 2)
            tr.translate_text("A door ahead.", target_language="hi", source_language="en")
            self.assertEqual(down.call_count, 2)                 # resting: answered offline at once
        tr._online_retry_at = 0.0                                # rest is over, service is back
        with mock.patch.object(OCRTranslator, "_translate_google", return_value="HINDI"):
            self.assertEqual(tr.translate_text("A chair ahead.", target_language="hi", source_language="en"), "HINDI")


class TestGuidanceWording(unittest.TestCase):
    def setUp(self):
        self.nlp = NLPDescriber()

    @staticmethod
    def d(c, sp, dist, sev="safe"):
        return {"class": c, "spatial": sp, "distance_m": dist, "severity": sev}

    def test_many_of_one_kind_are_summed_up(self):
        text = self.nlp.describe_scene([self.d("car", "left", 3.5), self.d("car", "ahead", 4), self.d("car", "ahead", 6),
                                        self.d("car", "right", 5), self.d("truck", "right", 8)])["text"]
        self.assertTrue(text.startswith("I can see 4 cars around you"), text)

    def test_pass_on_either_side_only_when_both_sides_are_free(self):
        text = lambda dets: self.nlp.describe_scene(dets)["text"]
        chair = self.d("chair", "ahead", 1.1, "hazard")
        self.assertIn("either side", text([chair]))
        self.assertIn("Move to your right", text([chair, self.d("person", "left", 1.5)]))
        self.assertIn("way ahead is blocked", text([chair, self.d("person", "left", 1.5), self.d("bag", "right", 1.2)]))
        self.assertIn("either side", text([chair, self.d("person", "left", 4.0)]))   # far away: not in the way

    def test_nearest_distance_is_said_when_close(self):
        self.assertIn("The chair is about 2 meters away.",
                      self.nlp.describe_scene([self.d("chair", "ahead", 1.9), self.d("door", "right", 4)])["text"])
        self.assertNotIn("meters away", self.nlp.describe_scene([self.d("door", "ahead", 5)])["text"])

    def test_mass_nouns(self):
        self.assertIn("some broccoli", self.nlp.describe_scene([self.d("broccoli", "left", 1.0)])["text"])


if __name__ == "__main__":
    unittest.main()
