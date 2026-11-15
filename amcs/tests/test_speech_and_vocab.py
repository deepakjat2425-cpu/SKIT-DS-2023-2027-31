"""
Tests for the calmer speech and the extended vocabulary:
- the person in front of a laptop/front camera is "you", not a hazard; left/right are mirrored
- small items (cups, phones, keys...) are described but never raise "careful" warnings
- worn items inside a person's box are not reported as separate objects
- routine updates only mention what is new
- item finder resolves whole words ("glasses" is not "glass")
- /api/detect skips server speech when the browser speaks itself
"""

import base64
import os
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.item_finder_engine import ItemFinderEngine
from modules.nlp_describer import NLPDescriber, say_distance
from modules.vision_detector import VisionDetector


def _stub_detector(primary, extended=None):
    """A VisionDetector whose models are replaced by fixed outputs (no weights needed)."""
    vd = VisionDetector.__new__(VisionDetector)
    vd.conf_thresh, vd.iou_thresh, vd.imgsz = 0.35, 0.45, 640
    vd.model, vd.model_name = object(), "stub.pt"
    vd.ext_model = object() if extended is not None else None
    vd.ext_model_name = "stub-ext.pt" if extended is not None else None
    vd.ext_conf, vd.ext_every, vd.ext_cache_s = 0.4, 1, 2.5
    vd._frame_counter, vd._ext_cache, vd._ext_previous = 0, (0.0, []), []
    vd._predict = lambda model, frame, conf: list(primary if model is vd.model else extended)
    return vd


class TestDetectorSemantics(unittest.TestCase):
    frame = np.zeros((360, 640, 3), dtype=np.uint8)

    def test_front_camera_marks_user_and_mirrors_sides(self):
        vd = _stub_detector([("person", 0.92, [150, 60, 500, 360]), ("cup", 0.7, [20, 200, 80, 260])])
        _, dets, _ = vd.detect(self.frame, camera_facing="user")
        user = next(d for d in dets if d["class"] == "person")
        cup = next(d for d in dets if d["class"] == "cup")
        self.assertTrue(user.get("is_user"))
        self.assertEqual(user["severity"], "safe")
        self.assertEqual(cup["spatial"], "right")      # image-left is the user's right on a front camera
        _, dets_rear, _ = vd.detect(self.frame, camera_facing="environment")
        self.assertEqual(next(d for d in dets_rear if d["class"] == "cup")["spatial"], "left")
        self.assertFalse(any(d.get("is_user") for d in dets_rear))

    def test_small_items_never_hazards(self):
        vd = _stub_detector([("cup", 0.8, [250, 100, 390, 300]), ("chair", 0.8, [240, 20, 420, 355])])
        _, dets, _ = vd.detect(self.frame)
        by = {d["class"]: d for d in dets}
        self.assertEqual(by["cup"]["severity"], "safe")
        self.assertEqual(by["chair"]["severity"], "hazard")

    def test_worn_items_and_duplicates_are_dropped(self):
        person = ("person", 0.9, [150, 40, 500, 360])
        ext = [("watch", 0.5, [200, 200, 230, 230]),        # on the wrist -> clothing
               ("keys", 0.5, [560, 300, 600, 330]),         # on the table -> keep
               ("door", 0.9, [520, 0, 640, 300])]
        vd = _stub_detector([person], ext)
        _, dets, _ = vd.detect(self.frame, extended=True)
        names = sorted(d["class"] for d in dets)
        self.assertEqual(names, ["door", "keys", "person"])
        door = next(d for d in dets if d["class"] == "door")
        self.assertEqual(door["source"], "extended")
        self.assertTrue(door["confirmed"])                  # confident on its own
        keys = next(d for d in dets if d["class"] == "keys")
        self.assertFalse(keys["confirmed"])                 # first sighting, low confidence
        _, dets2, _ = vd.detect(self.frame, extended=True)
        self.assertTrue(next(d for d in dets2 if d["class"] == "keys")["confirmed"])

    def test_tie_on_a_person_is_not_announced(self):
        vd = _stub_detector([("person", 0.9, [150, 40, 500, 360]), ("tie", 0.6, [300, 150, 330, 260]),
                             ("handbag", 0.6, [560, 280, 630, 350])])
        _, dets, _ = vd.detect(self.frame)
        self.assertEqual(sorted(d["class"] for d in dets), ["handbag", "person"])

    def test_body_part_fragments_fold_into_the_user(self):
        vd = _stub_detector([("person", 0.9, [150, 40, 500, 360]),
                             ("person", 0.45, [420, 250, 560, 360]),     # the user's knee
                             ("person", 0.46, [465, 209, 639, 359]),     # knee cut off by the bottom edge
                             ("person", 0.8, [560, 60, 640, 360])])      # someone else at the edge
        _, dets, _ = vd.detect(self.frame, camera_facing="user")
        people = [d for d in dets if d["class"] == "person"]
        self.assertEqual(len(people), 2)
        self.assertEqual(sum(1 for d in people if d.get("is_user")), 1)


class TestSpeechWording(unittest.TestCase):
    def setUp(self):
        self.nlp = NLPDescriber()

    @staticmethod
    def d(c, sp, dist, sev="safe", **k):
        return dict({"class": c, "spatial": sp, "distance_m": dist, "severity": sev}, **k)

    def test_user_is_not_a_hazard(self):
        res = self.nlp.describe_scene([self.d("person", "ahead", 0.6, is_user=True), self.d("door", "right", 3.0)])
        self.assertFalse(res["is_urgent"])
        self.assertTrue(res["text"].startswith("I can see you"))
        self.assertIn("door on your right", res["text"])

    def test_only_new_things_are_announced(self):
        first = self.nlp.describe_scene([self.d("door", "right", 3.0)], announced_key="")
        self.assertIn("door", first["speech_text"])
        more = self.nlp.describe_scene([self.d("door", "right", 3.0), self.d("cup", "ahead", 0.5)],
                                       announced_key=first["scene_key"])
        self.assertEqual(more["speech_text"], "Now also a cup ahead.")
        fewer = self.nlp.describe_scene([self.d("door", "right", 3.0)], announced_key=more["scene_key"])
        self.assertEqual(fewer["speech_text"], "")
        gone = self.nlp.describe_scene([], announced_key=more["scene_key"])
        self.assertIn("clear", gone["speech_text"])

    def test_hazard_key_and_short_wording(self):
        res = self.nlp.describe_scene([self.d("chair", "ahead", 1.1, "hazard")])
        self.assertEqual(res["hazard_key"], "chair:ahead")
        self.assertEqual(res["text"], "Careful, a chair right in front of you, about 1 meter away. You can pass on either side.")

    def test_friendly_distances(self):
        self.assertEqual(say_distance(0.6), "half a meter away")
        self.assertEqual(say_distance(2.3), "about 2.5 meters away")
        self.assertEqual(say_distance(7.4), "about 7 meters away")


class TestItemAliases(unittest.TestCase):
    def test_whole_word_aliases(self):
        f = ItemFinderEngine()
        self.assertEqual(f.resolve_target_class("find my glasses")[0], "spectacles")
        self.assertEqual(f.resolve_target_class("where is my glass")[0], "cup")
        self.assertEqual(f.resolve_target_class("my wallet")[0], "wallet")


class TestDetectSpeechFields(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.client = amcs.app.test_client()

    def test_browser_voice_skips_server_tts(self):
        ok, buf = cv2.imencode(".jpg", np.full((360, 640, 3), 128, np.uint8))
        img = "data:image/jpeg;base64," + base64.b64encode(buf).decode()
        res = self.client.post("/api/detect", json={"image": img, "lang": "en", "force_speech": True,
                                                    "tts": False, "camera_facing": "user", "announced_key": ""})
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIsNone(data["audio_url"])
        for key in ("scene_key", "hazard_key", "speech_en", "speech_translated"):
            self.assertIn(key, data)


if __name__ == "__main__":
    unittest.main()
