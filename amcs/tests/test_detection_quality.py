"""
Tests for the detection-accuracy stages and the features built on them:
- two-threshold tracking keeps a faint detection only when it continues a clearly seen object
- one object reported under two class names is reduced to the more confident one
- similar-sized neighbours are not merged as "fragments"
- bearing, low-light flag, severity refresh after distance smoothing
- automatic model choice by measured speed, and the lighter model used for "fast" (phone) requests
- Scene Q&A counting / locating, /api/health, quality profiles, new Assist panels
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

import config
from modules import vision_detector as vd_module
from modules.scene_qa_engine import SceneQAEngine
from modules.trajectory_tracker import TrajectoryTracker
from modules.vision_detector import QUALITY_IMAGE_SIZES, VisionDetector

FRAME = np.full((360, 640, 3), 128, dtype=np.uint8)


def _stub(outputs, low_conf=0.20):
    """A VisionDetector without weights; `outputs` is a mutable list returned by the 'model'."""
    vd = VisionDetector.__new__(VisionDetector)
    vd.conf_thresh, vd.iou_thresh, vd.imgsz = 0.35, 0.45, 640
    vd.model, vd.model_name = object(), "stub.pt"
    vd.ext_model, vd.ext_model_name = None, None
    vd.ext_conf, vd.ext_every, vd.ext_cache_s = 0.4, 1, 2.5
    vd._frame_counter, vd._ext_cache, vd._ext_previous = 0, (0.0, []), []
    vd.low_conf = low_conf
    vd._memory = []
    vd.seen_conf = []
    def _predict(model, frame, conf):
        vd.seen_conf.append(conf)
        return [o for o in outputs if o[1] >= conf]
    vd._predict = _predict
    return vd


class TestTwoThresholdTracking(unittest.TestCase):
    def test_faint_detection_continues_a_clear_one(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs)
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual([d["class"] for d in dets], ["chair"])
        self.assertEqual(vd.seen_conf[-1], 0.20)            # the model is asked at the weak threshold
        outputs[:] = [("chair", 0.24, [204, 122, 334, 342])]   # same chair, half hidden now
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual([d["class"] for d in dets], ["chair"])

    def test_faint_detection_without_history_is_dropped(self):
        vd = _stub([("dog", 0.27, [50, 200, 150, 300])])
        for _ in range(3):
            _, dets, _ = vd.detect(FRAME)
            self.assertEqual(dets, [])

    def test_faint_detection_elsewhere_is_not_the_same_object(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs)
        vd.detect(FRAME)
        outputs[:] = [("chair", 0.24, [480, 120, 610, 340])]
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual(dets, [])

    def test_memory_expires(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs)
        vd.detect(FRAME)
        for m in vd._memory:
            m[2] -= vd.hysteresis_seconds + 1                # seen clearly too long ago
        outputs[:] = [("chair", 0.24, [200, 120, 330, 340])]
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual(dets, [])

    def test_faint_sightings_do_not_keep_themselves_alive(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs)
        vd.detect(FRAME)
        first_seen = vd._memory[0][2]
        outputs[:] = [("chair", 0.24, [200, 120, 330, 340])]
        vd.detect(FRAME)
        self.assertEqual(vd._memory[0][2], first_seen)       # the clock runs from the clear sighting

    def test_new_view_forgets_objects(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs)
        vd.detect(FRAME, camera_facing="environment")
        outputs[:] = [("chair", 0.24, [200, 120, 330, 340])]
        _, dets, _ = vd.detect(FRAME, camera_facing="user")
        self.assertEqual(dets, [])

    def test_disabled_without_weak_threshold(self):
        outputs = [("chair", 0.72, [200, 120, 330, 340])]
        vd = _stub(outputs, low_conf=None)
        vd.detect(FRAME)
        self.assertEqual(vd.seen_conf[-1], 0.35)


class TestDuplicateHandling(unittest.TestCase):
    def test_same_box_two_classes_keeps_the_confident_one(self):
        vd = _stub([("couch", 0.70, [100, 150, 500, 340]), ("bed", 0.50, [104, 152, 498, 338])])
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual([d["class"] for d in dets], ["couch"])

    def test_different_objects_that_overlap_are_both_kept(self):
        vd = _stub([("person", 0.9, [200, 40, 360, 350]), ("chair", 0.8, [240, 200, 420, 350])])
        _, dets, _ = vd.detect(FRAME)
        self.assertEqual(sorted(d["class"] for d in dets), ["chair", "person"])

    def test_two_people_side_by_side_are_not_fragments(self):
        raw = [("person", 0.80, [100, 60, 260, 350]), ("person", 0.50, [210, 70, 370, 350])]   # ~31% overlap
        self.assertEqual(len(VisionDetector._merge_fragments(raw)), 2)

    def test_box_inside_a_stronger_one_is_a_fragment(self):
        raw = [("person", 0.80, [100, 60, 300, 350]), ("person", 0.45, [150, 200, 290, 350])]
        self.assertEqual(len(VisionDetector._merge_fragments(raw)), 1)


class TestGeometryAndLight(unittest.TestCase):
    def test_bearing_sign_and_mirroring(self):
        vd = _stub([("cup", 0.8, [560, 150, 620, 210]), ("book", 0.8, [300, 150, 340, 210])])
        _, dets, _ = vd.detect(FRAME, camera_facing="environment")
        by = {d["class"]: d for d in dets}
        self.assertGreater(by["cup"]["bearing_deg"], 20)
        self.assertLess(by["cup"]["bearing_deg"], 36)        # inside half of the 70 degree field of view
        self.assertAlmostEqual(by["book"]["bearing_deg"], 0, delta=1.0)
        _, dets, _ = vd.detect(FRAME, camera_facing="user")
        self.assertLess(next(d for d in dets if d["class"] == "cup")["bearing_deg"], -20)

    def test_dim_frame_is_flagged_but_left_alone_by_default(self):
        vd = _stub([])
        dark = np.full((360, 640, 3), 18, dtype=np.uint8)
        out, flagged = vd._prepare_frame(dark)
        self.assertTrue(flagged)
        self.assertIs(out, dark)                             # software brightening is opt-in
        same, flagged = vd._prepare_frame(FRAME)
        self.assertFalse(flagged)
        self.assertIs(same, FRAME)
        _, _, stats = vd.detect(dark)
        self.assertTrue(stats["low_light"])
        self.assertFalse(config.LOW_LIGHT_ENHANCE)

    def test_optional_brightening_still_works(self):
        vd = _stub([])
        vd.enhance_low_light = True
        dark = np.full((360, 640, 3), 18, dtype=np.uint8)
        out, flagged = vd._prepare_frame(dark)
        self.assertTrue(flagged)
        self.assertGreater(float(out.mean()), float(dark.mean()) * 2)
        self.assertEqual(out.shape, dark.shape)

    def test_quality_profile_sets_resolution(self):
        vd = _stub([])
        _, _, stats = vd.detect(FRAME, quality="high")
        self.assertEqual(stats["image_size"], QUALITY_IMAGE_SIZES["high"])
        _, _, stats = vd.detect(FRAME, quality="nonsense")
        self.assertEqual(stats["image_size"], 640)

    def test_severity_follows_smoothed_distance(self):
        vd = _stub([])
        dets = [{"class": "chair", "distance_m": 1.1, "spatial": "ahead", "severity": "safe", "is_obstacle": True},
                {"class": "cup", "distance_m": 0.5, "spatial": "ahead", "severity": "hazard", "is_obstacle": False}]
        counts = vd.refresh_severity(dets)
        by = {d["class"]: d for d in dets}
        self.assertEqual(by["chair"]["severity"], "hazard")
        self.assertEqual(by["cup"]["severity"], "safe")
        self.assertEqual(counts["hazard_count"], 1)
        self.assertEqual(counts["nearest"]["class"], "cup")


class TestDistanceSmoothing(unittest.TestCase):
    def test_stationary_object_reports_median_distance(self):
        tracker = TrajectoryTracker()
        t, out = 50.0, None
        for dist in [1.4, 1.2, 1.5, 1.2, 1.4]:              # jitter around the 1.3 m hazard line
            out, _ = tracker.update_tracks(
                [{"class": "chair", "bbox": [200, 80, 320, 360], "distance_m": dist, "spatial": "ahead"}], timestamp=t)
            t += 0.4
        self.assertEqual(out[0]["distance_m"], 1.4)          # median of the last three
        self.assertEqual(out[0]["distance_raw"], 1.4)

    def test_approaching_object_keeps_live_distance(self):
        tracker = TrajectoryTracker()
        t, out = 80.0, None
        for dist in [6.0, 5.2, 4.4, 3.6, 2.8]:
            out, _ = tracker.update_tracks(
                [{"class": "bicycle", "bbox": [250, 100, 350, 300], "distance_m": dist, "spatial": "ahead"}], timestamp=t)
            t += 0.3
        self.assertEqual(out[0]["distance_m"], 2.8)


class _Weights(str):
    """Stands in for a loaded YOLO model: its name, plus a predict() that does nothing."""
    def predict(self, *args, **kwargs):
        return []


class TestAutoModelChoice(unittest.TestCase):
    def _load(self, speeds, missing=()):
        def fake_yolo(path):
            name = os.path.basename(path)
            if name in missing:
                raise FileNotFoundError(name)
            return _Weights(name)
        vd = VisionDetector.__new__(VisionDetector)
        vd.imgsz, vd.model_path = 640, "auto"
        with mock.patch.object(vd_module, "YOLO", side_effect=fake_yolo), \
             mock.patch.object(VisionDetector, "_benchmark", lambda self, model: speeds[model]), \
             mock.patch.object(config, "YOLO_AUTO_MODELS", ["yolo26m.pt", "yolo26s.pt"]), \
             mock.patch.object(config, "YOLO_FALLBACK_MODELS", ["yolo26s.pt", "yolov8n.pt"]), \
             mock.patch.object(config, "YOLO_AUTO_BUDGET_MS", 330.0):
            vd._load_model()
        return vd

    def test_fast_machine_gets_the_accurate_model(self):
        self.assertEqual(self._load({"yolo26m.pt": 180, "yolo26s.pt": 80, "yolov8n.pt": 30}).model_name, "yolo26m.pt")

    def test_slow_machine_gets_a_lighter_model(self):
        vd = self._load({"yolo26m.pt": 700, "yolo26s.pt": 300, "yolov8n.pt": 90})
        self.assertEqual(vd.model_name, "yolo26s.pt")
        self.assertEqual(vd.infer_ms, 300)

    def test_very_slow_machine_gets_the_fastest_available(self):
        self.assertEqual(self._load({"yolo26m.pt": 2000, "yolo26s.pt": 900, "yolov8n.pt": 400}).model_name, "yolov8n.pt")

    def test_missing_weights_are_skipped(self):
        vd = self._load({"yolo26s.pt": 80, "yolov8n.pt": 30}, missing=("yolo26m.pt",))
        self.assertEqual(vd.model_name, "yolo26s.pt")

    def test_lighter_model_is_loaded_for_fast_quality(self):
        vd = self._load({"yolo26m.pt": 180, "yolo26s.pt": 80, "yolov8n.pt": 30})
        self.assertEqual(vd.fast_model_name, "yolo26s.pt")
        # Already on the lightest auto model: nothing extra to load
        self.assertIsNone(self._load({"yolo26m.pt": 700, "yolo26s.pt": 300, "yolov8n.pt": 90}).fast_model_name)


class TestFastQualityModel(unittest.TestCase):
    def _detector(self):
        outputs = [("chair", 0.8, [100, 120, 220, 300])]
        vd = _stub(outputs)
        vd.used = []
        vd._predict = lambda model, frame, conf: (vd.used.append(model), [o for o in outputs if o[1] >= conf])[1]
        vd.fast_model, vd.fast_model_name = "light-weights", "yolo26s.pt"
        return vd

    def test_fast_requests_use_the_lighter_model(self):
        vd = self._detector()
        _, dets, stats = vd.detect(FRAME.copy(), quality="fast")
        self.assertEqual(vd.used[-1], "light-weights")
        self.assertEqual(stats["model"], "yolo26s.pt")
        self.assertEqual([d["class"] for d in dets], ["chair"])

    def test_other_qualities_keep_the_main_model(self):
        vd = self._detector()
        for quality in ("balanced", "high", None):
            _, _, stats = vd.detect(FRAME.copy(), quality=quality)
            self.assertIs(vd.used[-1], vd.model)
            self.assertEqual(stats["model"], "stub.pt")

    def test_extended_model_runs_half_as_often_at_fast_quality(self):
        for quality, expected in (("balanced", 2), ("fast", 1)):
            vd = self._detector()
            vd.ext_model, vd.ext_every = object(), 3
            runs = []
            vd._run_extended = lambda frame, primary: (runs.append(1), [])[1]
            for _ in range(6):
                vd.detect(FRAME.copy(), quality=quality)
            self.assertEqual(len(runs), expected, quality)


class TestSceneQuestions(unittest.TestCase):
    dets = [
        {"class": "person", "spatial": "ahead", "distance_m": 2.0, "severity": "caution", "is_obstacle": True},
        {"class": "person", "spatial": "left", "distance_m": 3.0, "severity": "safe", "is_obstacle": True},
        {"class": "dining table", "spatial": "right", "distance_m": 1.5, "severity": "caution", "is_obstacle": True},
        {"class": "door", "spatial": "left", "distance_m": 4.0, "severity": "safe", "is_obstacle": True},
        {"class": "person", "spatial": "ahead", "distance_m": 0.6, "severity": "safe", "is_user": True},
    ]

    def setUp(self):
        self.qa = SceneQAEngine()

    def test_counting_ignores_the_user(self):
        self.assertIn("2 people", self.qa.answer_question("How many people are there?", detections=self.dets))
        self.assertIn("do not see any dogs", self.qa.answer_question("how many dogs are there", detections=self.dets))

    def test_locating_uses_everyday_names(self):
        ans = self.qa.answer_question("Where is the door?", detections=self.dets)
        self.assertIn("door on your left", ans)
        self.assertIn("4", ans)
        self.assertIn("table on your right", self.qa.answer_question("Is there a table?", detections=self.dets))
        self.assertIn("do not see a wallet", self.qa.answer_question("where is my wallet", detections=self.dets))
        bus = [{"class": "bus", "spatial": "ahead", "distance_m": 3.7, "severity": "safe", "is_obstacle": True}]
        self.assertIn("bus ahead of you", self.qa.answer_question("Where is the bus?", detections=bus))

    def test_nearest_and_side(self):
        self.assertIn("table", self.qa.answer_question("What is nearest to me?", detections=self.dets))
        left = self.qa.answer_question("What is on my left?", detections=self.dets)
        self.assertIn("person", left)
        self.assertIn("door", left)
        self.assertNotIn("table", left)

    def test_lighting_is_measured_not_assumed(self):
        self.assertIn("dark", self.qa.answer_question("Is it dark?", brightness=20).lower())
        self.assertIn("bright", self.qa.answer_question("Is it dark?", brightness=160).lower())
        self.assertIn("camera", self.qa.answer_question("Is it dark?").lower())

    def test_text_reader_only_when_needed(self):
        self.assertTrue(self.qa.needs_text("What does the sign say?"))
        self.assertTrue(self.qa.needs_text("Describe the scene"))
        self.assertFalse(self.qa.needs_text("How many chairs are there?"))


class TestApiAndPages(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.amcs = amcs
        cls.client = amcs.app.test_client()

    @staticmethod
    def _image():
        ok, buf = cv2.imencode(".jpg", FRAME)
        return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

    def test_health_reports_models_and_session(self):
        data = self.client.get("/api/health").get_json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["version"], config.APP_VERSION)
        self.assertTrue(data["detector"]["ready"])
        self.assertTrue(data["detector"]["model"].endswith(".pt"))
        self.assertEqual(data["detector"]["quality_profiles"], QUALITY_IMAGE_SIZES)
        self.assertIn("frames", data["session"])

    def test_detect_accepts_quality_and_reports_timing(self):
        before = self.client.get("/api/health").get_json()["session"]["frames"]
        for quality, size in (("fast", 512), ("high", 960), ("bogus", 640), (None, 640)):
            res = self.client.post("/api/detect", json={"image": self._image(), "quality": quality})
            self.assertEqual(res.status_code, 200)
            stats = res.get_json()["stats"]
            self.assertEqual(stats["image_size"], size)
            self.assertGreater(stats["detect_ms"], 0)
            self.assertIn("low_light", stats)
        self.assertEqual(self.client.get("/api/health").get_json()["session"]["frames"], before + 4)

    def test_scene_question_skips_text_reader_for_object_questions(self):
        with mock.patch.object(self.amcs.ocr_translator, "extract_text", return_value=([], "EXIT")) as ocr:
            res = self.client.post("/api/scene_qa", json={"question": "How many chairs?", "image": self._image()})
            self.assertEqual(res.status_code, 200)
            ocr.assert_not_called()
            res = self.client.post("/api/scene_qa", json={"question": "What does the sign say?", "image": self._image()})
            self.assertIn("EXIT", res.get_json()["answer_en"])
            ocr.assert_called_once()

    def test_assist_page_has_live_panels(self):
        html = self.client.get("/").get_data(as_text=True)
        for element_id in ["proximity-radar", "object-list", "guidance-history", "tele-objects", "tele-nearest",
                           "tele-latency", "tele-model", "low-light-chip", "spatial-ahead-val", "detected-count"]:
            self.assertIn(f'id="{element_id}"', html)
        self.assertIn("js/panels.js", html)
        self.assertLess(html.index("js/panels.js"), html.index("js/main.js"))

    def test_settings_offer_automatic_quality_and_lite_mode(self):
        html = self.client.get("/settings").get_data(as_text=True)
        self.assertIn('name="cfg-quality" value="auto"', html)
        self.assertIn('id="cfg-lite"', html)
        for value in ("auto", "on", "off"):
            self.assertIn(f'<option value="{value}">', html)

    def test_health_reports_fast_model(self):
        self.assertIn("fast_model", self.client.get("/api/health").get_json()["detector"])

    def test_settings_offer_detection_quality_and_show_loaded_model(self):
        html = self.client.get("/settings").get_data(as_text=True)
        for value in QUALITY_IMAGE_SIZES:
            self.assertIn(f'name="cfg-quality" value="{value}"', html)
        self.assertIn(self.amcs.vision_detector.model_name, html)

    def test_static_assets_are_served(self):
        for path in ["/static/js/panels.js", "/static/css/style.css"]:
            res = self.client.get(path)
            self.assertEqual(res.status_code, 200, path)
            res.close()


if __name__ == "__main__":
    unittest.main()
