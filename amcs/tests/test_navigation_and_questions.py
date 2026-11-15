"""
Tests for v3.2: saved places and turn-by-turn walking guidance relative to the way the user faces,
spoken scene questions ("what is in front of me", Hindi too), and portrait camera distances.
No test uses the internet: the map services are replaced by fakes.
Automatic Multilanguage Communication System - SKIT Jaipur.
"""

import math
import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from modules import navigation_assistant as nav_mod
from modules.navigation_assistant import (NavigationAssistant, SavedPlaces, clock_position, direction_phrase,
                                          relative_angle, round_distance)
from modules.scene_qa_engine import SceneQAEngine, normalize_question
from modules.vision_detector import VisionDetector, focal_length_px

LAT0, LON0 = 26.9000, 75.8000
M_LAT = 1 / 111195.0
M_LON = 1 / (111195.0 * math.cos(math.radians(LAT0)))


def at(north_m, east_m):
    """(lat, lon) a given number of meters north and east of the test origin."""
    return LAT0 + north_m * M_LAT, LON0 + east_m * M_LON


def l_shaped_route(origin, dest):
    """Fake router: 100 m north, then right (east) for 80 m. Ignores the requested points."""
    corner, end = at(100, 0), at(100, 80)
    return {
        "coords": [at(0, 0), at(50, 0), corner, at(100, 40), end],
        "steps": [
            {"type": "depart", "modifier": "straight", "name": "", "location": at(0, 0), "distance_m": 100},
            {"type": "turn", "modifier": "right", "name": "Station Road", "location": corner, "distance_m": 80},
            {"type": "arrive", "modifier": "left", "name": "", "location": end, "distance_m": 0},
        ],
        "distance_m": 180.0,
    }


class TestDirectionWords(unittest.TestCase):
    def test_relative_angle_and_clock(self):
        self.assertEqual(relative_angle(90, 0), 90)            # east while facing north: to the right
        self.assertEqual(relative_angle(0, 90), -90)
        self.assertEqual(relative_angle(10, 350), 20)          # across north
        self.assertEqual(abs(relative_angle(180, 0)), 180)
        self.assertEqual(clock_position(0), 12)
        self.assertEqual(clock_position(90), 3)
        self.assertEqual(clock_position(-90), 9)
        self.assertEqual(clock_position(60), 2)

    def test_phrases(self):
        self.assertEqual(direction_phrase(5), "straight ahead")
        self.assertEqual(direction_phrase(-40), "slightly to your left")
        self.assertEqual(direction_phrase(95), "to your right")
        self.assertEqual(direction_phrase(175), "behind you")
        self.assertEqual(round_distance(23), "25 meters")
        self.assertEqual(round_distance(2), "5 meters")
        self.assertEqual(round_distance(1260), "1.3 kilometers")


class TestTurnByTurn(unittest.TestCase):
    def setUp(self):
        self.router = mock.Mock(side_effect=l_shaped_route)
        self.nav = NavigationAssistant(router=self.router, geocoder=mock.Mock(return_value=None))
        self.nav.places.save("market", *at(100, 80))

    def plan(self, heading):
        return self.nav.plan_pedestrian_route(*at(0, 0), "market", heading=heading)

    def test_plan_uses_the_walking_route(self):
        plan = self.plan(heading=0)
        self.assertEqual(plan["status"], "success")
        self.assertEqual(plan["mode"], "route")
        self.assertEqual(plan["destination_source"], "saved")
        self.assertAlmostEqual(plan["total_distance_m"], 180, delta=3)
        self.assertEqual([s["maneuver"] for s in plan["steps"]], ["depart", "right", "arrive"])
        self.assertIn("Turn right onto Station Road", plan["steps"][1]["instruction"])
        self.assertIn("Walk straight ahead for 100 meters, then turn right onto Station Road", plan["initial_instruction"])

    def test_first_turn_depends_on_facing(self):
        self.assertIn("Turn around", self.plan(heading=180)["initial_instruction"])
        self.assertIn("Turn to your right", self.plan(heading=270)["initial_instruction"])
        self.assertIn("Hold the phone upright", self.plan(heading=None)["initial_instruction"])

    def test_guidance_along_the_route(self):
        self.plan(heading=0)
        far = self.nav.update_navigation_progress(*at(40, 0), heading=0)
        self.assertEqual(far["status"], "navigating")
        self.assertIn("Continue straight for 60 meters, then turn right", far["instruction"])
        soon = self.nav.update_navigation_progress(*at(72, 1), heading=2)
        self.assertIn("In 30 meters, turn right", soon["instruction"])
        now = self.nav.update_navigation_progress(*at(93, 0), heading=0)
        self.assertTrue(now["instruction"].startswith("Now, turn right"))
        self.assertTrue(now["urgent"])
        self.assertEqual(len({far["announce_key"], soon["announce_key"], now["announce_key"]}), 3)
        # Same advice a moment later: same key, so the phone does not repeat itself
        again = self.nav.update_navigation_progress(*at(95, 0), heading=0)
        self.assertEqual(again["announce_key"], now["announce_key"])
        # Round the corner, facing east: heading for the destination
        after = self.nav.update_navigation_progress(*at(100, 30), heading=90)
        self.assertIn("to reach market", after["instruction"])

    def test_facing_the_wrong_way_is_corrected(self):
        self.plan(heading=0)
        wrong = self.nav.update_navigation_progress(*at(30, 0), heading=180)
        self.assertIn("Turn around", wrong["instruction"])
        left = self.nav.update_navigation_progress(*at(31, 0), heading=90)   # facing east, route goes north
        self.assertIn("Turn to your left", left["instruction"])

    def test_leaving_the_route_plans_a_new_one(self):
        self.plan(heading=0)
        self.assertEqual(self.router.call_count, 1)
        self.nav.update_navigation_progress(*at(40, 60), heading=0)          # one bad GPS fix is ignored
        self.assertEqual(self.router.call_count, 1)
        res = self.nav.update_navigation_progress(*at(41, 60), heading=0)
        self.assertEqual(self.router.call_count, 2)
        self.assertIn("You have left the route", res["instruction"])
        self.assertTrue(res["urgent"])

    def test_starting_away_from_the_street(self):
        # The user stands 50 m west of where the street route begins (inside a campus)
        nav = NavigationAssistant(router=self.router, geocoder=mock.Mock(return_value=None))
        nav.places.save("market", *at(100, 80))
        plan = nav.plan_pedestrian_route(*at(0, -50), "market", heading=90)
        self.assertAlmostEqual(plan["total_distance_m"], 230, delta=4)
        self.assertIn("Walk straight ahead for 50 meters, then turn left", plan["initial_instruction"])
        self.assertEqual([st["maneuver"] for st in plan["steps"]], ["depart", "left", "right", "arrive"])
        for _ in range(3):
            res = nav.update_navigation_progress(*at(0, -48), heading=90)
            self.assertNotIn("left the route", res["instruction"])
        self.assertEqual(self.router.call_count, 1)

    def test_arrival(self):
        self.plan(heading=0)
        done = self.nav.update_navigation_progress(*at(98, 78), heading=90)
        self.assertEqual(done["status"], "arrived")
        self.assertIn("You have arrived at market", done["instruction"])
        self.assertIn("on your left", done["instruction"])
        self.assertIsNone(self.nav.update_navigation_progress(*at(98, 78)))

    def test_poor_gps_widens_the_arrival_circle(self):
        self.plan(heading=0)
        self.assertEqual(self.nav.update_navigation_progress(*at(100, 62))["status"], "navigating")
        self.assertEqual(self.nav.update_navigation_progress(*at(100, 62), accuracy=25)["status"], "arrived")

    def test_without_a_route_guides_by_clock_direction(self):
        nav = NavigationAssistant(router=mock.Mock(side_effect=OSError("offline")))
        nav.places.save("bus stop", *at(0, 200))
        plan = nav.plan_pedestrian_route(*at(0, 0), "bus stop", heading=0)
        self.assertEqual(plan["mode"], "direct")
        self.assertIn("straight towards it", plan["initial_instruction"])
        up = nav.update_navigation_progress(*at(0, 10), heading=0)
        self.assertIn("bus stop is at 3 o'clock", up["instruction"])
        up = nav.update_navigation_progress(*at(0, 10), heading=90)
        self.assertIn("12 o'clock", up["instruction"])

    def test_too_far_and_unknown(self):
        nav = NavigationAssistant(router=self.router, geocoder=mock.Mock(return_value=None))
        nav.places.save("delhi", 28.61, 77.21)
        self.assertIn("too far", nav.plan_pedestrian_route(*at(0, 0), "delhi")["message"])
        bad = nav.plan_pedestrian_route(*at(0, 0), "zzz nowhere")
        self.assertEqual(bad["status"], "error")
        self.assertIn("delhi", bad["message"])                             # lists the saved places
        self.router.assert_not_called()


class TestDestinations(unittest.TestCase):
    def test_saved_places_persist(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "places.json")
            places = SavedPlaces(path)
            places.save("  Home ", 26.91, 75.81)
            places.save("Office", 26.92, 75.82)
            again = SavedPlaces(path)
            self.assertEqual([p["name"] for p in again.all()], ["Home", "Office"])
            self.assertEqual(again.find("take me home")["name"], "Home")
            self.assertEqual(again.find("my office")["name"], "Office")
            self.assertTrue(again.remove("HOME"))
            self.assertFalse(SavedPlaces(path).remove("home"))
            with self.assertRaises(ValueError):
                places.save("   ", 1, 1)

    def test_order_saved_then_campus_then_map(self):
        geocoder = mock.Mock(return_value={"name": "Hawa Mahal", "lat": 26.9239, "lon": 75.8267, "description": "Jaipur"})
        nav = NavigationAssistant(geocoder=geocoder)
        nav.places.save("library", *at(1, 1))
        self.assertEqual(nav.resolve_destination("library")[0], "saved:library")
        self.assertEqual(nav.resolve_destination("girls hostel")[0], "girls_hostel")
        self.assertEqual(nav.resolve_destination("boys hostel")[0], "boys_hostel")
        self.assertEqual(nav.resolve_destination("hawa mahal"), (None, None))   # no position: no search
        geocoder.assert_not_called()
        key, place = nav.resolve_destination("hawa mahal", near=at(0, 0))
        self.assertEqual((key, place["name"]), ("search", "Hawa Mahal"))
        geocoder.side_effect = OSError("offline")
        self.assertEqual(nav.resolve_destination("city palace", near=at(0, 0)), (None, None))
        offline = NavigationAssistant(online=False, geocoder=geocoder)
        geocoder.reset_mock()
        self.assertEqual(offline.resolve_destination("city palace", near=at(0, 0)), (None, None))
        geocoder.assert_not_called()


class TestWhereAmI(unittest.TestCase):
    def test_street_and_nearest_place(self):
        reverse = mock.Mock(return_value={"road": "MI Road", "area": "C-Scheme", "city": "Jaipur"})
        nav = NavigationAssistant(reverse_geocoder=reverse)
        nav.places.save("home", *at(0, 100))
        ans = nav.describe_position(*at(0, 0), heading=0)
        self.assertIn("You are on MI Road, in C-Scheme, Jaipur.", ans)
        self.assertIn("home is 100 meters away, at 3 o'clock", ans)
        self.assertIn("to the east", nav.describe_position(*at(0, 0)))
        self.assertIn("You are at home", nav.describe_position(*at(0, 95)))

    def test_offline_and_unknown(self):
        nav = NavigationAssistant(reverse_geocoder=mock.Mock(side_effect=OSError("offline")))
        ans = nav.describe_position(*at(0, 0), accuracy=80)
        self.assertIn("cannot name this place", ans)
        self.assertIn("only accurate to about 80 meters", ans)

    def test_api(self):
        import app as amcs
        client = amcs.app.test_client()
        self.assertEqual(client.post("/api/where_am_i", json={}).status_code, 400)
        with mock.patch.object(amcs.navigation_assistant, "describe_position", return_value="You are on MI Road."):
            data = client.post("/api/where_am_i", json={"lat": LAT0, "lon": LON0, "lang": "en"}).get_json()
        self.assertEqual(data["answer_translated"], "You are on MI Road.")


class TestPlacesAndNavigationApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.amcs = amcs
        cls.client = amcs.app.test_client()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        nav = NavigationAssistant(places_file=os.path.join(self.tmp.name, "p.json"), router=l_shaped_route,
                                  geocoder=mock.Mock(return_value=None))
        patcher = mock.patch.object(self.amcs, "navigation_assistant", nav)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_save_list_delete(self):
        self.assertEqual(self.client.post("/api/places", json={"name": "home"}).status_code, 400)  # no position
        self.assertEqual(self.client.post("/api/places", json={"lat": 1, "lon": 1}).status_code, 400)
        self.assertEqual(self.client.post("/api/places", json={"name": "home", "lat": 95, "lon": 1}).status_code, 400)
        res = self.client.post("/api/places", json={"name": "home", "lat": LAT0, "lon": LON0})
        self.assertEqual(res.status_code, 200)
        self.assertIn("Saved this place as home", res.get_json()["message"])
        self.assertEqual([p["name"] for p in self.client.get("/api/places").get_json()["places"]], ["home"])
        self.assertEqual(self.client.delete("/api/places", json={"name": "home"}).status_code, 200)
        self.assertEqual(self.client.delete("/api/places", json={"name": "home"}).status_code, 404)

    def test_plan_needs_a_real_position(self):
        res = self.client.post("/api/navigate/plan", json={"destination": "library"})
        self.assertEqual(res.status_code, 400)
        self.assertEqual(res.get_json()["error"], "location_unknown")

    def test_plan_and_follow_with_heading(self):
        self.client.post("/api/places", json={"name": "market", "lat": at(100, 80)[0], "lon": at(100, 80)[1]})
        lat, lon = at(0, 0)
        plan = self.client.post("/api/navigate/plan", json={"destination": "market", "origin_lat": lat,
                                                             "origin_lon": lon, "heading": 180, "lang": "en"}).get_json()
        self.assertEqual(plan["status"], "success")
        self.assertIn("Turn around", plan["instruction_translated"])
        lat, lon = at(40, 0)
        up = self.client.post("/api/navigate/update", json={"lat": lat, "lon": lon, "heading": 0, "accuracy": 8,
                                                            "lang": "en"}).get_json()
        self.assertEqual(up["status"], "navigating")
        self.assertIn("then turn right", up["instruction_translated"])
        with mock.patch.object(self.amcs.ocr_translator, "translate_text", return_value="hindi") as tr:
            again = self.client.post("/api/navigate/update", json={"lat": lat, "lon": lon, "heading": 0, "lang": "hi",
                                                                   "announced_key": up["announce_key"]}).get_json()
            tr.assert_not_called()                     # already spoken: nothing to translate
            self.assertEqual(again["announce_key"], up["announce_key"])

    def test_unknown_place_message_is_spoken(self):
        lat, lon = at(0, 0)
        res = self.client.post("/api/navigate/plan", json={"destination": "zzz", "origin_lat": lat,
                                                            "origin_lon": lon}).get_json()
        self.assertEqual(res["status"], "error")
        self.assertIn("could not find", res["instruction_translated"])


class TestSpokenQuestions(unittest.TestCase):
    dets = [
        {"class": "person", "spatial": "ahead", "distance_m": 2.0, "severity": "caution", "confidence": 0.9},
        {"class": "chair", "spatial": "ahead", "distance_m": 3.5, "confidence": 0.8},
        {"class": "bench", "spatial": "left", "distance_m": 3.1, "confidence": 0.7},
        {"class": "car", "spatial": "right", "distance_m": 6.0, "confidence": 0.7},
        {"class": "person", "spatial": "ahead", "distance_m": 0.6, "is_user": True},
    ]

    def setUp(self):
        self.qa = SceneQAEngine()

    def test_front_answers_lead_with_what_is_ahead(self):
        for q in ("What is in front of me?", "what is ahead", "saamne kya hai", "मेरे सामने क्या है?", "aage kya hai"):
            ans = self.qa.answer_question(q, detections=self.dets)
            self.assertTrue(ans.startswith("Ahead of you: a person at about 2 meters and a chair at about 3.5 meters."), (q, ans))
            self.assertIn("On your left: a bench.", ans)
            self.assertIn("On your right: a car.", ans)

    def test_front_with_nothing_there_is_honest(self):
        self.assertEqual(self.qa.answer_question("what is in front of me", detections=[]),
                         "I do not see any objects in front of you.")
        side_only = [{"class": "bench", "spatial": "left", "distance_m": 2.0}]
        self.assertIn("Nothing is directly in front of you", self.qa.answer_question("what is ahead", detections=side_only))
        hazard = [{"class": "car", "spatial": "ahead", "distance_m": 1.0, "severity": "hazard"}]
        self.assertTrue(self.qa.answer_question("what is ahead", detections=hazard).startswith("Careful."))

    def test_what_is_this(self):
        held = [{"class": "person", "spatial": "ahead", "distance_m": 0.8, "confidence": 0.9},
                {"class": "bottle", "spatial": "ahead", "distance_m": 0.4, "confidence": 0.8}]
        self.assertEqual(self.qa.answer_question("What am I holding?", detections=held), "It looks like a bottle.")
        self.assertEqual(self.qa.answer_question("ये क्या है", detections=held[1:]), "It looks like a bottle.")
        unsure = [dict(held[1], confidence=0.35)]
        self.assertIn("It might be a bottle", self.qa.answer_question("what is this", detections=unsure))
        self.assertIn("says: Paracetamol", self.qa.answer_question("what is this", ocr_text="Paracetamol 500"))
        self.assertIn("Hold it about 30 centimeters", self.qa.answer_question("what is this"))

    def test_who_is_in_front(self):
        people = [{"class": "person", "spatial": "ahead", "distance_m": 2.0,
                   "face_info": {"recognized": True, "name": "Asha", "role": "Sister"}},
                  {"class": "person", "spatial": "left", "distance_m": 3.0, "face_info": {"recognized": False}}]
        ans = self.qa.answer_question("Who is in front of me?", detections=people)
        self.assertIn("Asha, your sister, is ahead of you", ans)
        self.assertIn("a person I do not recognise, on your left", ans)
        self.assertEqual(self.qa.answer_question("कौन है", detections=[]), "I do not see anyone in front of you.")
        self.assertIn("I do not recognise them", self.qa.answer_question("who is there", detections=people[1:]))
        self.assertTrue(self.qa.needs_faces("who is in front of me"))
        self.assertFalse(self.qa.needs_faces("what is in front of me"))

    def test_hindi_and_hinglish(self):
        self.assertEqual(normalize_question("मेरे बाएं क्या है?"), "mere left kya hai")
        self.assertIn("bench", self.qa.answer_question("baayen kya hai", detections=self.dets))
        self.assertIn("car", self.qa.answer_question("दाएं क्या है", detections=self.dets))
        self.assertIn("2 people", self.qa.answer_question("kitne log hain", detections=self.dets[:1] * 2))
        self.assertIn("clear", self.qa.answer_question("rasta saaf hai?", detections=[]))

    def test_unclear_question_says_so(self):
        ans = self.qa.answer_question("banana telephone", detections=self.dets)
        self.assertTrue(ans.startswith("I am not sure what you asked"))
        self.assertIn("Ahead of you", ans)

    def test_sign_question_in_front_still_reads_text(self):
        self.assertIn("EXIT", self.qa.answer_question("what does the sign in front of me say", ocr_text="EXIT"))
        self.assertTrue(self.qa.needs_text("saamne kya hai"))
        self.assertTrue(self.qa.needs_text("what am i holding"))


class TestQuestionApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app as amcs
        cls.amcs = amcs
        cls.client = amcs.app.test_client()

    def test_device_voice_needs_no_server_audio(self):
        with mock.patch.object(self.amcs.speech_synthesizer, "synthesize_later", return_value="/api/speech/x.mp3") as tts:
            data = self.client.post("/api/scene_qa", json={"question": "Is the path clear?", "tts": False}).get_json()
            tts.assert_not_called()
            self.assertIsNone(data["audio_url"])
            self.assertTrue(data["fallback_browser"])
            data = self.client.post("/api/scene_qa", json={"question": "Is the path clear?"}).get_json()
            self.assertEqual(data["audio_url"], "/api/speech/x.mp3")

    def test_other_languages_are_translated_first(self):
        def fake_translate(text, target_language="en", source_language="auto"):
            return "what is in front of me" if target_language == "en" else text
        with mock.patch.object(self.amcs.ocr_translator, "translate_text", side_effect=fake_translate) as tr, \
             mock.patch.object(self.amcs.speech_synthesizer, "synthesize_later", return_value=None):
            data = self.client.post("/api/scene_qa", json={"question": "ನನ್ನ ಮುಂದೆ ಏನಿದೆ", "lang": "kn", "tts": False}).get_json()
            self.assertEqual(data["question_en"], "what is in front of me")
            self.assertIn("in front of you", data["answer_en"])
            tr.reset_mock()
            # Hindi the engine understands is not sent to the translator
            data = self.client.post("/api/scene_qa", json={"question": "सामने क्या है", "lang": "en", "tts": False}).get_json()
            tr.assert_not_called()
            self.assertEqual(data["question_en"], "सामने क्या है")

    def test_who_question_runs_face_matching(self):
        person = {"class": "person", "spatial": "ahead", "distance_m": 2.0, "bbox": [10, 10, 100, 300], "confidence": 0.9}
        img = self._image()
        with mock.patch.object(self.amcs.vision_detector, "detect", return_value=(None, [dict(person)], {})), \
             mock.patch.object(self.amcs.face_engine, "identify_person",
                               return_value={"recognized": True, "name": "Asha", "role": "Sister"}) as faces, \
             mock.patch.object(self.amcs.speech_synthesizer, "synthesize_later", return_value=None):
            data = self.client.post("/api/scene_qa", json={"question": "who is in front of me", "image": img}).get_json()
            faces.assert_called_once()
            self.assertIn("Asha", data["answer_en"])
            faces.reset_mock()
            self.client.post("/api/scene_qa", json={"question": "how many people", "image": img})
            faces.assert_not_called()

    @staticmethod
    def _image():
        import base64
        import cv2
        ok, buf = cv2.imencode(".jpg", np.full((120, 160, 3), 90, np.uint8))
        return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode()


class TestPhoneProfile(unittest.TestCase):
    def test_mobile_uses_the_light_model_at_640(self):
        import app as amcs
        det = amcs.vision_detector
        if det.fast_model is None:
            self.skipTest("light model not installed")
        frame = np.full((480, 640, 3), 90, np.uint8)
        _, _, stats = det.detect(frame, quality="mobile")
        self.assertEqual(stats["image_size"], 640)
        self.assertEqual(stats["model"], det.fast_model_name)
        _, _, stats = det.detect(frame, quality="balanced")
        self.assertEqual(stats["model"], det.model_name)


class TestPortraitDistance(unittest.TestCase):
    def test_focal_length_uses_the_long_side(self):
        self.assertAlmostEqual(focal_length_px(640, 480), focal_length_px(480, 640))

    def test_portrait_and_landscape_agree(self):
        det = VisionDetector.__new__(VisionDetector)
        box_h = 200
        landscape = det.estimate_distance("person", box_h, 480, bbox=[200, 100, 280, 300], frame_width=640)
        portrait = det.estimate_distance("person", box_h, 640, bbox=[200, 100, 280, 300], frame_width=480)
        self.assertAlmostEqual(landscape, portrait, delta=0.05)
        # A 1.7 m person 200 px tall in a 640 px-wide 70 degree view is about 3.9 m away
        self.assertAlmostEqual(landscape, 1.7 * focal_length_px(640, 480) / box_h, delta=0.4)


if __name__ == "__main__":
    unittest.main()
