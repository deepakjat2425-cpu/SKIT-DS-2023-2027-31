"""
Automatic Multilanguage Communication System - Main Flask Application.
Department of Computer Science & Engineering
Swami Keshvanand Institute of Technology, Management & Gramothan, Jaipur.

Final Year Project
Presented By:
- Amit Tiwari (23ESKCX010)
- Deepak Choudhary (23ESKCX026)
- Deependra Singh (23ESKCX027)
Supervised By: Vikas (Assistant Professor-2, Dept. of CSE)
"""

import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory

import config
from modules.vision_detector import VisionDetector, QUALITY_IMAGE_SIZES
from modules.nlp_describer import NLPDescriber
from modules.ocr_translator import OCRTranslator
from modules.speech_synthesizer import SpeechSynthesizer
from modules.location_tracker import LocationTracker
from modules.currency_recognizer import CurrencyRecognizer
from modules.trajectory_tracker import TrajectoryTracker
from modules.face_social_engine import FaceSocialEngine
from modules.traffic_crossing_assistant import TrafficCrossingAssistant
from modules.medicine_reader import MedicineReader
from modules.ground_hazard_detector import GroundHazardDetector
from modules.navigation_assistant import NavigationAssistant
from modules.scene_qa_engine import SceneQAEngine, normalize_question
from modules.item_finder_engine import ItemFinderEngine
from modules.indoor_door_locator import IndoorDoorLocator
from modules.acoustic_hazard_detector import AcousticHazardDetector
from modules.color_clothing_matcher import ColorClothingMatcher
from modules.teleassist_engine import TeleAssistEngine
from modules.viva_evaluation_sandbox import VivaEvaluationSandbox
from modules.scene_stabilizer import TemporalConfirmer

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("AMCS_SECRET_KEY") or os.urandom(24).hex()
# Camera frames arrive as base64 JPEGs; cap request bodies to keep the server responsive
app.config["MAX_CONTENT_LENGTH"] = config.MAX_UPLOAD_MB * 1024 * 1024

# Initialize Core AI & Vision Modules
print("[System] Initializing Automatic Multilanguage Communication System modules...")
vision_detector = VisionDetector()
nlp_describer = NLPDescriber()
ocr_translator = OCRTranslator()
speech_synthesizer = SpeechSynthesizer()
location_tracker = LocationTracker()
currency_recognizer = CurrencyRecognizer()
trajectory_tracker = TrajectoryTracker()
face_engine = FaceSocialEngine()
traffic_assistant = TrafficCrossingAssistant()
medicine_reader = MedicineReader()
ground_hazard_detector = GroundHazardDetector()
navigation_assistant = NavigationAssistant(places_file=str(config.BASE_DIR / "saved_places.json"))
scene_qa_engine = SceneQAEngine()
item_finder_engine = ItemFinderEngine()
indoor_door_locator = IndoorDoorLocator()
acoustic_hazard_detector = AcousticHazardDetector()
color_clothing_matcher = ColorClothingMatcher()
teleassist_engine = TeleAssistEngine()
viva_sandbox = VivaEvaluationSandbox()
# Heuristic ground/door findings must appear in 3 of the last 4 frames before they are announced
item_finder_engine.use_heuristic_fallback = vision_detector.ext_model is None
ground_confirmer = TemporalConfirmer(window=4, need=3)
door_confirmer = TemporalConfirmer(window=4, need=3)
_stage_errors_logged = set()


def run_stage(name, fn, default):
    """Runs one optional pipeline stage; a failure there must never break object detection."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001 - deliberately broad, logged once per stage
        if name not in _stage_errors_logged:
            _stage_errors_logged.add(name)
            app.logger.exception("Pipeline stage '%s' failed (%s); continuing without it", name, exc)
        return default


def is_stable_detection(d):
    """Seen in at least two frames (or runs of the extended model), or confident enough on its own."""
    if d.get("source") == "extended":
        return bool(d.get("confirmed"))
    return d.get("hits", 1) >= 2 or d.get("confidence", 0) >= 0.6

# In-memory session state for detections and caregiver logs
recent_detections_log = []
MAX_LOG_ITEMS = 30
SERVER_STARTED_AT = time.time()
# Running tally of the live session (shown on the Assist page and in /api/health)
session_stats = {"frames": 0, "objects": 0, "hazard_alerts": 0, "detect_ms": 0.0}


# Online translation takes about a second per sentence. The live loop translates only what the
# browser is about to speak (two sentences side by side), and reuses the last translation otherwise.
translation_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="translate")
last_live_translation = {"lang": None, "description": "", "speech": ""}


def live_speech_due(scene_info, force_speech, announced_key, announced_hazard):
    """
    Will the browser speak this frame's guidance? It mirrors the browser's own rules (main.js,
    decideAndPlaySpeech): always on request, for a new / closer / due hazard, and when the scene
    differs from what the user was last told.
    """
    if force_speech or announced_key is None:
        return True
    if scene_info.get("is_urgent") and scene_info.get("hazard_key"):
        told = announced_hazard if isinstance(announced_hazard, dict) else {}
        if scene_info["hazard_key"] != told.get("key") or told.get("due"):
            return True
        distance, told_distance = scene_info.get("hazard_distance"), told.get("distance")
        if isinstance(distance, (int, float)) and isinstance(told_distance, (int, float)):
            return distance < told_distance - 0.5
        return True
    return scene_info.get("scene_key", "") != announced_key


def translate_pair(description, speech, lang):
    """Translates the full description and the short update at the same time."""
    jobs = {text: translation_pool.submit(ocr_translator.translate_text, text, target_language=lang,
                                          source_language="en")
            for text in {description, speech} if text}
    done = {text: job.result() for text, job in jobs.items()}
    return done.get(description, ""), done.get(speech, "")


def detection_quality(value):
    """Validated quality profile requested by the client: 'fast', 'mobile', 'balanced' or 'high'."""
    return value if value in QUALITY_IMAGE_SIZES else config.DEFAULT_DETECTION_QUALITY

def add_log(event_type, description, severity="info"):
    """Appends to the caregiver activity feed."""
    entry = {
        "timestamp": time.strftime("%H:%M:%S"),
        "type": event_type,
        "description": description,
        "severity": severity
    }
    recent_detections_log.insert(0, entry)
    if len(recent_detections_log) > MAX_LOG_ITEMS:
        recent_detections_log.pop()

# Initial log entry
add_log("system", "Automatic Multilanguage Communication System initialized successfully.", "info")


@app.context_processor
def inject_globals():
    """Values every template can use (header, footer, cache-busting)."""
    return {
        "project_title": config.PROJECT_TITLE,
        "college": config.COLLEGE_NAME,
        "department": config.DEPARTMENT,
        "app_version": config.APP_VERSION,
        "asset_version": config.APP_VERSION,
        "geofence_radius": config.GEOFENCE_RADIUS_METERS,
        "detector_model": vision_detector.model_name or "unavailable",
        "detector_extended": vision_detector.ext_model_name,
    }


@app.errorhandler(413)
def payload_too_large(_err):
    return jsonify({"status": "error", "error": f"Upload too large (limit {config.MAX_UPLOAD_MB} MB)."}), 413


@app.errorhandler(404)
def not_found(err):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Endpoint not found."}), 404
    return err


@app.errorhandler(500)
def server_error(err):
    if request.path.startswith("/api/"):
        return jsonify({"status": "error", "error": "Internal server error."}), 500
    return err


@app.route("/sw.js")
def service_worker():
    """Serve the service worker from the site root so it can control every page."""
    response = send_from_directory(config.STATIC_DIR, "sw.js", mimetype="application/javascript")
    response.headers["Cache-Control"] = "no-cache"
    return response


# ==========================================
# PAGE ROUTES
# ==========================================

@app.route("/")
def index():
    """User Assistive Interface: Live Camera, Real-Time Vision & Audio, OCR."""
    return render_template(
        "index.html",
        project_title=config.PROJECT_TITLE,
        college=config.COLLEGE_NAME,
        languages=config.SUPPORTED_LANGUAGES,
        default_lang=config.DEFAULT_LANGUAGE
    )

@app.route("/caregiver")
def caregiver():
    """Caregiver Safety & Monitoring Dashboard."""
    return render_template(
        "caregiver.html",
        project_title=config.PROJECT_TITLE,
        college=config.COLLEGE_NAME,
        location=location_tracker.get_status_payload()
    )

@app.route("/ocr")
def ocr_page():
    """Dedicated Signboard & Text Multilingual Translation View."""
    return render_template(
        "text_translator.html",
        project_title=config.PROJECT_TITLE,
        languages=config.SUPPORTED_LANGUAGES,
        default_lang=config.DEFAULT_LANGUAGE
    )

@app.route("/settings")
def settings():
    """System & User Configuration."""
    return render_template(
        "settings.html",
        project_title=config.PROJECT_TITLE,
        languages=config.SUPPORTED_LANGUAGES,
        config=config
    )

@app.route("/about")
def about():
    """Project Details, College & Team Information."""
    return render_template(
        "about.html",
        project_title=config.PROJECT_TITLE,
        college=config.COLLEGE_NAME,
        department=config.DEPARTMENT,
        team=config.TEAM_MEMBERS,
        supervisor=config.SUPERVISOR
    )

@app.route("/sandbox")
def sandbox_page():
    """Examiner Viva Evaluation Sandbox & Benchmark Dashboard."""
    return render_template(
        "sandbox.html",
        project_title=config.PROJECT_TITLE,
        college=config.COLLEGE_NAME,
        department=config.DEPARTMENT,
        team=config.TEAM_MEMBERS,
        supervisor=config.SUPERVISOR,
        scenarios=viva_sandbox.get_available_scenarios(),
        metrics=viva_sandbox.run_benchmark_suite()
    )


# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route("/api/detect", methods=["POST"])
def api_detect():
    """
    Core Vision-to-Speech Processing Pipeline:
    1. Decodes incoming webcam frame.
    2. Runs YOLO26 (+ extended-vocabulary YOLOE) object & obstacle detection.
    3. NLP module generates contextual natural scene description.
    4. Translates description to user's selected language.
    5. Synthesizes speech (gTTS / pyttsx3) and returns audio URL.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)
    mode = data.get("mode", "vision")  # 'vision', 'ocr', or 'hybrid'
    force_speech = data.get("force_speech", False)
    # "user" = front / laptop camera facing the person (left and right are mirrored)
    camera_facing = data.get("camera_facing") if data.get("camera_facing") in ("user", "environment") else None
    # The browser can speak with a natural device voice; then server-side TTS is skipped
    want_server_tts = data.get("tts", True) is not False
    # What the user was last told (scene_key); lets routine updates mention only what is new
    announced_key = data.get("announced_key") if isinstance(data.get("announced_key"), str) else None
    # The hazard the user was last warned about: {"key", "distance", "due"} (due = time to repeat it)
    announced_hazard = data.get("announced_hazard")

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    # 1. Decode Frame
    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    # 2. Vision Detection
    h_orig, w_orig = frame.shape[:2]
    detect_started = time.perf_counter()
    annotated_frame, detections, stats = vision_detector.detect(
        frame, extended=True if force_speech else None, camera_facing=camera_facing,
        quality=detection_quality(data.get("quality")))
    stats["detect_ms"] = round((time.perf_counter() - detect_started) * 1000.0, 1)

    object_boxes = [d["bbox"] for d in detections]

    # 2b. Dynamic Trajectory & Time-to-Collision (TTC) Tracking
    trajectory_tracker.set_view((w_orig, h_orig, camera_facing))
    detections, dynamic_hazard = run_stage(
        "trajectory", lambda: trajectory_tracker.update_tracks(detections), (detections, None))
    if dynamic_hazard:
        stats["dynamic_hazard"] = dynamic_hazard
    # The tracker steadies the distance of stationary objects; severity follows the steadied value
    stats.update(run_stage("severity", lambda: vision_detector.refresh_severity(detections), {}))

    session_stats["frames"] += 1
    session_stats["objects"] += len(detections)
    session_stats["detect_ms"] += stats["detect_ms"]

    # 2c. Face & Social Interaction Recognition
    def _faces():
        for d in detections:
            if d["class"].lower() in ["person", "human"]:
                f_info = face_engine.identify_person(frame, d["bbox"])
                if f_info:
                    d["face_info"] = f_info
                    if f_info.get("recognized"):
                        d["class"] = f_info["name"]
                        add_log("social", f"Identified contact: {f_info['name']} ({f_info['role']})", "info")
    run_stage("faces", _faces, None)

    # 2d. Traffic Light & Intersection Pedestrian Signal Analysis
    no_crossing = {"is_intersection": False, "advisory_en": "", "signal_state": None, "safe_to_cross": False}
    crossing_info = run_stage("crossing", lambda: traffic_assistant.analyze_intersection(frame, detections), no_crossing)
    stats["intersection"] = crossing_info

    # Optional base64 frame (only generated if client specifically asks)
    annotated_base64 = None
    if data.get("return_annotated_image", False):
        annotated_base64 = vision_detector.encode_base64_image(annotated_frame)

    # 3. NLP Scene Description
    # Automatic announcements only use objects seen in more than one frame (or very confident ones),
    # so a single misdetection is never spoken. An explicit "describe now" uses everything.
    speech_detections = detections if force_speech else [d for d in detections if is_stable_detection(d)]
    scene_info = nlp_describer.describe_scene(speech_detections, force_generate=force_speech,
                                              announced_key=announced_key)
    original_description = scene_info["text"]

    # Prepend pedestrian traffic signal advisory
    if crossing_info["is_intersection"] and crossing_info["advisory_en"]:
        if crossing_info["signal_state"] == "RED":
            scene_info["is_urgent"] = True
            original_description = f"{crossing_info['advisory_en']} {original_description}"
        elif crossing_info["safe_to_cross"]:
            original_description = f"{crossing_info['advisory_en']} {original_description}"

    # 3b. Ground Elevation Drop-Off, Curb & Pothole Hazard Detection
    ground_hazards = []
    if mode in ["vision", "hybrid"]:
        _, ground_hazards = run_stage(
            "ground", lambda: ground_hazard_detector.analyze_ground_plane(frame, object_boxes=object_boxes), (None, []))
        confirmed = ground_confirmer.update({g["category"] for g in ground_hazards})
        ground_hazards = [g for g in ground_hazards if g["category"] in confirmed]
        stats["ground_hazards"] = ground_hazards
        if ground_hazards:
            nearest_gh = ground_hazards[0]
            if nearest_gh["severity"] == "hazard" or nearest_gh["distance_m"] <= 1.4:
                scene_info["is_urgent"] = True
                scene_info["hazard_key"] = f"ground:{nearest_gh['category']}"
                scene_info["hazard_distance"] = nearest_gh["distance_m"]
                original_description = f"{nearest_gh['advice']} {original_description}"
                # Append ground hazard into detections list for AR HUD
                detections.append({
                    "class": f"⚠️ {nearest_gh['type']}",
                    "confidence": 0.88,
                    "bbox": [int(w_orig * 0.1), int(h_orig * 0.72), int(w_orig * 0.9), int(h_orig * 0.96)],
                    "center": [w_orig // 2, int(h_orig * 0.84)],
                    "spatial": "ahead",
                    "distance_m": nearest_gh["distance_m"],
                    "severity": nearest_gh["severity"],
                    "is_priority": True
                })
                add_log("hazard", f"Ground Hazard: {nearest_gh['description']}", "danger" if nearest_gh["severity"] == "hazard" else "warning")

    # 3c. Indoor Door, Handle & Room Sign Detection
    doors = []
    # The extended-vocabulary model recognises doors directly and far more reliably than the
    # edge heuristic, so the heuristic only runs when that model is not installed.
    if mode in ["vision", "hybrid"] and vision_detector.ext_model is None:
        _, doors = run_stage(
            "doors", lambda: indoor_door_locator.analyze_frame_for_doors(frame, object_boxes=object_boxes), (None, []))
        if "door" not in door_confirmer.update({"door"} if doors else set()):
            doors = []
        stats["doors"] = doors
        if doors:
            nearest_door = doors[0]
            if nearest_door["distance_m"] <= 2.2:
                scene_info["changed"] = True
                original_description = f"{nearest_door['guidance']} {original_description}"
                detections.append({
                    "class": f"🚪 Door ({nearest_door['state']})",
                    "confidence": 0.85,
                    "bbox": nearest_door["bbox"],
                    "center": [(nearest_door["bbox"][0] + nearest_door["bbox"][2]) // 2, (nearest_door["bbox"][1] + nearest_door["bbox"][3]) // 2],
                    "spatial": "ahead",
                    "distance_m": nearest_door["distance_m"],
                    "severity": "safe" if nearest_door["state"] == "open" else "caution",
                    "is_priority": True
                })

    # 4. Currency Recognition if in Currency Mode
    currency_data = None
    if mode == "currency":
        curr_res = currency_recognizer.detect(frame)
        currency_data = curr_res
        if curr_res["detected"]:
            original_description = curr_res["description_en"]
            scene_info["changed"] = True
            # Add currency highlight to detections list
            cx, cy, cw, ch = curr_res["bbox"]
            detections = [{
                "class": f"₹{curr_res['denomination']} Note",
                "confidence": curr_res["confidence"],
                "bbox": [cx, cy, cx + cw, cy + ch],
                "center": [cx + cw // 2, cy + ch // 2],
                "spatial": "ahead",
                "distance_m": 0.4,
                "severity": "safe",
                "is_priority": True
            }]
            add_log("currency", f"Identified Banknote: ₹{curr_res['denomination']} ({curr_res['label']})", "success")
        else:
            original_description = curr_res["description_en"]

    # 5. Medicine Blister Strip & Expiry Analysis if in Medicine Mode
    medicine_data = None
    if mode == "medicine":
        preprocessed = medicine_reader.preprocess_foil(frame)
        _, ocr_text = ocr_translator.extract_text(preprocessed if preprocessed is not None else frame)
        med_res = medicine_reader.analyze_strip(frame, raw_text=ocr_text)
        medicine_data = med_res
        stats["medicine"] = med_res
        if med_res["detected"]:
            original_description = med_res["advisory_en"]
            scene_info["changed"] = True
            if med_res.get("is_expired"):
                scene_info["is_urgent"] = True
            med_sev = "danger" if med_res.get("is_expired") else "success"
            add_log("medicine", f"Medicine: {med_res['full_name'] or 'Detected strip'} (Exp: {med_res['expiry_str'] or 'N/A'})", med_sev)
            # Add medicine detection to HUD overlay
            detections = [{
                "class": f"💊 {med_res['full_name'] or 'Medicine'} (Exp: {med_res['expiry_str'] or 'OK'})",
                "confidence": 0.95,
                "bbox": [int(w_orig * 0.15), int(h_orig * 0.2), int(w_orig * 0.85), int(h_orig * 0.8)],
                "center": [w_orig // 2, h_orig // 2],
                "spatial": "ahead",
                "distance_m": 0.3,
                "severity": "hazard" if med_res.get("is_expired") else "safe",
                "is_priority": True
            }]
        else:
            original_description = med_res["advisory_en"]

    # 6. OCR processing if in hybrid mode or ocr mode
    ocr_text = ""
    if mode in ["ocr", "hybrid"]:
        extracted_items, combined = ocr_translator.extract_text(frame)
        if combined:
            ocr_text = combined
            if mode == "ocr":
                original_description = f"Detected sign: {ocr_text}"

    # 7. Short routine update ("Now also a cup ahead."). If other stages added their own advice
    # (crossing signals, currency, medicine, signs...), the full description is spoken instead.
    speech_en = scene_info.get("speech_text", original_description)
    if original_description != scene_info["text"] and speech_en != "":
        speech_en = original_description

    # 7b. Will the browser speak this frame? Only then are translation and server speech worth
    # their time (about a second each). In the live vision mode a frame that will not be spoken
    # reuses the last translation, so the on-screen text stays in the user's language.
    speech_due = live_speech_due(scene_info, force_speech, announced_key, announced_hazard)
    live_vision = mode == "vision"

    # 8. Multilingual Translation
    translation_stale = False
    if target_lang == "hi" and mode == "currency" and currency_data and currency_data.get("detected"):
        translated_description = speech_translated = currency_data["description_hi"]
    elif target_lang == "hi" and mode == "medicine" and medicine_data and medicine_data.get("detected"):
        translated_description = speech_translated = medicine_data["advisory_hi"]
    elif target_lang == "en":
        translated_description, speech_translated = original_description, speech_en
    elif live_vision and not speech_due and last_live_translation["lang"] == target_lang:
        translated_description = last_live_translation["description"]
        speech_translated = last_live_translation["speech"]
        translation_stale = True
    else:
        translated_description, speech_translated = translate_pair(original_description, speech_en, target_lang)
        if live_vision:
            last_live_translation.update(lang=target_lang, description=translated_description, speech=speech_translated)

    # 9. Server speech, only for a frame the browser will speak and only when it has no voice of its
    # own for this language. Only the message it will play is made, in the background: the frame
    # answers at once and the audio link waits until the sound is ready (/api/speech).
    audio_info = {"audio_url": None, "fallback_browser": True}
    speech_audio_url = None
    if want_server_tts and speech_due:
        full = force_speech or scene_info["is_urgent"] or speech_translated == translated_description
        if full:
            speech_audio_url = speech_synthesizer.synthesize_later(translated_description, lang=target_lang)
            audio_info = {"audio_url": speech_audio_url, "fallback_browser": speech_audio_url is None}
        elif speech_translated:
            speech_audio_url = speech_synthesizer.synthesize_later(speech_translated, lang=target_lang)

    if scene_info["changed"] or force_speech:
        # Log significant events for caregiver (only when something actually changed)
        if scene_info["is_urgent"]:
            session_stats["hazard_alerts"] += 1
            add_log("hazard", f"Obstacle Alert: {original_description}", "danger")
        elif scene_info["changed"]:
            add_log("detection", f"Scene update: {original_description}", "info")

    return jsonify({
        "status": "success",
        "frame_size": [int(w_orig), int(h_orig)],
        "annotated_image": annotated_base64,
        "detections": detections,
        "stats": stats,
        "ground_hazards": ground_hazards,
        "description_en": original_description,
        "description_translated": translated_description,
        "target_lang": target_lang,
        "is_urgent": scene_info["is_urgent"],
        "priority": scene_info["priority"],
        "scene_key": scene_info.get("scene_key", ""),
        "speech_en": speech_en,
        "speech_translated": speech_translated,
        "speech_audio_url": speech_audio_url,
        "translation_stale": translation_stale,
        "hazard_key": scene_info.get("hazard_key", ""),
        "hazard_distance": scene_info.get("hazard_distance"),
        "audio_url": audio_info.get("audio_url"),
        "fallback_browser": audio_info.get("fallback_browser", False)
    })

@app.route("/api/speech/<name>", methods=["GET"])
def api_speech(name):
    """Live-loop audio made in the background; waits (up to 10 s) until the sound is ready."""
    if not re.fullmatch(r"speech_[0-9a-f]{32}\.mp3", name):
        return jsonify({"status": "error", "error": "Unknown audio file."}), 404
    if speech_synthesizer.wait_for(name) is None:
        return jsonify({"status": "error", "error": "Speech could not be generated."}), 404
    return send_from_directory(speech_synthesizer.cache_dir, name, mimetype="audio/mpeg")


@app.route("/api/ocr_translate", methods=["POST"])
def api_ocr_translate():
    """
    Dedicated Signboard & Text OCR Translation endpoint.
    Extracts text and translates into multiple languages simultaneously.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    requested = data.get("target_langs") or ["en", "hi", "es", "fr", "de", "zh-CN", "ja", "ar"]
    if not isinstance(requested, list):
        requested = [requested]
    # Keep only supported codes, without duplicates, preserving order
    target_langs = list(dict.fromkeys(l for l in requested if l in config.SUPPORTED_LANGUAGES)) or ["en"]

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    # Extract Text
    ocr_items, raw_text = ocr_translator.extract_text(frame)
    annotated = ocr_translator.draw_ocr_boxes(frame, ocr_items)
    annotated_base64 = vision_detector.encode_base64_image(annotated)

    if not raw_text:
        return jsonify({
            "status": "success",
            "annotated_image": annotated_base64,
            "raw_text": "",
            "translations": {},
            "message": "No readable text or signboard detected in frame."
        })

    # Translate into requested languages
    translations = {}
    for lang in target_langs:
        lang_meta = config.SUPPORTED_LANGUAGES.get(lang, {"name": lang, "flag": ""})
        t_text = ocr_translator.translate_text(raw_text, target_language=lang)
        translations[lang] = {
            "name": lang_meta["name"],
            "flag": lang_meta["flag"],
            "translated_text": t_text
        }

    add_log("ocr", f"Signboard read: '{raw_text}' -> Translated to {len(translations)} languages", "success")

    return jsonify({
        "status": "success",
        "annotated_image": annotated_base64,
        "raw_text": raw_text,
        "translations": translations
    })

@app.route("/api/location/update", methods=["POST"])
def api_location_update():
    """Receives live GPS coordinates from user device."""
    data = request.get_json(silent=True) or {}
    lat = data.get("latitude")
    lon = data.get("longitude")
    accuracy = data.get("accuracy", 10.0)
    label = data.get("location_name")

    if lat is None or lon is None:
        return jsonify({"error": "Latitude and Longitude required"}), 400

    success = location_tracker.update_location(lat, lon, accuracy, label)
    if success:
        # Check geofence status
        geo = location_tracker.check_geofence()
        if geo["breached"]:
            add_log("geofence", f"User outside safe perimeter! ({geo['distance_from_center_m']}m from center)", "warning")
        return jsonify({"status": "success", "geofence": geo})
    return jsonify({"error": "Failed to update location"}), 500

@app.route("/api/location/status", methods=["GET"])
def api_location_status():
    """Provides current user location, geofence, and activity feed for caregivers."""
    payload = location_tracker.get_status_payload()
    payload["activity_log"] = recent_detections_log
    return jsonify(payload)

@app.route("/api/map", methods=["GET"])
def api_map():
    """Returns interactive Folium map HTML snippet."""
    map_html = location_tracker.generate_folium_map()
    return map_html

@app.route("/api/currency_detect", methods=["POST"])
def api_currency_detect():
    """
    Dedicated Indian Banknote Denomination Identification Endpoint.
    Recognizes ₹10, ₹20, ₹50, ₹100, ₹200, ₹500 notes.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    # Also run quick OCR to look for numerical denomination tokens
    _, ocr_text = ocr_translator.extract_text(frame)
    res = currency_recognizer.detect(frame, ocr_text=ocr_text)

    if res["detected"]:
        if target_lang == "hi":
            spoken_text = res["description_hi"]
        elif target_lang != "en":
            spoken_text = ocr_translator.translate_text(res["description_en"], target_language=target_lang)
        else:
            spoken_text = res["description_en"]

        add_log("currency", f"Scanned: ₹{res['denomination']} note ({res['label']})", "success")
    else:
        spoken_text = res["description_hi"] if target_lang == "hi" else res["description_en"]

    audio_info = speech_synthesizer.synthesize(spoken_text, lang=target_lang, force=True)

    return jsonify({
        "status": "success",
        "result": res,
        "spoken_text": spoken_text,
        "audio_url": audio_info.get("audio_url"),
        "fallback_browser": audio_info.get("fallback_browser", False)
    })

@app.route("/api/medicine_detect", methods=["POST"])
def api_medicine_detect():
    """
    Dedicated Medicine Blister Strip & Expiry Date Reader Endpoint.
    Analyzes pill blister packs to extract name, strength, and expiry date.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    # Foil reflection suppression
    enhanced = medicine_reader.preprocess_foil(frame)
    # OCR extraction
    _, raw_text = ocr_translator.extract_text(enhanced if enhanced is not None else frame)
    res = medicine_reader.analyze_strip(frame, raw_text=raw_text)

    if res["detected"]:
        if target_lang == "hi":
            spoken_text = res["advisory_hi"]
        elif target_lang != "en":
            spoken_text = ocr_translator.translate_text(res["advisory_en"], target_language=target_lang)
        else:
            spoken_text = res["advisory_en"]

        severity = "danger" if res.get("is_expired") else "success"
        add_log("medicine", f"Strip Scanned: {res['full_name'] or 'Medicine'} (Exp: {res['expiry_str'] or 'Unknown'})", severity)
    else:
        spoken_text = res["advisory_hi"] if target_lang == "hi" else res["advisory_en"]

    audio_info = speech_synthesizer.synthesize(spoken_text, lang=target_lang, force=True)

    return jsonify({
        "status": "success",
        "result": res,
        "raw_text": raw_text,
        "spoken_text": spoken_text,
        "is_expired": res.get("is_expired"),
        "audio_url": audio_info.get("audio_url"),
        "fallback_browser": audio_info.get("fallback_browser", False)
    })

@app.route("/api/contacts", methods=["GET"])
def api_get_contacts():
    """Returns list of registered known contacts."""
    return jsonify({
        "status": "success",
        "contacts": face_engine.list_contacts()
    })

@app.route("/api/contacts/register", methods=["POST"])
def api_register_contact():
    """Registers a new person/face into the known face memory database."""
    data = request.get_json(silent=True) or {}
    name = data.get("name")
    role = data.get("role", "Friend / Family")
    image_b64 = data.get("image")

    if not name or not image_b64:
        return jsonify({"error": "Name and image required"}), 400

    frame = vision_detector.decode_base64_image(image_b64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    success, msg = face_engine.register_contact(name, role, frame)
    if success:
        add_log("social", f"New contact enrolled: {name} ({role})", "info")
        return jsonify({"status": "success", "message": msg})
    return jsonify({"error": msg}), 400

@app.route("/api/sos/trigger", methods=["POST"])
def api_sos_trigger():
    """Triggers high-priority caregiver emergency alert and generates WhatsApp dispatch link."""
    data = request.get_json(silent=True) or {}
    message = str(data.get("message") or "User triggered Emergency SOS Alert!")[:300]
    if data.get("latitude") is not None and data.get("longitude") is not None:
        location_tracker.update_location(data["latitude"], data["longitude"], data.get("accuracy", 10.0))
    location_tracker.trigger_sos(message)
    add_log("sos", f"EMERGENCY SOS: {message}", "danger")

    # Generate Google Maps & WhatsApp emergency dispatch links
    lat = location_tracker.current_lat
    lon = location_tracker.current_lon
    maps_url = f"https://maps.google.com/?q={lat},{lon}"
    sos_text = f"🚨 EMERGENCY ALERT: Assistive navigation user needs urgent assistance!\nLocation: {location_tracker.location_name}\nGPS: {lat}, {lon}\nLive Map: {maps_url}"
    import urllib.parse
    whatsapp_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(sos_text)}"

    return jsonify({
        "status": "sos_active",
        "message": message,
        "latitude": lat,
        "longitude": lon,
        "maps_url": maps_url,
        "whatsapp_url": whatsapp_url
    })

@app.route("/api/caregiver/export_report", methods=["GET"])
def api_caregiver_export_report():
    """Generates comprehensive daily mobility & safety analytics summary."""
    analytics = location_tracker.get_safety_analytics()
    analytics["recent_events"] = recent_detections_log[:20]
    analytics["project_title"] = config.PROJECT_TITLE
    analytics["college"] = config.COLLEGE_NAME
    analytics["generated_at"] = time.strftime("%Y-%m-%d %H:%M:%S")

    # If requested as JSON
    if request.args.get("format") == "json":
        return jsonify(analytics)

    # Return clean printable HTML report
    return render_template("mobility_report.html", report=analytics)

@app.route("/api/sos/resolve", methods=["POST"])
def api_sos_resolve():
    """Clears emergency SOS status."""
    location_tracker.resolve_sos()
    add_log("sos", "Emergency SOS resolved by caregiver.", "info")
    return jsonify({"status": "sos_resolved"})

@app.route("/api/tts", methods=["POST"])
def api_tts():
    """Direct Text-to-Speech API."""
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    lang = data.get("lang", config.DEFAULT_LANGUAGE)
    force = data.get("force", True)

    result = speech_synthesizer.synthesize(text, lang=lang, force=force)
    return jsonify(result)

@app.route("/api/languages", methods=["GET"])
def api_languages():
    """Returns supported multilingual dictionary."""
    return jsonify(config.SUPPORTED_LANGUAGES)


@app.route("/api/health", methods=["GET"])
def api_health():
    """Server status: loaded models, detection settings and live-session counters."""
    frames = session_stats["frames"]
    return jsonify({
        "status": "ok",
        "version": config.APP_VERSION,
        "uptime_s": int(time.time() - SERVER_STARTED_AT),
        "detector": {
            "model": vision_detector.model_name,
            "ready": vision_detector.model is not None,
            "benchmark_ms": vision_detector.infer_ms,
            "fast_model": vision_detector.fast_model_name,
            "extended_model": vision_detector.ext_model_name,
            "confidence": vision_detector.conf_thresh,
            "weak_confidence": vision_detector.low_conf,
            "low_light_enhance": bool(vision_detector.enhance_low_light),
            "quality_profiles": QUALITY_IMAGE_SIZES,
            "default_quality": config.DEFAULT_DETECTION_QUALITY,
        },
        "session": {
            "frames": frames,
            "objects": session_stats["objects"],
            "hazard_alerts": session_stats["hazard_alerts"],
            "avg_detect_ms": round(session_stats["detect_ms"] / frames, 1) if frames else None,
        },
        "languages": len(config.SUPPORTED_LANGUAGES),
    })


# ==========================================
# OUTDOOR WAYFINDING, SCENE Q&A & FALL SOS
# ==========================================

@app.route("/api/navigate/landmarks", methods=["GET"])
def api_navigate_landmarks():
    """Returns list of searchable pedestrian landmarks."""
    from modules.navigation_assistant import SKIT_WAYPOINTS
    landmarks = [
        {"id": k, "name": v["name"], "description": v["description"], "lat": v["lat"], "lon": v["lon"]}
        for k, v in SKIT_WAYPOINTS.items()
    ]
    return jsonify({"status": "success", "landmarks": landmarks})

def _number(value, low, high):
    """A float within [low, high] from request JSON, or None."""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if low <= v <= high and v == v else None


def _position(data, lat_key, lon_key):
    lat, lon = _number(data.get(lat_key), -90, 90), _number(data.get(lon_key), -180, 180)
    return (lat, lon) if lat is not None and lon is not None else None


@app.route("/api/navigate/plan", methods=["POST"])
def api_navigate_plan():
    """
    Plans a walking route from the user's GPS position to a destination (saved place, campus
    landmark or a place found on the map). heading: compass direction the user faces, if known.
    """
    data = request.get_json(silent=True) or {}
    dest_query = str(data.get("destination", "")).strip()
    origin = _position(data, "origin_lat", "origin_lon")
    heading = _number(data.get("heading"), 0, 360)
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not dest_query:
        return jsonify({"error": "Destination query is required"}), 400
    # Never guess where the user is: a route from the wrong start sends them the wrong way
    if origin is None:
        return jsonify({"status": "error", "error": "location_unknown",
                        "message": "I do not know where you are yet. Turn on location and try again."}), 400

    plan = navigation_assistant.plan_pedestrian_route(origin[0], origin[1], dest_query, heading=heading)
    if plan.get("status") == "success":
        instruction_en = plan["initial_instruction"]
        instruction_trans = ocr_translator.translate_text(instruction_en, target_language=target_lang) if target_lang != "en" else instruction_en
        plan["instruction_translated"] = instruction_trans
        add_log("navigation", f"Route started: {plan['destination']} ({int(plan['total_distance_m'])}m, {plan['mode']})", "info")
    else:
        msg = plan.get("message", "")
        plan["instruction_translated"] = ocr_translator.translate_text(msg, target_language=target_lang) if target_lang != "en" else msg
    return jsonify(plan)

@app.route("/api/navigate/update", methods=["POST"])
def api_navigate_update():
    """
    Guidance for the user's position along the active route. heading (compass degrees the user faces)
    and accuracy (GPS meters) are optional. Translation is skipped when the browser already spoke
    this instruction (announced_key).
    """
    data = request.get_json(silent=True) or {}
    position = _position(data, "lat", "lon")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if position is None:
        return jsonify({"error": "Latitude and longitude required"}), 400

    update = navigation_assistant.update_navigation_progress(
        position[0], position[1], heading=_number(data.get("heading"), 0, 360),
        accuracy=_number(data.get("accuracy"), 0, 5000))
    if not update:
        return jsonify({"status": "no_active_route"})

    fresh = update.get("announce_key") != data.get("announced_key")
    if target_lang != "en" and "instruction" in update and fresh:
        update["instruction_translated"] = ocr_translator.translate_text(update["instruction"], target_language=target_lang)
    else:
        update["instruction_translated"] = update.get("instruction", "")
    if update["status"] == "arrived":
        add_log("navigation", f"Arrived: {update['instruction']}", "info")

    return jsonify(update)

@app.route("/api/where_am_i", methods=["POST"])
def api_where_am_i():
    """Spoken answer to 'Where am I?' for the user's GPS position."""
    data = request.get_json(silent=True) or {}
    position = _position(data, "lat", "lon")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)
    if position is None:
        return jsonify({"status": "error", "error": "location_unknown",
                        "answer_en": "I cannot find your location. Turn on location for this site and try again."}), 400
    answer_en = navigation_assistant.describe_position(position[0], position[1],
                                                       heading=_number(data.get("heading"), 0, 360),
                                                       accuracy=_number(data.get("accuracy"), 0, 5000))
    answer_trans = ocr_translator.translate_text(answer_en, target_language=target_lang) if target_lang != "en" else answer_en
    return jsonify({"status": "success", "answer_en": answer_en, "answer_translated": answer_trans})

@app.route("/api/places", methods=["GET", "POST", "DELETE"])
def api_places():
    """
    The user's saved places ("location setter"): GET lists them, POST {name, lat, lon} saves the
    current position under a name, DELETE {name} removes one.
    """
    places = navigation_assistant.places
    if request.method == "GET":
        return jsonify({"status": "success", "places": places.all()})
    data = request.get_json(silent=True) or {}
    name = str(data.get("name", "")).strip()
    if not name:
        return jsonify({"status": "error", "error": "A place needs a name."}), 400
    if request.method == "DELETE":
        if not places.remove(name):
            return jsonify({"status": "error", "error": f"No saved place called {name}."}), 404
        add_log("navigation", f"Saved place removed: {name}", "info")
        return jsonify({"status": "success", "places": places.all()})
    position = _position(data, "lat", "lon")
    if position is None:
        return jsonify({"status": "error", "error": "location_unknown",
                        "message": "I do not know where you are yet. Turn on location and try again."}), 400
    place = places.save(name, *position)
    add_log("navigation", f"Place saved: {place['name']}", "info")
    return jsonify({"status": "success", "place": place, "places": places.all(),
                    "message": f"Saved this place as {place['name']}."})

@app.route("/api/navigate/cancel", methods=["POST"])
def api_navigate_cancel():
    """Cancels active navigation route."""
    res = navigation_assistant.cancel_navigation()
    add_log("navigation", "Pedestrian navigation cancelled.", "info")
    return jsonify(res)

@app.route("/api/scene_qa", methods=["POST"])
def api_scene_qa():
    """
    Conversational Scene Q&A ("Ask AMCS").
    Answers natural language queries about the current camera view.
    """
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()[:300]
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)
    # The browser can speak most languages itself; then no server audio is made (tts: false)
    want_server_tts = data.get("tts", True) is not False

    if not question:
        return jsonify({"error": "Question is required"}), 400

    # Hindi words the engine knows are understood directly; anything else not in English is
    # translated first ("ಮುಂದೆ ಏನಿದೆ" -> "what is ahead")
    question_en = question
    if re.search(r"[^\x00-\x7f]", normalize_question(question)):
        question_en = ocr_translator.translate_text(question, target_language="en") or question

    detections = []
    ocr_text = ""
    ground_hazards = []
    brightness = None

    if image_base64:
        frame = vision_detector.decode_base64_image(image_base64)
        if frame is not None:
            brightness = float(frame.mean())
            facing = data.get("camera_facing") if data.get("camera_facing") in ("user", "environment") else None
            _, detections, _ = vision_detector.detect(frame, extended=True, camera_facing=facing,
                                                      quality=detection_quality(data.get("quality")))
            # Faces are matched against saved contacts only for "who is in front of me"
            if scene_qa_engine.needs_faces(question_en):
                for d in detections:
                    if d["class"] == "person" and not d.get("is_user"):
                        info = run_stage("qa_faces", lambda d=d: face_engine.identify_person(frame, d["bbox"]), None)
                        if info:
                            d["face_info"] = info
            # Reading text is slow, so it only runs when the question is about text or the whole scene
            if scene_qa_engine.needs_text(question_en):
                _, ocr_text = run_stage("qa_ocr", lambda: ocr_translator.extract_text(frame), ([], ""))
            _, ground_hazards = run_stage(
                "qa_ground", lambda: ground_hazard_detector.analyze_ground_plane(
                    frame, object_boxes=[d["bbox"] for d in detections]), (None, []))

    answer_en = scene_qa_engine.answer_question(
        question_en,
        detections=detections,
        ocr_text=ocr_text,
        ground_hazards=ground_hazards,
        lang=target_lang,
        brightness=brightness
    )

    answer_trans = ocr_translator.translate_text(answer_en, target_language=target_lang) if target_lang != "en" else answer_en

    # Server audio for the answer, made in the background so the answer text returns at once
    audio_url = speech_synthesizer.synthesize_later(answer_trans, lang=target_lang) if want_server_tts else None

    add_log("qa", f"Q: '{question}' -> A: '{answer_en}'", "info")

    return jsonify({
        "status": "success",
        "question": question,
        "question_en": question_en,
        "answer_en": answer_en,
        "answer_translated": answer_trans,
        "target_lang": target_lang,
        "objects_seen": len([d for d in detections if not d.get("is_user")]),
        "audio_url": audio_url,
        "fallback_browser": audio_url is None
    })

@app.route("/api/fall_detected", methods=["POST"])
def api_fall_detected():
    """
    Receives automated Fall Detection alert from device accelerometer.
    Dispatches immediate distress alert to Caregiver.
    """
    data = request.get_json(silent=True) or {}
    lat = data.get("latitude", 26.82285)
    lon = data.get("longitude", 75.86475)
    battery = data.get("battery", "unknown")

    sos_msg = f"EMERGENCY: User fall detected at SKIT Campus! Inactivity confirmation timed out. Battery: {battery}%."
    location_tracker.update_location(lat, lon)
    location_tracker.trigger_sos(sos_msg)
    add_log("sos", sos_msg, "danger")

    import urllib.parse
    maps_url = f"https://www.google.com/maps?q={lat},{lon}"
    whatsapp_text = f"🚨 URGENT: Automatic Multilanguage Communication System detected a user fall! Location: {maps_url}"
    whatsapp_url = f"https://api.whatsapp.com/send?text={urllib.parse.quote(whatsapp_text)}"

    return jsonify({
        "status": "fall_alert_dispatched",
        "message": sos_msg,
        "latitude": lat,
        "longitude": lon,
        "maps_url": maps_url,
        "whatsapp_url": whatsapp_url
    })


@app.route("/api/find_item", methods=["POST"])
def api_find_item():
    """
    Item Finder & Acoustic Audio Radar endpoint ("Find My Item").
    Guides visually impaired user to misplaced personal belongings with acoustic Geiger feedback.
    """
    data = request.get_json(silent=True) or {}
    target_query = data.get("target", "").strip()
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not target_query:
        return jsonify({"error": "Target item query required"}), 400

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    # Run object detections
    facing = data.get("camera_facing") if data.get("camera_facing") in ("user", "environment") else None
    _, detections, _ = vision_detector.detect(frame, extended=True, camera_facing=facing,
                                              quality=detection_quality(data.get("quality")))

    # Search for item with acoustic radar engine
    radar_res = item_finder_engine.search_frame_for_item(frame, detections, target_query)

    guidance_en = radar_res.get("guidance", "")
    if target_lang != "en" and guidance_en:
        guidance_trans = ocr_translator.translate_text(guidance_en, target_language=target_lang)
    else:
        guidance_trans = guidance_en
    radar_res["guidance_translated"] = guidance_trans

    if radar_res.get("found"):
        add_log("finder", f"Located {radar_res['target']} (~{radar_res.get('distance_m')}m) - Radar Locked", "success")

    return jsonify({
        "status": "success",
        "radar": radar_res
    })


@app.route("/api/scan_door", methods=["POST"])
def api_scan_door():
    """Dedicated Indoor Door, Handle & Room Sign detection endpoint."""
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    ocr_items, _ = ocr_translator.extract_text(frame)
    annotated, doors = indoor_door_locator.analyze_frame_for_doors(frame, ocr_items=ocr_items)

    if not doors:
        msg_en = "No doors or doorways detected in front of you."
        msg_trans = ocr_translator.translate_text(msg_en, target_language=target_lang) if target_lang != "en" else msg_en
        return jsonify({
            "status": "success",
            "doors": [],
            "guidance_en": msg_en,
            "guidance_translated": msg_trans
        })

    primary_door = doors[0]
    guidance_en = primary_door["guidance"]
    guidance_trans = ocr_translator.translate_text(guidance_en, target_language=target_lang) if target_lang != "en" else guidance_en

    add_log("indoor", f"Detected Door: {primary_door['state'].upper()} (Handle: {primary_door['handle_side']})", "info")

    return jsonify({
        "status": "success",
        "doors": doors,
        "primary_door": primary_door,
        "guidance_en": guidance_en,
        "guidance_translated": guidance_trans
    })


@app.route("/api/audio_hazard_detect", methods=["POST"])
def api_audio_hazard_detect():
    """
    Acoustic Siren & Car Horn AI Detector endpoint (Multimodal Sensor Fusion).
    Monitors audio frequency spectrum for oncoming horns, sirens, and hazardous sounds.
    """
    data = request.get_json(silent=True) or {}
    frequency_bins = data.get("frequency_bins", [])
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    result = acoustic_hazard_detector.analyze_audio_spectrum(frequency_bins)

    if result["detected"]:
        advisory_en = result["advisory_en"]
        if target_lang != "en":
            advisory_trans = ocr_translator.translate_text(advisory_en, target_language=target_lang)
        else:
            advisory_trans = advisory_en
        result["advisory_translated"] = advisory_trans

        # High priority caregiver alert
        add_log("acoustic", f"Acoustic Alert: {result['hazard_type'].upper()} ({int(result['confidence']*100)}% conf)", "danger")

    return jsonify({
        "status": "success",
        "result": result
    })


@app.route("/api/match_clothing", methods=["POST"])
def api_match_clothing():
    """
    Color & Clothing Wardrobe Matcher endpoint.
    Identifies garment shade, pattern, and styling pairing recommendations.
    """
    data = request.get_json(silent=True) or {}
    image_base64 = data.get("image")
    target_lang = data.get("lang", config.DEFAULT_LANGUAGE)

    if not image_base64:
        return jsonify({"error": "No image data provided"}), 400

    frame = vision_detector.decode_base64_image(image_base64)
    if frame is None:
        return jsonify({"error": "Invalid image payload"}), 400

    analysis = color_clothing_matcher.analyze_clothing(frame)
    if not analysis:
        return jsonify({"error": "Failed to analyze garment image"}), 400

    summary_en = analysis["summary_en"]
    if target_lang != "en":
        summary_trans = ocr_translator.translate_text(summary_en, target_language=target_lang)
    else:
        summary_trans = summary_en
    analysis["summary_translated"] = summary_trans

    add_log("wardrobe", f"Garment Identified: {analysis['primary_color']} ({analysis['pattern']})", "info")

    return jsonify({
        "status": "success",
        "analysis": analysis
    })


# ==========================================
# CAREGIVER "SEE WHAT I SEE" TELE-ASSISTANCE
# ==========================================

@app.route("/api/teleassist/call", methods=["POST"])
def api_teleassist_call():
    """Starts or terminates tele-assistance call."""
    data = request.get_json(silent=True) or {}
    action = data.get("action", "start")
    if action == "start":
        res = teleassist_engine.start_call()
        add_log("teleassist", "Remote Tele-Assistance Call initiated.", "danger")
    else:
        res = teleassist_engine.end_call()
        add_log("teleassist", "Remote Tele-Assistance Call ended.", "info")
    return jsonify(res)

@app.route("/api/teleassist/frame", methods=["POST", "GET"])
def api_teleassist_frame():
    """User device uploads frame; Caregiver dashboard streams frame."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        image = data.get("image")
        # Privacy: frames are only kept while the caregiver has a call open
        if not teleassist_engine.call_active:
            return jsonify({"status": "frame_ignored", "call_active": False})
        if isinstance(image, str) and image.startswith("data:image"):
            teleassist_engine.update_frame(image)
            return jsonify({"status": "frame_received", "call_active": True})
        return jsonify({"status": "error", "error": "Expected a data:image URL"}), 400
    else:
        return jsonify(teleassist_engine.get_latest_frame())

@app.route("/api/teleassist/intercom", methods=["POST", "GET"])
def api_teleassist_intercom():
    """Caregiver sends spoken voice advice; user device polls for advice."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        msg = str(data.get("message") or "").strip()[:200]
        if not msg:
            return jsonify({"status": "error", "error": "Message is empty"}), 400
        teleassist_engine.post_caregiver_guidance(msg)
        add_log("teleassist", f"Caregiver Intercom: '{msg}'", "info")
        return jsonify({"status": "guidance_sent"})
    else:
        msgs = teleassist_engine.fetch_pending_guidance()
        return jsonify({"status": "success", "messages": msgs, "call_active": teleassist_engine.call_active})

@app.route("/api/teleassist/signal", methods=["POST", "GET"])
def api_teleassist_signal():
    """WebRTC peer-to-peer signaling exchange."""
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        sig_type = data.get("type")
        payload = data.get("payload")
        teleassist_engine.set_signal(sig_type, payload)
        return jsonify({"status": "signal_recorded"})
    else:
        return jsonify(teleassist_engine.get_signals())


# ==========================================
# EXAMINER VIVA EVALUATION SANDBOX & BENCHMARKS
# ==========================================

@app.route("/api/sandbox/benchmark", methods=["GET", "POST"])
def api_sandbox_benchmark():
    """Runs system latency, throughput, and accuracy benchmark suite."""
    return jsonify({
        "status": "success",
        "benchmark": viva_sandbox.run_benchmark_suite()
    })

@app.route("/api/sandbox/simulate", methods=["POST"])
def api_sandbox_simulate():
    """Executes an academic simulation scenario and returns verification report."""
    data = request.get_json(silent=True) or {}
    scenario_id = data.get("scenario_id", "speeding_vehicle_ttc")
    res = viva_sandbox.execute_scenario(scenario_id)
    if res.get("status") == "passed":
        add_log("benchmark", f"Viva Test Passed: {res['title']} ({res['execution_latency_ms']}ms)", "success")
    return jsonify(res)


# ==========================================
# APPLICATION ENTRYPOINT
# ==========================================

def dev_certificate():
    """
    Self-signed certificate for HTTPS on the local network, kept in .certs/ and reused, so a phone
    that accepted it once is not asked again after every restart. Returns (cert_file, key_file).
    """
    from werkzeug.serving import make_ssl_devcert     # uses the 'cryptography' package
    base = config.BASE_DIR / ".certs" / "amcs-dev"
    cert_file, key_file = base.with_suffix(".crt"), base.with_suffix(".key")
    if not (cert_file.exists() and key_file.exists()):
        base.parent.mkdir(exist_ok=True)
        make_ssl_devcert(str(base), host="localhost")
    return str(cert_file), str(key_file)


if __name__ == "__main__":
    print(f"\n==================================================================")
    print(f"  {config.PROJECT_TITLE}")
    print(f"  {config.COLLEGE_NAME}")
    print(f"  {config.DEPARTMENT}")
    print(f"  Presented By: Amit Tiwari, Deepak Choudhary, Deependra Singh")
    print(f"  Supervised By: Vikas, Assistant Professor-2")
    print(f"==================================================================")
    ssl_context = None
    if config.USE_HTTPS:
        # Phones only grant camera/mic access over HTTPS (localhost is exempt)
        try:
            ssl_context = dev_certificate()
        except ImportError:
            print("  [!] AMCS_HTTPS=1 needs the 'cryptography' package (pip install cryptography). Falling back to HTTP.")
    # Flask's built-in server stalls for 1.5-2 s on every few HTTPS requests (measured), which a
    # phone feels as lag. uvicorn serves HTTPS without that when it is installed.
    try:
        import uvicorn
    except ImportError:
        uvicorn = None
    use_uvicorn = bool(ssl_context) and uvicorn is not None and not config.DEBUG_MODE
    scheme = "https" if ssl_context else "http"
    print(f"  Server starting at: {scheme}://localhost:{config.SERVER_PORT}")
    print(f"  Caregiver Dashboard: {scheme}://localhost:{config.SERVER_PORT}/caregiver")
    print(f"  OCR Translation: {scheme}://localhost:{config.SERVER_PORT}/ocr")
    if not ssl_context:
        print(f"  Tip: to use a phone's camera over Wi-Fi, run with AMCS_HTTPS=1")
    elif not use_uvicorn:
        print(f"  Tip: pip install uvicorn for faster HTTPS (the built-in server stalls now and then)")
    print(f"==================================================================\n")
    if use_uvicorn:
        uvicorn.run(app, host=config.SERVER_HOST, port=config.SERVER_PORT, interface="wsgi",
                    ssl_certfile=ssl_context[0], ssl_keyfile=ssl_context[1], log_level="warning")
    else:
        app.run(host=config.SERVER_HOST, port=config.SERVER_PORT, debug=config.DEBUG_MODE, ssl_context=ssl_context)
