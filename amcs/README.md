# Automatic Multilanguage Communication System

**Department of Computer Science & Engineering**  
**Swami Keshvanand Institute of Technology, Management & Gramothan (SKIT), Jaipur**  
*Final Year Project Proposal & Implementation*

---

### Project Team & Supervision
- **Presented By:**
  - **Amit Tiwari** &mdash; Roll No: `23ESKCX010`
  - **Deepak Choudhary** &mdash; Roll No: `23ESKCX026`
  - **Deependra Singh** &mdash; Roll No: `23ESKCX027`
- **Supervised By:**
  - **Vikas**, Assistant Professor-2, Dept. of CSE, SKIT Jaipur

---

## 📖 Project Overview

**Automatic Multilanguage Communication System** is an assistive vision-to-speech and real-time mapping platform designed to empower visually impaired individuals with independent mobility, health safety, and social awareness while keeping caregivers informed.

### Core Modules & Capabilities:
1. **Computer Vision & Obstacle Detection (OpenCV & YOLO26):** Detects obstacles, humans, vehicles, furniture, and path obstructions in real time, calculating spatial positioning (Left, Center, Right) and proximity distance in meters.
2. **Autonomous Vehicle-Grade Time-to-Collision (TTC) & Dynamic Trajectory Tracking:** Adapts ADAS (Tesla/Waymo) collision mechanics to calculate relative velocity ($\Delta d / \Delta t$) and TTC. Overrides standard speech with rapid evasive commands (*"Warning! Approaching bicycle closing in at 2.1s! Step right immediately!"*).
3. **"Who Is In Front of Me?" Known Face & Social Intelligence Engine:** Identifies registered family, friends, and faculty (*"Vikas Sir"*, *"Amit"*, *"Deepak"*, *"Deependra"*) along with social expressions (smiling, neutral) to foster natural social engagement.
4. **Traffic Signal & Pedestrian Crossing Assistant:** Distinguishes active traffic light states (Red, Yellow, Green) to advise the visually impaired user when it is safe to cross or when to wait at the curb.
5. **Medicine Blister Strip & Expiry Date Reader:** Uses CLAHE metallic glare reduction, OCR, and medical entity extraction to read medicine names (e.g. *Dolo 650*, *Azithromycin 500mg*, *Pan 40*) and verify expiry dates (*"DANGER! This medicine expired in March 2024!"* vs *"Safe to use. Valid until August 2028."*).
6. **Indian Banknote Recognition (Mahatma Gandhi New Series):** Identifies ₹10, ₹20, ₹50, ₹100, ₹200, and ₹500 notes using HSV color profiling and OCR tokens.
7. **Signboard & Text OCR (OpenCV & EasyOCR):** Extracts text from public signs, boards, labels, and notices.
8. **Multilingual Translation Engine (NMT):** Translates descriptions and extracted text across 14+ languages (English, Hindi, Spanish, French, German, Chinese, Japanese, Arabic, Russian, Bengali, Tamil, Telugu, Marathi, Gujarati) in real time.
9. **Automated Voice Feedback & 3D Spatial Audio:** gTTS and Web Speech synthesis with persistent caching, 3D directional stereo panning (`StereoPannerNode`), and intelligent 7.5s debouncing.
10. **Live GPS Tracking & Caregiver Safety Map:** Real-time coordinates, breadcrumb history, 500m geofencing perimeter, and interactive Folium Leaflet maps for caregivers with automated WhatsApp emergency SOS.
11. **Ground Drop-Off, Curb & Pothole Hazard Detection:** Edge-gradient and vanishing plane analysis detects downward stairs, curbs, potholes, and sudden ground depressions with rapid warning tones.
12. **Voice-Guided Turn-by-Turn Pedestrian Wayfinding:** A-to-B routing with bearing and distance alerts (*"Head North along pedestrian walkway for 35 meters"*), blending seamlessly with obstacle perception.
13. **Fall Detection & Automated Caregiver SOS:** Accelerometer-driven impact detection with a 10-second cancelable voice countdown and automated emergency GPS dispatch.
14. **Auto-Torch Low-Light Night Mode:** Ambient luminance monitoring automatically activates smartphone flashlight in dark outdoor conditions.
15. **Conversational Scene Q&A ("Ask AMCS"):** Double-tap or voice query to ask natural questions (*"What is in front of me?"*, *"Is there a place to sit?"*, *"What does the sign say?"*).
16. **Hands-Free Voice Commands:** Speech recognition for seamless hands-free control (switch modes, start/stop, check location, trigger SOS, navigate, ask questions).
17. **"Find My Item" Audio Radar (Spatial Geiger Counter):** Guides visually impaired users to lost belongings (phone, keys, bottle, backpack, spectacles, book) using real-time spatial panning and dynamic Geiger counter audio pings that accelerate (1.5 Hz to 9.0 Hz) and pitch up as the camera aligns.
18. **Indoor Door, Handle & Room Sign Locator:** Scans indoor hallways and classrooms to identify doors, open doorways, door handle/knob orientation (`left` vs `right`), distance, and room sign text tokens.
19. **Acoustic Siren & Car Horn Detector (Multimodal Sensor Fusion):** Real-time Web Audio API background microphone spectrum analyzer (256-point FFT) that detects automotive horn blasts (2800–4500 Hz) and emergency sirens (650–1600 Hz warble) with haptic vibration alerts.
20. **Color & Clothing Wardrobe Matcher:** Recognizes 20 distinct garment colors and clothing patterns (solid, striped, plaid/checkered) with smart fashion pairing advice for daily independence.
21. **Caregiver "See What I See" Live Tele-Assistance:** Live video viewport and WebRTC peer signaling allowing remote caregivers to inspect the user's perspective in real time and deliver live intercom voice guidance.
22. **Examiner Viva Evaluation Sandbox & Benchmark Dashboard:** Dedicated academic evaluation console (`/sandbox`) featuring 5 interactive simulation test benches (TTC collision, expired medicine foil, ₹500 banknote, indoor door/knob, acoustic siren) with sub-40ms vision telemetry and one-click printable Viva reports.

---

## ✨ What is new in v3.2 (walking directions, voice questions, phone accuracy)

| Change | Effect |
| :--- | :--- |
| **Saved places ("location setter")** - say *"save this place as home"* (or tap **Save this place**); the places are listed and can be deleted in Settings | *"Take me to home"* works for any spot the user saved, not just campus buildings |
| **Real walking routes** - destinations are saved places, campus buildings, or any address or landmark nearby (OpenStreetMap search); routes follow footpaths and streets (OpenStreetMap foot routing) | Spoken turns at the right corners: *"Continue straight for 60 meters, then turn right"*, *"In 30 meters, turn left onto MI Road"*, *"Now, turn right"*. Without internet the user is guided straight towards the place by clock direction |
| **Turns told from the way the user faces** - the phone compass (camera held upright) or the GPS direction of travel | *"Turn around to face your route"*, *"Turn to your left"*; the arrow on the route banner points the way to walk; vibration on every turn (two pulses right, one long pulse left) |
| **Leaving the route is noticed** - more than 30 m off the route on two readings plans a new route from where the user is | *"You have left the route. New route: ..."* |
| **No guessed positions** - routing and saving need a real GPS fix; the campus coordinates are never used as a stand-in | A wrong start point can no longer send the user the wrong way |
| **"Where am I?"** - street and area from the map, plus the nearest saved place or campus building | *"You are on MI Road, in C-Scheme. Home is 100 meters away, at 3 o'clock."* |
| **Ask by voice** - double-tap the camera view (or the microphone button) and ask; any question goes to the scene Q&A; live guidance pauses while the user asks and hears the answer | *"What is in front of me?"* answers what is ahead first, with distances, then each side briefly |
| **New questions** - *"What is this?"* / *"What am I holding?"* (with any text on it), *"Who is in front of me?"* (saved contacts by name), Hindi in Devanagari and Hinglish (*"saamne kya hai"*, *"मेरे सामने क्या है"*), other languages translated first | Questions it does not understand say so, then describe what is ahead, instead of a misleading guess |
| **Correct distances in portrait** - the camera's field of view is applied to the long side of the picture | Distances on a phone held upright were about 25-30% too short |
| **Phone detection profile** - the light model at 640 px instead of 512 px | On 500 COCO photos: F1 0.603 -> 0.640, small objects found 21% -> 29%, about 100 ms per frame |

---

## ✨ What is new in v3.1 (real-time guidance on phones)

Measured on this project's laptop CPU, phone profile (512 px frames):

| Change | Effect |
| :--- | :--- |
| **Fewer false ground alarms** - steps and curbs must be long horizontal edges on otherwise plain ground, each a change of surface; potholes must be compact dark patches on plain, lit ground | False "stairs / curb / pothole" warnings on 500 everyday COCO photos: **30% -> 1.4%**. Steps are still found with camera noise, blur, low contrast and a person on them; grass, tiles and painted lines stay silent |
| **Phones get answers faster** - the lighter detector for phones, smaller uploads, and the next frame is sent as soon as the previous answer arrives (it waited for a fixed 600 ms timer) | Server time per phone frame about 70-100 ms; up to 4 frames a second |
| **Translation and server speech only when something will be spoken** - other frames reuse the last translation | Non-English live loop: about 1.9 s -> about 0.1 s for frames that are not spoken |
| **No stale objects after turning** - cached extended-model results are dropped when the camera sees a new view | Objects from the previous view are no longer announced |
| **Stricter extended classes** - coin, helmet, towel, stool, trash can, mirror, window need more confidence | Removes the stone-as-coin and turban-as-helmet mistakes seen on the test photos |
| **Clearer speech** - "5 cars around you", "Move to your right to pass" / "Stop, the way ahead is blocked" only from what is actually beside the user, and the distance of the nearest thing when it is close | Shorter, more useful announcements |
| **Lite mode on phones** - no background blur or glow, canvases redrawn less often | Smoother page on mid-range phones |

---

## ✨ What is new in v3.0

**Detection accuracy** (measured on 500 COCO val2017 images, boxes scored at IoU 0.5, script-level numbers below):

| Change | Effect |
| :--- | :--- |
| **Automatic model choice** - `yolo26m.pt` is used when the machine runs it within 330 ms per frame, otherwise `yolo26s.pt` | F1 0.636 -> 0.663, recall 52.0% -> 56.0% at the same precision |
| **Two-threshold tracking** - a faint detection (20-35%) is kept only when it continues an object seen clearly in the last 3 s | On blurred, darker follow-up frames: recall 37.7% -> 41.1%, F1 0.521 -> 0.546 |
| **Cross-class duplicate removal** - one box reported as two classes (sofa/bed) keeps the confident one | Fewer double announcements, precision +0.3 points |
| **Smarter fragment merging** - two similar-sized people side by side are no longer merged | Small recall gain in groups |
| **Distance smoothing** - a stationary object reports the median of its last sightings | Severity no longer flips between caution and hazard |
| **Detection quality setting** - Fast 512 px / Balanced 640 px / High detail 960 px | High detail raises small-object recall (34% -> 41% with `yolo26m`) at about 2x the time |

Software brightening of dark frames (gamma + CLAHE) was also benchmarked. It **lowered** accuracy on darkened
images, so it ships switched off (`LOW_LIGHT_ENHANCE` in `config.py`); the app shows a "Low light" chip and uses
the torch instead.

**Features**
- **Proximity radar** - a live top-down map of every detected object by direction and distance, with hazard and caution zones.
- **Objects in view** - nearest-first list with side, distance and confidence for each object.
- **Live telemetry** - object count, nearest obstacle, response time and the active vision model under the camera.
- **Recent guidance** - a log of the last 20 spoken messages, with warnings highlighted.
- **Smarter Scene Q&A** - *"How many people are there?"*, *"Where is the door?"*, *"What is nearest?"*, *"What is on my left?"*, and a measured answer to *"Is it dark?"*.
- **`/api/health`** - loaded models, benchmark time, detection settings and session counters.

**Interface** - new design system (indigo-to-cyan brand, layered dark surfaces, refined light theme), floating
header and mobile tab bar, camera viewfinder with corner brackets and a scan sweep while the assistant runs,
and restyled cards, buttons, dialogs and toasts across every page. High-contrast mode is kept.

---

## 🏗️ System Architecture

```text
       +-----------------------------------------------------+
       |      User Smartphone / Camera Mount / Microphone    |
       +-----------------------------------------------------+
                                  |
                                  v
              +---------------------------------------+
              |       Flask Application Server        |
              +---------------------------------------+
                |        |        |        |        |
    +-----------+        |        |        |        +-----------+
    |                    |        |        |                    |
    v                    v        v        v                    v
+--------------+ +-----------+ +------+ +------+ +---------------------+
| YOLO26 Vision| | Trajectory| | Face | |Traffic| | Medicine & Banknote|
| & Obstacles  | |  & TTC    | |Social| |Signals| | OCR & Glare Reader  |
+--------------+ +-----------+ +------+ +------+ +---------------------+
    |                    |        |        |                    |
    +--------------------+--------+--------+--------------------+
                                  |
                                  v
        +----------------------------------------------------+
        |   NLP Scene Synthesizer & Multilingual Translation |
        |   - 14+ Languages (English, Hindi, Spanish, etc.)  |
        |   - Natural Navigational Phrasing & Warnings       |
        +----------------------------------------------------+
                                  |
                                  v
        +----------------------------------------------------+
        |          Multimodal Output & Accessibility         |
        | - Automated Non-Repetitive Speech Synthesis (gTTS) |
        | - 3D Directional Spatial Panning & Haptic Vibrate  |
        | - 60 FPS Transparent Canvas HUD Bounding Boxes     |
        | - Hands-Free Voice Control ("Medicine", "Start")   |
        | - Caregiver Folium GPS Map & WhatsApp SOS Dispatch |
        +----------------------------------------------------+
```

---

## 📁 Project Directory Structure

```text
Automatic Communication language translation/
│
├── app.py                             # Flask application server & REST API routes
├── config.py                          # System configuration, languages, campus GPS
├── requirements.txt                   # Project dependencies
├── README.md                          # Comprehensive documentation
├── yolo26m.pt                         # YOLO26-Medium detector weights (used when the machine is fast enough)
├── yolo26s.pt                         # YOLO26-Small detector weights (automatic fallback; yolov8n.pt as last resort)
│
├── modules/
│   ├── __init__.py
│   ├── vision_detector.py             # YOLO26 obstacle detection, fragment merging & proximity estimation
│   ├── nlp_describer.py               # Natural Language Scene Description generator
│   ├── ocr_translator.py              # EasyOCR text detection & 14-language translation
│   ├── speech_synthesizer.py          # gTTS & Web Speech synthesis with MD5 caching
│   ├── location_tracker.py            # GPS tracking, Folium map, geofence, safety analytics
│   ├── currency_recognizer.py         # Indian banknote recognition (₹10 - ₹500)
│   ├── trajectory_tracker.py          # ADAS Time-to-Collision (TTC) & trajectory tracking
│   ├── face_social_engine.py          # Known face identification & social emotion cues
│   ├── traffic_crossing_assistant.py  # Traffic signal (Red/Green/Yellow) assistant
│   ├── medicine_reader.py             # Foil glare reduction, pill name & expiry reader
│   ├── ground_hazard_detector.py      # Elevation drop-offs, curbs, potholes & stair detection
│   ├── navigation_assistant.py        # Saved places, walking routes, heading-relative turn-by-turn guidance
│   ├── scene_qa_engine.py             # Conversational Scene Q&A ("Ask AMCS")
│   ├── item_finder_engine.py          # "Find My Item" audio Geiger radar engine
│   ├── indoor_door_locator.py         # Door, handle/knob orientation & room sign detector
│   ├── acoustic_hazard_detector.py    # Siren & car horn audio AI spectrum analyzer
│   ├── color_clothing_matcher.py      # Wardrobe color & fabric pattern matching engine
│   ├── teleassist_engine.py           # Caregiver live tele-assistance & intercom engine
│   └── viva_evaluation_sandbox.py     # Examiner viva simulation benches & latency telemetry
│
├── static/
│   ├── css/
│   │   └── style.css                  # Design system: tokens, light/dark/high-contrast themes, all page styles
│   ├── js/
│   │   ├── ui.js                      # Shared UI: theme, preferences, toasts, accessible dialogs, SOS banner
│   │   ├── panels.js                  # Assist live panels: telemetry, proximity radar, objects in view, history
│   │   ├── main.js                    # Camera feed, HUD, speech, voice commands, radar, SOS, shortcuts
│   │   └── caregiver.js               # Caregiver live telemetry, GPS & tele-assistance
│   ├── icons/                         # App icons (SVG + 192/512 PNG) for the PWA manifest
│   ├── audio_cache/                   # MD5-hashed speech audio (auto-pruned to 400 files)
│   ├── manifest.json                  # Progressive Web App (PWA) manifest
│   └── sw.js                          # Service worker (served at /sw.js, network-first)
│
├── templates/
│   ├── base.html                      # App shell: header, mobile tab bar, SOS banner, toasts, dialog
│   ├── index.html                     # Assistive interface (Obstacles, Signs, Currency, Radar)
│   ├── caregiver.html                 # Caregiver Live GPS & Tele-Assistance Dashboard
│   ├── text_translator.html           # Signboard OCR & Real-time Multilingual Translator
│   ├── sandbox.html                   # Examiner Viva Evaluation Sandbox & Benchmarks
│   ├── settings.html                  # Per-device voice, feedback, sensor & theme preferences
│   ├── about.html                     # Team attribution, SKIT Jaipur, Supervisor details
│   └── mobility_report.html           # Printable daily mobility & safety analytics report
│
└── tests/
    ├── test_modules.py                # Unit tests for core AI vision and NLP modules
    ├── test_app.py                    # Integration tests for all Flask API endpoints
    ├── test_new_features.py           # Unit tests for currency and safety analytics
    ├── test_trajectory.py             # Unit tests for TTC and evasive maneuvers
    ├── test_face_social.py            # Unit tests for face memory and social recognition
    ├── test_traffic_crossing.py       # Unit tests for pedestrian traffic signals
    ├── test_medicine_reader.py        # Unit tests for medicine identification & expiry logic
    ├── test_outdoor_suite.py          # Unit tests for ground hazard, navigation & Q&A APIs
    ├── test_item_finder.py            # Unit tests for Item Finder Audio Radar
    ├── test_door_locator.py           # Unit tests for Door, Handle & Sign Locator
    ├── test_acoustic_hazard.py        # Unit tests for Acoustic Horn & Siren Detector
    ├── test_clothing_matcher.py       # Unit tests for Color & Clothing Matcher
    ├── test_teleassist.py             # Unit tests for Caregiver Tele-Assistance
    ├── test_viva_sandbox.py           # Unit tests for Viva Sandbox & Benchmarks
    ├── test_ui_pages.py               # Page rendering, app shell & API hardening tests
    ├── test_detection_quality.py      # Two-threshold tracking, duplicates, auto model choice, Q&A, /api/health
    ├── test_environment_guidance.py   # Ground alarms, stale objects, speak-only translation, spoken wording
    └── test_navigation_and_questions.py # Saved places, turn-by-turn, where am I, voice questions, portrait distance
```

---

## 🚀 Quick Setup & Installation

### 1. Prerequisites
- Python 3.10 to 3.13
- Webcam / Smartphone Camera
- Web Browser (Google Chrome, Microsoft Edge, or Firefox)

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Unit and Integration Tests (205 Tests)
```bash
# Run the complete test suite
python -m unittest discover tests

# Or run individual feature test suites:
python -m unittest tests/test_modules.py            # Core Vision & NLP
python -m unittest tests/test_app.py                # REST API Endpoints & Page Routes
python -m unittest tests/test_trajectory.py         # ADAS Time-to-Collision (TTC)
python -m unittest tests/test_face_social.py        # Known Face & Social Interaction
python -m unittest tests/test_traffic_crossing.py   # Pedestrian Traffic Signals (Red/Green/Yellow)
python -m unittest tests/test_medicine_reader.py    # Medicine Blister Strip & Expiry
python -m unittest tests/test_outdoor_suite.py       # Ground Hazards, Wayfinding & Q&A
python -m unittest tests/test_item_finder.py        # "Find My Item" Audio Radar
python -m unittest tests/test_door_locator.py       # Indoor Door & Handle Locator
python -m unittest tests/test_acoustic_hazard.py    # Acoustic Horn & Siren Detector
python -m unittest tests/test_clothing_matcher.py   # Color & Clothing Matcher
python -m unittest tests/test_teleassist.py         # Caregiver Live Tele-Assistance
python -m unittest tests/test_viva_sandbox.py       # Examiner Viva Sandbox & Benchmarks
python -m unittest tests/test_ui_pages.py           # Pages, app shell & API hardening
```

### 4. Start the Application Server
```bash
python app.py
```

Browsers only allow camera and microphone access on **HTTPS or localhost**. To use a phone on the same Wi-Fi, start the server with a self-signed certificate and open `https://<your-pc-ip>:5000` on the phone (accept the certificate warning once; the certificate is kept in `.certs/` and reused):
```bash
pip install uvicorn cryptography
# PowerShell:  $env:AMCS_HTTPS="1"; python app.py
# Bash:        AMCS_HTTPS=1 python app.py
```
With `uvicorn` installed it serves HTTPS; Flask's built-in server also works but stalls for 1.5-2 s on every few HTTPS requests, which feels like lag on the phone.

Optional environment variables:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `AMCS_HOST` | `0.0.0.0` | Interface to listen on |
| `AMCS_PORT` | `5000` | Port |
| `AMCS_HTTPS` | off | Self-signed HTTPS for phone cameras |
| `AMCS_DEBUG` | off | Flask debug mode |
| `AMCS_SECRET_KEY` | random | Fixed Flask secret key |
| `AMCS_YOLO_MODEL` | `auto` | Detector weights. `auto` picks `yolo26m.pt` or `yolo26s.pt` by measured speed; name a file to force one (`yolov8n.pt` for very slow PCs) |
| `AMCS_YOLO_BUDGET_MS` | `330` | Per-frame time limit used by `auto` |

### 5. Access the Web Application
Open your browser and navigate to:
- **User Assistive Interface:** [http://localhost:5000](http://localhost:5000)
- **Caregiver Live Map & Tele-Assistance:** [http://localhost:5000/caregiver](http://localhost:5000/caregiver)
- **Examiner Viva Evaluation Sandbox:** [http://localhost:5000/sandbox](http://localhost:5000/sandbox)
- **Signboard OCR & Translator:** [http://localhost:5000/ocr](http://localhost:5000/ocr)
- **Project Proposal & Presentation Details:** [http://localhost:5000/about](http://localhost:5000/about)
- **Preferences (voice, feedback, theme):** [http://localhost:5000/settings](http://localhost:5000/settings)

### 6. Using the Interface
- **Themes:** the header buttons switch between system, light and dark themes, and toggle high contrast (yellow on black, larger text).
- **Mobile:** a bottom tab bar replaces the top navigation on phones; the app can be installed as a PWA.
- **Keyboard:** on the Assist page, <kbd>Space</kbd> starts or pauses, <kbd>D</kbd> describes the scene, <kbd>R</kbd> repeats, <kbd>V</kbd> starts a voice command, <kbd>M</kbd> mutes.
- **Screen readers:** spoken guidance is mirrored in a live region, all controls are labelled, and confirmations use native accessible dialogs.
- **Privacy:** camera frames are only shared with the caregiver while they have a tele-assistance call open.

---

## 🌐 Supported Languages (14+)

| Code | Language | Native Name | Speech Code |
| :--- | :--- | :--- | :--- |
| `en` | English | English | `en` |
| `hi` | Hindi | हिन्दी | `hi` |
| `es` | Spanish | Español | `es` |
| `fr` | French | Français | `fr` |
| `de` | German | Deutsch | `de` |
| `zh-CN` | Chinese (Simplified) | 中文 (简体) | `zh-CN` |
| `ja` | Japanese | 日本語 | `ja` |
| `ar` | Arabic | العربية | `ar` |
| `ru` | Russian | Русский | `ru` |
| `bn` | Bengali | বাংলা | `bn` |
| `ta` | Tamil | தமிழ் | `ta` |
| `te` | Telugu | తెలుగు | `te` |
| `mr` | Marathi | मराठी | `mr` |
| `gu` | Gujarati | ગુજરાતી | `gu` |

---

## 🎙️ Hands-Free Voice Commands
Visually impaired users can click or toggle the voice microphone and speak natural commands:
- *"Describe"* / *"What do you see"* &rarr; Captures a frame and speaks a description.
- *"What is in front of me?"* / *"Saamne kya hai?"* / *"Is there a place to sit?"* &rarr; Conversational scene Q&A. Double-tap the camera view to ask.
- *"What is this?"* / *"What am I holding?"* / *"Who is in front of me?"* &rarr; Object in hand (and its text), people by name.
- *"Find my keys"* &rarr; Starts the audio item radar. *"Stop radar"* ends it.
- *"Save this place as home"* &rarr; Remembers the current position under that name.
- *"Take me to home"* / *"Take me to library"* / *"Take me to Jaipur Junction"* &rarr; Turn-by-turn walking directions. *"Cancel navigation"* ends it.
- *"Where am I?"* &rarr; Street, area and the nearest saved place.
- *"Medicine"* / *"Currency"* / *"Read sign"* / *"Obstacle"* / *"Hybrid"* &rarr; Switches recognition mode.
- *"Find the door"* / *"What colour"* &rarr; Door and handle guidance, clothing colour and pattern.
- *"Hindi"* / *"English"* / *"Tamil"* … &rarr; Switches voice and translation language.
- *"Torch on"* / *"Torch off"* &rarr; Controls the flashlight.
- *"Start"* / *"Stop"* / *"Repeat"* / *"Mute"* &rarr; Assistant and speech control.
- *"Emergency"* / *"Help"* / *"SOS"* &rarr; Asks to confirm, then alerts caregivers with live location.
- *"I am fine"* &rarr; Cancels a fall-detection countdown.

---

## 📚 Literature Survey Citations & References
1. **Shaikh, S. (2020)** &mdash; *Assistive Object Recognition System for Visually Impaired*, IJERT, Vol 09, Issue 09.
2. **Naqvi, K., Hazela, B., Mishra, S., Asthana, P. (2021)** &mdash; *Employing Real-Time Object Detection for Visually Impaired People*, Springer Nature, LNDECT Vol 54, pp. 285–299.
3. **Alshaker, R. R. M. (2019)** &mdash; *Real Time Object Detection for Visually Impaired Person (YOLOv3 + Raspberry Pi + gTTS)*, Academia / University of Mosul.
