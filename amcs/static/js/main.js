/**
 * Automatic Multilanguage Communication System - Ultra-Smooth Client Engine
 * 60 FPS Native Video Stream with Transparent HUD Bounding Boxes & Smart Speech
 * SKIT Jaipur - Final Year Project
 */

// ── State ──
let isAutoRunning = false;
let currentMode = "vision"; // 'vision', 'ocr', 'hybrid'
let activeFacingMode = "environment";
let cameraStream = null;
let autoDetectTimer = null;
let isProcessingFrame = false;
let isMuted = false;
let screenWakeLock = null;
// Smoothed HUD boxes drawn over the video (was never declared, which crashed the render loop)
let trackedARObjects = [];

// Speech Debouncing & Cooldown
let lastSpokenText = "";
let lastSpokenLang = "en";
let lastSpokenAudioUrl = "";
let lastSpeechTimestamp = 0;
// User preferences saved from the Settings page (see static/js/ui.js)
const prefs = (window.AMCS && window.AMCS.getPrefs) ? window.AMCS.getPrefs() : {
    lang: null, speechRate: 0.95, speechIntervalSec: 7.5, haptics: true,
    earcons: true, autoTorch: true, acousticMonitor: true, gps: true
};
const SPEECH_MIN_INTERVAL_MS = Math.max(2000, (prefs.speechIntervalSec || 7.5) * 1000);
// Lite mode (phones by default): canvases redraw at a lower rate and without glow, which is the
// most expensive part of drawing on a phone GPU
const LITE_MODE = !!(window.AMCS && window.AMCS.liteActive && window.AMCS.liteActive(prefs));
const LITE_FRAME_MS = 1000 / 30;
const HUD_LERP = LITE_MODE ? 0.42 : 0.24;      // per drawn frame; 0.42 at 30 fps matches 0.24 at 60 fps
function withoutGlow(context) {
    if (LITE_MODE && context && !Object.prototype.hasOwnProperty.call(context, "shadowBlur")) {
        Object.defineProperty(context, "shadowBlur", { get: () => 0, set: () => {} });
    }
    return context;
}

// Latest GPS fix [lat, lon]; null until the browser reports a position
let currentUserCoords = null;
let currentUserAccuracy = null;   // GPS accuracy in meters
let currentUserFixAt = 0;
let gpsHeading = null;            // direction of travel from GPS, only while walking
let gpsHeadingAt = 0;
let gpsWatchId = null;
const FALLBACK_COORDS = [26.82285, 75.86475]; // SKIT campus; only for the SOS message, never for routing

function notify(title, message, type = "info") {
    if (window.AMCS && window.AMCS.toast) window.AMCS.toast({ title, message, type });
}

// One shared AudioContext: browsers cap how many can exist, and the item radar
// can ping up to 9 times a second.
let sharedAudioCtx = null;
function getAudioCtx() {
    const AudioCtx = window.AudioContext || window.webkitAudioContext;
    if (!AudioCtx) return null;
    if (!sharedAudioCtx) sharedAudioCtx = new AudioCtx();
    if (sharedAudioCtx.state === "suspended") sharedAudioCtx.resume().catch(() => {});
    return sharedAudioCtx;
}

// Loop interval (runs in background without affecting 60fps video). One frame every 0.6 s gives the
// server enough consecutive frames to confirm objects and measure motion; speech has its own cooldown.
// A new frame is only sent once the previous response has arrived, so slow PCs self-throttle.
// Live loop: the next frame goes out as soon as the previous answer is back, but not more often
// than FRAME_TARGET_MS. (A fixed 600 ms timer used to add up to half a second of waiting per frame.)
const FRAME_TARGET_MS = 250;
const FRAME_MIN_GAP_MS = 60;
let liveLoopId = 0;

// DOM Elements
const videoEl = document.getElementById("webcam");
const canvasEl = document.getElementById("annotated-canvas");
const ctx = canvasEl ? withoutGlow(canvasEl.getContext("2d")) : null;
const langSelect = document.getElementById("lang-select");
const speechAudio = document.getElementById("global-speech-player");

// Dedicated Offscreen Snapshot Canvas (Leaves displayed video 100% smooth)
const captureCanvas = document.createElement("canvas");
captureCanvas.width = 640;
captureCanvas.height = 360;
const captureCtx = captureCanvas.getContext("2d", { alpha: false, willReadFrequently: true });
// Detection quality (Settings): the frame is captured at the size the detector will run at.
// "auto" is fast on phones, where the upload over Wi-Fi is a large part of each frame's delay.
const DETECTION_QUALITY = (window.AMCS && window.AMCS.resolveQuality)
    ? window.AMCS.resolveQuality(prefs)
    : (["fast", "balanced", "high"].includes(prefs.quality) ? prefs.quality : "balanced");
const CAPTURE_LONG_SIDE = { fast: 512, mobile: 640, balanced: 640, high: 960 }[DETECTION_QUALITY] || 640;
const CAPTURE_JPEG_QUALITY = (DETECTION_QUALITY === "fast" || DETECTION_QUALITY === "mobile") ? 0.72 : 0.82;

// Frames sent for recognition keep the camera's aspect ratio. Squashing a 16:9 picture into
// 4:3 distorts every object and was a major cause of missed or wrong detections.
function syncCaptureSize() {
    const vw = videoEl && videoEl.videoWidth, vh = videoEl && videoEl.videoHeight;
    if (!vw || !vh) return;
    const scale = Math.min(1, CAPTURE_LONG_SIDE / Math.max(vw, vh));
    const w = Math.round(vw * scale), h = Math.round(vh * scale);
    if (captureCanvas.width !== w || captureCanvas.height !== h) {
        captureCanvas.width = w;
        captureCanvas.height = h;
    }
}

function captureFrame(quality = CAPTURE_JPEG_QUALITY) {
    syncCaptureSize();
    captureCtx.drawImage(videoEl, 0, 0, captureCanvas.width, captureCanvas.height);
    return captureCanvas.toDataURL("image/jpeg", quality);
}

// ─────────────────────────────────────────────
// 1. Camera Initialization (Full 60 FPS Native)
// ─────────────────────────────────────────────
function showCameraOverlay(title, text) {
    const overlay = document.getElementById("video-overlay-msg");
    if (!overlay) return;
    overlay.style.display = "flex";
    const t = document.getElementById("overlay-title");
    const p = document.getElementById("overlay-text");
    if (t) t.textContent = title;
    if (p) p.textContent = text;
}

let cameraStarting = null;       // in-flight start, so double clicks don't race
let activeCameraId = null;
let activeCameraLabel = "";
let activeModelLabel = "YOLO26";   // updated from the server's response
let cameraFacesUser = false;    // front / laptop camera: preview and boxes mirrored, left/right from the user's view

function detectFacingUser(stream, label) {
    const track = stream && stream.getVideoTracks()[0];
    const settings = track && track.getSettings ? track.getSettings() : {};
    if (settings.facingMode) return settings.facingMode === "user";
    // Laptop and desktop webcams don't report facingMode, and they face the user.
    // Phones do report it, so this fallback only applies to PCs.
    return !/back|rear|environment|world/i.test(label || "");
}

function applyCameraMirror() {
    if (videoEl) videoEl.style.transform = cameraFacesUser ? "scaleX(-1)" : "";
}

// Friendly names for the boxes drawn on the video
const AR_NAMES = {
    "dining table": "table", "tv": "TV", "cell phone": "phone", "couch": "sofa", "potted plant": "plant",
    "wall switch": "light switch", "trash can": "dustbin", "wine glass": "glass", "handbag": "bag"
};
function arLabel(d) {
    return d.is_user ? "You" : (AR_NAMES[d.class] || d.class);
}

async function startCamera(deviceId = null) {
    if (cameraStarting) return cameraStarting;
    cameraStarting = (async () => {
        const btn = document.getElementById("overlay-start-btn");
        if (btn) { btn.disabled = true; btn.innerHTML = '<i class="fa-solid fa-spinner fa-spin" aria-hidden="true"></i> Starting camera…'; }
        try {
            if (cameraStream) {
                cameraStream.getTracks().forEach(track => track.stop());
                cameraStream = null;
            }
            const cam = await AMCS.openCamera({ facingMode: activeFacingMode, width: 1280, height: 720, frameRate: 30, deviceId });
            cameraStream = cam.stream;
            activeCameraId = cam.deviceId;
            activeCameraLabel = cam.label || "Camera";
            cameraFacesUser = detectFacingUser(cam.stream, activeCameraLabel);
            applyCameraMirror();
            videoEl.srcObject = cameraStream;
            // play() can reject (e.g. interrupted by a re-start) even though the stream is fine
            try { await videoEl.play(); } catch (e) { console.warn("[AMCS] video.play():", e.name); }

            document.getElementById("video-overlay-msg").style.display = "none";
            syncCanvasDimensions();
            updateIslandTelemetry();
            const thisStream = cameraStream;
            const thisDevice = activeCameraId;
            AMCS.guardCameraStream(thisStream, {
                // Driver dropped the stream (or it was unplugged): reopen the same camera
                onReconnecting: () => {
                    if (cameraStream !== thisStream) return;
                    thisStream.getTracks().forEach(t => t.stop());
                    cameraStream = null;
                    showCameraOverlay("Reconnecting camera…", "The camera stopped for a moment. Reconnecting automatically.");
                },
                reopen: () => { if (!cameraStream) startCamera(thisDevice); },
                onGiveUp: () => {
                    if (cameraStream === thisStream) cameraStream = null;
                    showCameraOverlay("Camera keeps stopping",
                        "Windows keeps closing the laptop camera. Close other apps that use it (Zoom, Teams, the Camera app), then press Try again.");
                }
            });
            console.log(`[AMCS] Camera active: ${activeCameraLabel}`);
        } catch (err) {
            console.error("[AMCS] Camera error:", err.name, err.message, err.attempts || "");
            cameraStream = null;
            const msg = AMCS.describeCameraError(err);
            showCameraOverlay(msg.title, msg.text);
        } finally {
            if (btn) { btn.disabled = false; btn.innerHTML = '<i class="fa-solid fa-rotate-right" aria-hidden="true"></i> Try again'; }
            cameraStarting = null;
        }
    })();
    return cameraStarting;
}

function syncCanvasDimensions() {
    if (!videoEl || !canvasEl) return;
    const rect = videoEl.getBoundingClientRect();
    canvasEl.width = rect.width || videoEl.videoWidth || 640;
    canvasEl.height = rect.height || videoEl.videoHeight || 480;
}
window.addEventListener("resize", syncCanvasDimensions);

// ─────────────────────────────────────────────
// Audio Earcon Synthesizer (Zero-Latency Web Audio)
// ─────────────────────────────────────────────
function playEarcon(type) {
    if (isMuted || prefs.earcons === false) return;
    try {
        const actx = getAudioCtx();
        if (!actx) return;
        const now = actx.currentTime;

        if (type === "mode") {
            // Crystalline dual-tone ping
            const osc = actx.createOscillator();
            const gain = actx.createGain();
            osc.type = "sine";
            osc.frequency.setValueAtTime(587.33, now);
            osc.frequency.exponentialRampToValueAtTime(880.00, now + 0.12);
            gain.gain.setValueAtTime(0.08, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.22);
            osc.connect(gain);
            gain.connect(actx.destination);
            osc.start(now);
            osc.stop(now + 0.24);
        } else if (type === "lock") {
            // Soft precision glass click
            const osc = actx.createOscillator();
            const gain = actx.createGain();
            osc.type = "triangle";
            osc.frequency.setValueAtTime(1174.66, now);
            gain.gain.setValueAtTime(0.05, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.07);
            osc.connect(gain);
            gain.connect(actx.destination);
            osc.start(now);
            osc.stop(now + 0.08);
        } else if (type === "start") {
            // Ascending major triad
            [523.25, 659.25, 783.99].forEach((freq, idx) => {
                const osc = actx.createOscillator();
                const gain = actx.createGain();
                osc.type = "sine";
                osc.frequency.setValueAtTime(freq, now + idx * 0.07);
                gain.gain.setValueAtTime(0.06, now + idx * 0.07);
                gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.07 + 0.22);
                osc.connect(gain);
                gain.connect(actx.destination);
                osc.start(now + idx * 0.07);
                osc.stop(now + idx * 0.07 + 0.25);
            });
        } else if (type === "pause") {
            // Descending warm tone
            [783.99, 523.25].forEach((freq, idx) => {
                const osc = actx.createOscillator();
                const gain = actx.createGain();
                osc.type = "sine";
                osc.frequency.setValueAtTime(freq, now + idx * 0.09);
                gain.gain.setValueAtTime(0.05, now + idx * 0.09);
                gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.09 + 0.20);
                osc.connect(gain);
                gain.connect(actx.destination);
                osc.start(now + idx * 0.09);
                osc.stop(now + idx * 0.09 + 0.22);
            });
        } else if (type === "hazard") {
            // Low-frequency warning thud
            const osc = actx.createOscillator();
            const gain = actx.createGain();
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(220, now);
            osc.frequency.linearRampToValueAtTime(140, now + 0.24);
            gain.gain.setValueAtTime(0.10, now);
            gain.gain.exponentialRampToValueAtTime(0.001, now + 0.28);
            osc.connect(gain);
            gain.connect(actx.destination);
            osc.start(now);
            osc.stop(now + 0.30);
        }
    } catch (e) {}
}

// ─────────────────────────────────────────────
// Dynamic Island Telemetry & Notification Engine
// ─────────────────────────────────────────────
let islandTimer = null;
let currentFPS = 60;
let lastServerLatency = 38;
let lastFpsTimestamp = performance.now();
let frameCounter = 0;

function updateIslandTelemetry() {
    const el = document.getElementById("island-telemetry");
    if (!el) return;
    if (!cameraStream) el.textContent = "Camera idle";
    else if (!isAutoRunning) el.textContent = `Ready • ${currentFPS} FPS preview`;
    else el.textContent = `${currentFPS} FPS • ${lastServerLatency}ms • ${activeModelLabel}`;
}

function updateLiveChip() {
    const chip = document.getElementById("live-chip");
    const txt = document.getElementById("live-chip-text");
    if (!chip || !txt) return;
    chip.classList.toggle("on", isAutoRunning);
    txt.textContent = isAutoRunning ? "LIVE" : "STANDBY";
    if (window.AMCSPanels) {
        AMCSPanels.setLive(isAutoRunning);
        if (!isAutoRunning) AMCSPanels.reset();
    }
}

function showIslandNotification(title, subtitle, iconClass, isHazard = false, durationMs = 2800) {
    const island = document.getElementById("dynamic-island");
    const titleEl = document.getElementById("island-title");
    const subEl = document.getElementById("island-subtitle");
    const iconEl = document.getElementById("island-icon");
    if (!island) return;

    if (titleEl) titleEl.textContent = title;
    if (subEl) subEl.textContent = subtitle;
    if (iconEl) iconEl.className = iconClass;

    island.classList.remove("hazard-mode");
    if (isHazard) island.classList.add("hazard-mode");
    island.classList.add("expanded");

    clearTimeout(islandTimer);
    islandTimer = setTimeout(() => {
        island.classList.remove("expanded");
        island.classList.remove("hazard-mode");
    }, durationMs);
}

function expandIslandManually() {
    const island = document.getElementById("dynamic-island");
    if (island) {
        if (island.classList.contains("expanded")) {
            island.classList.remove("expanded");
        } else {
            showIslandNotification("AMCS Telemetry", `${currentFPS} FPS | Latency: ${lastServerLatency}ms | ${activeModelLabel}`, "fa-solid fa-microchip", false, 3500);
        }
    }
}

async function flipCamera() {
    // Cycle through the real cameras (phones: front/back; PCs: built-in, USB).
    // Virtual cameras such as OBS are skipped because they show nothing unless their app runs.
    const cams = AMCS.usableCameras(await AMCS.listCameras().catch(() => []));
    if (cams.length < 2) {
        notify("Only one camera", `${activeCameraLabel || "This camera"} is the only real camera. Virtual cameras like OBS are skipped.`);
        return;
    }
    const idx = cams.findIndex(c => c.deviceId === activeCameraId);
    const next = cams[(idx + 1) % cams.length];
    await startCamera(next.deviceId);
    if (cameraStream) {
        showIslandNotification("Camera switched", activeCameraLabel, "fa-solid fa-camera-rotate", false, 2200);
    }
}

// ─────────────────────────────────────────────
// 2. Mode Switching
// ─────────────────────────────────────────────
function setMode(mode) {
    currentMode = mode;
    document.querySelectorAll(".seg-btn").forEach(b => {
        b.classList.remove("active");
        b.setAttribute("aria-pressed", "false");
    });
    const activeBtn = document.getElementById(`mode-${mode}`);
    if (activeBtn) {
        activeBtn.classList.add("active");
        activeBtn.setAttribute("aria-pressed", "true");
    }
    clearCanvas();
    trackedARObjects = [];
    playEarcon("mode");

    const modeLabels = {
        vision: { title: "Obstacle Vision", sub: "Spatial proximity & collision warnings", icon: "fa-solid fa-person-walking" },
        ocr: { title: "Signboard OCR", sub: "14-Language text translation active", icon: "fa-solid fa-file-lines" },
        currency: { title: "Currency Reader", sub: "₹10 to ₹500 banknote identifier active", icon: "fa-solid fa-indian-rupee-sign" },
        medicine: { title: "Medicine Reader", sub: "Foil glare reduction & expiry reader active", icon: "fa-solid fa-pills" },
        hybrid: { title: "Hybrid Mode", sub: "Multimodal AI vision & text active", icon: "fa-solid fa-wand-magic-sparkles" }
    };
    const m = modeLabels[mode] || { title: "Mode Switched", sub: mode, icon: "fa-solid fa-wand-magic-sparkles" };
    showIslandNotification(m.title, m.sub, m.icon, false, 2400);
}

// ─────────────────────────────────────────────
// Screen Wake-Lock (Prevents Phone Screen Sleeping)
// ─────────────────────────────────────────────
async function acquireWakeLock() {
    try {
        if ('wakeLock' in navigator) {
            screenWakeLock = await navigator.wakeLock.request('screen');
            console.log("[AMCS] Screen Wake-Lock active (display will not sleep).");
        }
    } catch (e) {
        console.warn("[AMCS] WakeLock request error:", e);
    }
}

function releaseWakeLock() {
    if (screenWakeLock !== null) {
        screenWakeLock.release().catch(() => {});
        screenWakeLock = null;
    }
}

// ─────────────────────────────────────────────
// 3D Spatial Audio & Haptic Directional Guidance
// ─────────────────────────────────────────────
function playDirectionalSpatialChime(direction, severity) {
    if (isMuted) return;
    try {
        const actx = getAudioCtx();
        if (!actx) return;
        const osc = actx.createOscillator();
        const gain = actx.createGain();

        // 3D Stereo Panner: Left = -0.85, Center = 0.0, Right = 0.85
        if (actx.createStereoPanner) {
            const panner = actx.createStereoPanner();
            const panVal = direction === "left" ? -0.85 : (direction === "right" ? 0.85 : 0.0);
            panner.pan.setValueAtTime(panVal, actx.currentTime);
            osc.connect(panner);
            panner.connect(gain);
        } else {
            osc.connect(gain);
        }

        const baseFreq = severity === "hazard" ? 880 : 520;
        osc.frequency.setValueAtTime(baseFreq, actx.currentTime);
        osc.type = "sine";

        gain.gain.setValueAtTime(0.12, actx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.25);
        gain.connect(actx.destination);

        osc.start();
        osc.stop(actx.currentTime + 0.28);
    } catch (e) {}
}

function triggerHapticFeedback(severity, direction) {
    if (!("vibrate" in navigator) || prefs.haptics === false) return;
    if (severity === "hazard") {
        // Urgent double-pulse for danger
        navigator.vibrate([180, 70, 180]);
    } else if (severity === "caution") {
        // Subtle single tap
        navigator.vibrate([80]);
    }
}

// ─────────────────────────────────────────────
// 3. Hands-Free Assistant Toggle
// ─────────────────────────────────────────────
function toggleAutoDetection() {
    const btn = document.getElementById("start-auto-btn");
    const icon = document.getElementById("auto-btn-icon");
    const text = document.getElementById("auto-btn-text");

    if (isAutoRunning) {
        // Stop
        isAutoRunning = false;
        liveLoopId++;
        clearTimeout(autoDetectTimer);
        autoDetectTimer = null;
        releaseWakeLock();
        btn.classList.remove("active");
        if (icon) icon.className = "fa-solid fa-play";
        if (text) text.textContent = "Start Assistant";
        updateLiveChip();
        updateIslandTelemetry();
        setSpeakingIndicator(false);
        clearCanvas();
        trackedARObjects = [];
        playEarcon("pause");
        showIslandNotification("Assistant Paused", "Tap start to resume guidance", "fa-solid fa-pause", false, 2000);
    } else {
        // Start
        if (!cameraStream) startCamera();
        acquireWakeLock();
        isAutoRunning = true;
        btn.classList.add("active");
        if (icon) icon.className = "fa-solid fa-pause";
        if (text) text.textContent = "Pause Assistant";
        updateLiveChip();
        updateIslandTelemetry();
        playEarcon("start");
        showIslandNotification("Assistant Active", "Real-time 60 FPS vision online", "fa-solid fa-play", false, 2400);

        // Immediate first snapshot, then the continuous loop
        processCurrentFrame(true);
        const loopId = ++liveLoopId;
        autoDetectTimer = setTimeout(() => runLiveFrame(loopId), FRAME_TARGET_MS);
    }
}

async function runLiveFrame(loopId) {
    if (!isAutoRunning || loopId !== liveLoopId) return;     // paused, or a newer loop took over
    const started = performance.now();
    if (!isProcessingFrame) await processCurrentFrame(false);
    if (!isAutoRunning || loopId !== liveLoopId) return;
    const wait = Math.max(FRAME_MIN_GAP_MS, FRAME_TARGET_MS - (performance.now() - started));
    autoDetectTimer = setTimeout(() => runLiveFrame(loopId), wait);
}

// ─────────────────────────────────────────────
// 4. Lightweight Frame Snapshot & Processing
// ─────────────────────────────────────────────
async function processCurrentFrame(forceSpeech = false) {
    if (!videoEl || videoEl.readyState < 2 || isProcessingFrame) return;

    isProcessingFrame = true;

    // Snapshot directly to offscreen canvas (instant, zero impact on displayed video)
    let base64Image;
    try {
        base64Image = captureFrame();
        checkAmbientLuminanceAndTorch();
    } catch (e) {
        isProcessingFrame = false;
        return;
    }
    // Caregiver messages are checked every 1.5 s; during a live caregiver call every frame is shared
    if (intercomCallActive || Date.now() - intercomCheckedAt >= INTERCOM_CHECK_MS) {
        intercomCheckedAt = Date.now();
        syncTeleAssistIntercom(base64Image);
    }
    const selectedLang = langSelect ? langSelect.value : "en";

    try {
        const frameStart = performance.now();
        const response = await fetch("/api/detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                image: base64Image,
                lang: selectedLang,
                mode: currentMode,
                force_speech: forceSpeech,
                camera_facing: cameraFacesUser ? "user" : "environment",
                tts: !AMCS.voice.useDeviceVoice(selectedLang),
                quality: DETECTION_QUALITY,
                announced_key: speechState.spokenSceneKey,
                // The server translates and voices a message only when this page is going to speak it
                announced_hazard: {
                    key: speechState.hazardKey,
                    distance: speechState.hazardDist,
                    due: !!speechState.hazardKey && Date.now() - speechState.hazardAt > HAZARD_REPEAT_MS
                },
                return_annotated_image: false
            })
        });

        if (!response.ok) throw new Error(`Detect API returned HTTP ${response.status}`);
        const data = await response.json();
        lastServerLatency = Math.round(performance.now() - frameStart);

        if (data.status === "success") {
            if (data.stats && data.stats.model) {
                activeModelLabel = data.stats.model.replace(/\.pt$/, "").replace(/^yolo/i, "YOLO") + (data.stats.extended_model ? " + YOLOE" : "");
            }
            // Update smooth AR target coordinates with 60 FPS lerp
            const serverW = (data.frame_size && data.frame_size[0]) || captureCanvas.width;
            const serverH = (data.frame_size && data.frame_size[1]) || captureCanvas.height;
            updateTrackedARObjects(data.detections || [], serverW, serverH);

            // Update guidance text
            updateGuidanceUI(data);

            // Update surroundings radar strip
            updateSurroundingsStrip(data.detections || []);
            // A pause pressed while this frame was in flight must not repaint the panels
            if (window.AMCSPanels && (isAutoRunning || forceSpeech)) AMCSPanels.update(data, { latency: lastServerLatency });

            // Speech, sounds and vibration are decided together so nothing repeats every frame
            decideAndPlaySpeech(data, forceSpeech);
        }
    } catch (err) {
        console.warn("[AMCS] Frame pipeline warning:", err);
    } finally {
        isProcessingFrame = false;
    }
}

// ─────────────────────────────────────────────
// 5. Client-Side HUD Bounding Boxes (Transparent)
// ─────────────────────────────────────────────
function clearCanvas() {
    if (ctx && canvasEl) {
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);
    }
}

function updateTrackedARObjects(newDetections, srcW, srcH) {
    syncCanvasDimensions();
    // Same mapping as the <video>'s object-fit: cover - scale to fill, crop the overflow equally
    const scale = Math.max(canvasEl.width / srcW, canvasEl.height / srcH);
    const offX = (canvasEl.width - srcW * scale) / 2;
    const offY = (canvasEl.height - srcH * scale) / 2;

    trackedARObjects.forEach(o => o.matched = false);

    (newDetections || []).forEach((d, idx) => {
        const [x1, y1, x2, y2] = d.bbox;
        const targetW = (x2 - x1) * scale;
        const targetH = (y2 - y1) * scale;
        // Front cameras are shown mirrored, so the boxes are mirrored too
        const targetX = cameraFacesUser ? canvasEl.width - (x2 * scale + offX) : x1 * scale + offX;
        const targetY = y1 * scale + offY;

        let bestMatch = null;
        let bestDist = 130;

        for (const obj of trackedARObjects) {
            if (!obj.matched) {
                const dist = Math.hypot((obj.targetX + obj.targetW / 2) - (targetX + targetW / 2),
                                       (obj.targetY + obj.targetH / 2) - (targetY + targetH / 2));
                if (dist < bestDist) {
                    bestDist = dist;
                    bestMatch = obj;
                }
            }
        }

        const isDynamic = d.collision_risk === 'IMMINENT_HAZARD' && d.ttc_seconds !== null;

        if (bestMatch) {
            bestMatch.matched = true;
            bestMatch.targetX = targetX;
            bestMatch.targetY = targetY;
            bestMatch.targetW = targetW;
            bestMatch.targetH = targetH;
            bestMatch.label = arLabel(d);
            bestMatch.distanceM = d.distance_m;
            bestMatch.severity = d.severity;
            bestMatch.isDynamicCollision = isDynamic;
            bestMatch.ttcSeconds = d.ttc_seconds;
            bestMatch.speedKmh = d.speed_kmh;
            bestMatch.evasiveAction = d.evasive_action;
            bestMatch.isFadingOut = false;
        } else {
            const newObj = {
                id: 'obj_' + Date.now() + '_' + idx,
                currentX: targetX,
                currentY: targetY,
                currentW: targetW,
                currentH: targetH,
                targetX: targetX,
                targetY: targetY,
                targetW: targetW,
                targetH: targetH,
                label: arLabel(d),
                distanceM: d.distance_m,
                severity: d.severity,
                isDynamicCollision: isDynamic,
                ttcSeconds: d.ttc_seconds,
                speedKmh: d.speed_kmh,
                evasiveAction: d.evasive_action,
                alpha: 0.15,
                matched: true,
                isFadingOut: false,
                pulsePhase: Math.random() * Math.PI
            };
            trackedARObjects.push(newObj);
        }
    });

    trackedARObjects.forEach(o => {
        if (!o.matched) o.isFadingOut = true;
    });
}

let lastHudDraw = 0;
function renderARHUDLoop(timestamp) {
    if (LITE_MODE && timestamp - lastHudDraw < LITE_FRAME_MS) {
        requestAnimationFrame(renderARHUDLoop);
        return;
    }
    lastHudDraw = timestamp;
    frameCounter++;
    if (timestamp - lastFpsTimestamp >= 1000) {
        currentFPS = Math.min(60, Math.round((frameCounter * 1000) / (timestamp - lastFpsTimestamp)));
        frameCounter = 0;
        lastFpsTimestamp = timestamp;
        updateIslandTelemetry();
    }

    if (ctx && canvasEl) {
        ctx.clearRect(0, 0, canvasEl.width, canvasEl.height);

        for (let i = trackedARObjects.length - 1; i >= 0; i--) {
            const obj = trackedARObjects[i];

            obj.currentX += (obj.targetX - obj.currentX) * HUD_LERP;
            obj.currentY += (obj.targetY - obj.currentY) * HUD_LERP;
            obj.currentW += (obj.targetW - obj.currentW) * HUD_LERP;
            obj.currentH += (obj.targetH - obj.currentH) * HUD_LERP;

            if (obj.isFadingOut) {
                obj.alpha -= 0.08;
                if (obj.alpha <= 0) {
                    trackedARObjects.splice(i, 1);
                    continue;
                }
            } else {
                obj.alpha = Math.min(1.0, obj.alpha + 0.14);
            }

            obj.pulsePhase += 0.06;

            const boxX = obj.currentX;
            const boxY = obj.currentY;
            const boxW = Math.max(10, obj.currentW);
            const boxH = Math.max(10, obj.currentH);

            let strokeColor = '#45d9e8';
            let glowColor = 'rgba(69, 217, 232, 0.45)';
            let pillBg = 'rgba(10, 12, 18, 0.82)';

            if (obj.isDynamicCollision || obj.severity === 'hazard') {
                strokeColor = '#f87171';
                glowColor = 'rgba(248, 113, 113, 0.6)';
                pillBg = obj.isDynamicCollision ? 'rgba(220, 38, 38, 0.96)' : 'rgba(239, 68, 68, 0.92)';
            } else if (obj.severity === 'caution') {
                strokeColor = '#fbbf24';
                glowColor = 'rgba(251, 191, 36, 0.5)';
                pillBg = 'rgba(245, 158, 11, 0.92)';
            }

            ctx.save();
            ctx.globalAlpha = obj.alpha;

            const cornerLen = Math.min(20, boxW / 3, boxH / 3);
            ctx.strokeStyle = strokeColor;
            ctx.lineWidth = obj.isDynamicCollision ? 3.2 : 2.2;
            ctx.lineCap = 'round';
            ctx.shadowBlur = 12;
            ctx.shadowColor = glowColor;

            ctx.beginPath();
            ctx.moveTo(boxX, boxY + cornerLen);
            ctx.lineTo(boxX, boxY);
            ctx.lineTo(boxX + cornerLen, boxY);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(boxX + boxW - cornerLen, boxY);
            ctx.lineTo(boxX + boxW, boxY);
            ctx.lineTo(boxX + boxW, boxY + cornerLen);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(boxX + boxW, boxY + boxH - cornerLen);
            ctx.lineTo(boxX + boxW, boxY + boxH);
            ctx.lineTo(boxX + boxW - cornerLen, boxY + boxH);
            ctx.stroke();

            ctx.beginPath();
            ctx.moveTo(boxX + cornerLen, boxY + boxH);
            ctx.lineTo(boxX, boxY + boxH);
            ctx.lineTo(boxX, boxY + boxH - cornerLen);
            ctx.stroke();

            if (obj.distanceM && obj.distanceM <= 1.2) {
                const pulseR = 14 + Math.sin(obj.pulsePhase) * 6;
                ctx.beginPath();
                ctx.arc(boxX + boxW / 2, boxY + boxH / 2, pulseR, 0, Math.PI * 2);
                ctx.strokeStyle = strokeColor;
                ctx.lineWidth = 1.2;
                ctx.stroke();
            }

            let label = obj.label.charAt(0).toUpperCase() + obj.label.slice(1) + ' · ' + obj.distanceM + ' m';
            if (obj.isDynamicCollision) {
                const evasiveIcon = obj.evasiveAction === 'step_right' ? '➔ STEP RIGHT' : (obj.evasiveAction === 'step_left' ? '⬅ STEP LEFT' : '⚠️ STOP');
                label = '⚠️ ' + obj.label + ' | TTC ' + obj.ttcSeconds + 's (' + obj.speedKmh + 'km/h) ' + evasiveIcon;
            }

            ctx.font = obj.isDynamicCollision ? '700 12px Inter, system-ui, sans-serif' : '600 12px Inter, system-ui, sans-serif';
            const textMetrics = ctx.measureText(label);
            const tagW = textMetrics.width + 16;
            const tagH = 22;
            const tagX = Math.max(4, boxX);
            const tagY = Math.max(tagH + 4, boxY - 6);

            ctx.shadowBlur = 8;
            ctx.shadowColor = 'rgba(0,0,0,0.5)';
            ctx.fillStyle = pillBg;
            ctx.beginPath();
            if (ctx.roundRect) ctx.roundRect(tagX, tagY - tagH, tagW, tagH, 11);
            else ctx.rect(tagX, tagY - tagH, tagW, tagH);
            ctx.fill();

            ctx.shadowBlur = 0;
            ctx.fillStyle = '#ffffff';
            ctx.fillText(label, tagX + 8, tagY - 7);

            ctx.restore();
        }
    }

    requestAnimationFrame(renderARHUDLoop);
}

let isActivelySpeaking = false;
let wavePhase = 0;
let lastOrbDraw = 0;

function renderFluidAudioOrb(timestamp = 0) {
    const orbCanvas = document.getElementById('voice-orb-canvas');
    if (!orbCanvas || (LITE_MODE && timestamp - lastOrbDraw < 50)) {
        requestAnimationFrame(renderFluidAudioOrb);
        return;
    }
    lastOrbDraw = timestamp;
    const orbCtx = withoutGlow(orbCanvas.getContext('2d'));
    if (!orbCtx) return;

    const cssW = Math.round(orbCanvas.clientWidth || 400);
    if (cssW > 0 && orbCanvas.width !== cssW) { orbCanvas.width = cssW; orbCanvas.height = 40; }
    const w = orbCanvas.width;
    const h = orbCanvas.height;
    orbCtx.clearRect(0, 0, w, h);

    wavePhase += (isActivelySpeaking ? 0.08 : 0.025) * (LITE_MODE ? 3 : 1);   // same speed at the lower lite frame rate
    const baseAmp = isActivelySpeaking ? 12 : 3;
    const midY = h / 2;

    const grad1 = orbCtx.createLinearGradient(0, 0, w, 0);
    grad1.addColorStop(0, 'rgba(124, 138, 255, 0.1)');
    grad1.addColorStop(0.5, isActivelySpeaking ? 'rgba(124, 138, 255, 0.95)' : 'rgba(124, 138, 255, 0.45)');
    grad1.addColorStop(1, 'rgba(69, 217, 232, 0.15)');

    orbCtx.save();
    orbCtx.beginPath();
    orbCtx.moveTo(0, midY);

    for (let x = 0; x <= w; x += 4) {
        const norm = (x / w) * Math.PI * 2;
        const y = midY + Math.sin(norm * 2 + wavePhase) * baseAmp
                       + Math.sin(norm * 4 - wavePhase * 1.4) * (baseAmp * 0.45);
        orbCtx.lineTo(x, y);
    }
    orbCtx.strokeStyle = grad1;
    orbCtx.lineWidth = isActivelySpeaking ? 3.2 : 1.8;
    orbCtx.shadowBlur = isActivelySpeaking ? 16 : 6;
    orbCtx.shadowColor = '#7c8aff';
    orbCtx.stroke();

    if (isActivelySpeaking) {
        orbCtx.beginPath();
        orbCtx.moveTo(0, midY);
        for (let x = 0; x <= w; x += 4) {
            const norm = (x / w) * Math.PI * 2;
            const y = midY + Math.cos(norm * 3 + wavePhase * 1.8) * (baseAmp * 0.7);
            orbCtx.lineTo(x, y);
        }
        orbCtx.strokeStyle = 'rgba(69, 217, 232, 0.85)';
        orbCtx.lineWidth = 2.0;
        orbCtx.shadowBlur = 12;
        orbCtx.shadowColor = '#45d9e8';
        orbCtx.stroke();
    }

    orbCtx.restore();
    requestAnimationFrame(renderFluidAudioOrb);
}

// ─────────────────────────────────────────────
// 6. Intelligent Non-Annoying Speech Logic
// ─────────────────────────────────────────────
// ─────────────────────────────────────────────
// Speech director: decides what is worth saying
//  * every hazard is announced once (with a tone + vibration); it is repeated only if it comes
//    clearly closer, or after 30 s
//  * routine updates only when what is in view really changes and stays changed for 2 frames,
//    and then only the new things are mentioned ("Now also a cup ahead.")
//  * never more often than the "pause between announcements" setting; never over itself
// ─────────────────────────────────────────────
const HAZARD_REPEAT_MS = 30000;
let guidanceQuietUntil = 0;        // routine scene speech waits until then (questions, directions)

/** Holds routine scene speech back for about as long as `text` takes to say. */
function quietGuidanceFor(text, extraMs = 1500) {
    const ms = Math.min(20000, (text || "").length * 75 + extraMs);
    guidanceQuietUntil = Math.max(guidanceQuietUntil, Date.now() + ms);
}
const HAZARD_FORGET_MS = 4000;     // a hazard must be gone this long before it counts as new again
const SCENE_SETTLE_FRAMES = 2;
const SCENE_SETTLE_MS = 600;       // and for this long, now that frames arrive several times a second
const speechState = {
    spokenSceneKey: "",    // scene the user was last told about (sent to the server for "what's new")
    pendingSceneKey: "",
    pendingFrames: 0,
    pendingSince: 0,
    hazardKey: "",
    hazardAt: 0,
    hazardSeenAt: 0,
    hazardDist: 99
};

function decideAndPlaySpeech(data, forceSpeech) {
    if (isMuted) return;
    const S = speechState;
    const now = Date.now();
    const sceneKey = data.scene_key || "";

    if (forceSpeech) {
        S.spokenSceneKey = sceneKey;
        S.pendingFrames = 0;
        if (data.hazard_key) { S.hazardKey = data.hazard_key; S.hazardAt = S.hazardSeenAt = now; S.hazardDist = data.hazard_distance ?? 99; }
        speakGuidance(data, { full: true, interrupt: true });
        return;
    }

    // 1. Hazards
    if (data.is_urgent && data.hazard_key) {
        const dist = typeof data.hazard_distance === "number" ? data.hazard_distance : 99;
        const isNew = data.hazard_key !== S.hazardKey;
        const closer = !isNew && dist < S.hazardDist - 0.5;
        const stale = !isNew && now - S.hazardAt > HAZARD_REPEAT_MS;
        S.hazardSeenAt = now;
        if (isNew || closer || stale) {
            S.hazardKey = data.hazard_key;
            S.hazardAt = now;
            S.hazardDist = dist;
            S.spokenSceneKey = sceneKey;
            S.pendingFrames = 0;
            speakGuidance(data, { full: true, interrupt: true, alert: isNew || closer ? (data.priority === "CRITICAL" ? "critical" : "hazard") : null });
        }
        return;
    }
    if (S.hazardKey && now - S.hazardSeenAt > HAZARD_FORGET_MS) {
        S.hazardKey = "";
        S.hazardDist = 99;
    }

    // 2. Routine scene updates (held back while the user asks a question or hears the answer)
    if (now < guidanceQuietUntil) return;
    if (sceneKey === S.spokenSceneKey) { S.pendingFrames = 0; return; }
    if (sceneKey === S.pendingSceneKey) S.pendingFrames++;
    else { S.pendingSceneKey = sceneKey; S.pendingFrames = 1; S.pendingSince = now; }
    if (S.pendingFrames < SCENE_SETTLE_FRAMES || now - S.pendingSince < SCENE_SETTLE_MS) return;
    if (now - lastSpeechTimestamp < SPEECH_MIN_INTERVAL_MS || isActivelySpeaking) return;

    S.pendingFrames = 0;
    // The server only fills speech_en with what is new. If things only disappeared it is empty:
    // stay quiet and keep remembering them, so an object that flickers back is not announced again.
    if (data.speech_en === "") return;
    S.spokenSceneKey = sceneKey;
    speakGuidance(data, { full: false, interrupt: false });
}

/** Plays a guidance message. opts: { full, interrupt, alert: null|"hazard"|"critical" } */
function speakGuidance(data, opts = {}) {
    const text = opts.full
        ? (data.description_translated || data.description_en)
        : (data.speech_translated || data.speech_en || data.description_translated || data.description_en);
    if (!text) return;
    if (opts.alert) {
        const nearest = (data.detections || []).find(d => d.severity === "hazard") || (data.detections || [])[0];
        if (prefs.earcons !== false) playDirectionalSpatialChime(nearest ? nearest.spatial : "ahead", "hazard");
        triggerHapticFeedback("hazard", nearest ? nearest.spatial : "ahead");
        showIslandNotification(opts.alert === "critical" ? "Watch out!" : "Careful",
            data.description_en, "fa-solid fa-triangle-exclamation", true, 3200);
    }
    lastSpeechTimestamp = Date.now();
    speakText(text, data.target_lang, {
        audioUrl: opts.full ? data.audio_url : (data.speech_audio_url || data.audio_url),
        interrupt: opts.interrupt !== false,
        alert: !!opts.alert
    });
}

/**
 * Speaks text. Uses the device's natural voice when it has one for this language (instant, human
 * sounding); otherwise plays the server's audio file. opts: { audioUrl, interrupt=true }
 */
function speakText(text, lang, opts = {}) {
    if (isMuted || (!text && !opts.audioUrl)) return;
    text = text || "";
    lang = lang || (langSelect ? langSelect.value : "en");
    lastSpokenText = text;
    lastSpokenLang = lang;
    if (window.AMCSPanels) AMCSPanels.logGuidance(text, opts.alert);
    const interrupt = opts.interrupt !== false;
    if (interrupt) { speechAudio.pause(); AMCS.voice.stop(); }

    const viaDevice = () => AMCS.voice.speak(text, lang, {
        interrupt,
        onstart: () => setSpeakingIndicator(true),
        onend: () => setSpeakingIndicator(false)
    });

    if ((AMCS.voice.useDeviceVoice(lang) && text.trim()) || !opts.audioUrl) {
        lastSpokenAudioUrl = "";
        if (!viaDevice()) setSpeakingIndicator(false);
        return;
    }
    lastSpokenAudioUrl = opts.audioUrl;
    setSpeakingIndicator(true);
    speechAudio.src = opts.audioUrl;
    speechAudio.playbackRate = Math.min(1.5, Math.max(0.75, Number(prefs.speechRate) || 1));
    speechAudio.onended = () => setSpeakingIndicator(false);
    speechAudio.play().catch(() => viaDevice());
}

// Kept for the other tools (scene Q&A, navigation, item finder...)
function playSpeechAudio(data) {
    speakText(data.description_translated || data.description_en, data.target_lang, { audioUrl: data.audio_url });
}

function speakBrowserFallback(text, lang) {
    speakText(text, lang, {});
}

function playAudioUrl(url, fallbackText, lang) {
    speakText(fallbackText || "", lang, { audioUrl: url });
}

function replayAudio() {
    if (!lastSpokenText) {
        notify("Nothing to repeat yet", "Start the assistant or ask a question first.");
        return;
    }
    speakText(lastSpokenText, lastSpokenLang, { audioUrl: lastSpokenAudioUrl });
}

function setSpeakingIndicator(speaking) {
    isActivelySpeaking = speaking;
    const dot = document.getElementById("orb-status-dot");
    const label = document.getElementById("voice-status-label");
    if (dot) {
        if (speaking) dot.classList.add("speaking");
        else dot.classList.remove("speaking");
    }
    if (label) {
        label.textContent = speaking ? "Speaking…" : (isAutoRunning ? "Listening to scene" : "Standby");
    }
}

function toggleMute() {
    isMuted = !isMuted;
    const btn = document.getElementById("mute-btn");
    if (btn) {
        btn.setAttribute("aria-pressed", isMuted ? "true" : "false");
        if (isMuted) {
            btn.classList.add("muted");
            btn.innerHTML = '<i class="fa-solid fa-volume-xmark" aria-hidden="true"></i>';
            AMCS.voice.stop();
            speechAudio.pause();
            setSpeakingIndicator(false);
        } else {
            btn.classList.remove("muted");
            btn.innerHTML = '<i class="fa-solid fa-volume-high" aria-hidden="true"></i>';
        }
    }
}

// ─────────────────────────────────────────────
// 7. Refined UI Update Handlers
// ─────────────────────────────────────────────
function updateGuidanceUI(data) {
    const translatedEl = document.getElementById("speech-translated-text");
    const originalEl = document.getElementById("speech-original-text");
    const badge = document.getElementById("hazard-total-badge");
    const countTag = document.getElementById("detected-count");

    const spoken = data.description_translated || data.description_en || "";
    if (translatedEl && spoken && translatedEl.textContent !== spoken) translatedEl.textContent = spoken;
    if (originalEl) {
        // Only show the English line when it differs from what was spoken
        originalEl.textContent = (data.description_en && data.description_en !== spoken) ? data.description_en : "";
    }
    const card = document.getElementById("guidance-card");

    const count = (data.detections || []).length;
    if (countTag) countTag.textContent = `${count} detected`;

    if (badge && data.stats) {
        const hazards = data.stats.hazard_count || 0;
        const cautions = data.stats.caution_count || 0;
        if (hazards > 0) {
            badge.className = "badge badge-danger";
            badge.textContent = hazards === 1 ? "1 Hazard" : `${hazards} Hazards`;
        } else if (cautions > 0) {
            badge.className = "badge badge-warning";
            badge.textContent = `${cautions} Caution`;
        } else {
            badge.className = "badge badge-safe";
            badge.textContent = "Path Clear";
        }
        if (card) {
            card.classList.toggle("is-hazard", hazards > 0 || !!data.is_urgent);
            card.classList.toggle("is-caution", hazards === 0 && cautions > 0);
        }
    }
}

function updateSurroundingsStrip(detections) {
    const bySpatial = { left: [], ahead: [], right: [] };
    detections.forEach(d => {
        if (bySpatial[d.spatial]) bySpatial[d.spatial].push(d);
    });

    ["left", "ahead", "right"].forEach(sp => {
        const valEl = document.getElementById(`spatial-${sp}-val`);
        if (!valEl) return;

        const items = bySpatial[sp];
        if (items.length === 0) {
            valEl.innerHTML = '<span class="status-clear">Clear</span>';
        } else {
            // Nearest obstacle in this zone
            const nearest = items[0];
            const sev = ["hazard", "caution", "safe"].includes(nearest.severity) ? nearest.severity : "safe";
            const esc = window.AMCS ? window.AMCS.escapeHtml : (s => String(s));
            const extra = items.length > 1 ? ` +${items.length - 1}` : "";
            const name = arLabel(nearest);
            valEl.innerHTML = `<span class="obstacle-chip sev-${sev}" title="${esc(name)}">${esc(name)} · ${esc(nearest.distance_m)} m${extra}</span>`;
        }
    });
}

// ─────────────────────────────────────────────
// 8. GPS Broadcaster
// ─────────────────────────────────────────────
function initGPSTracker() {
    const coordsEl = document.getElementById("gps-coords");
    if (prefs.gps === false) {
        if (coordsEl) coordsEl.textContent = "Location sharing off (Settings)";
        return;
    }
    startGPSWatch();
}

/**
 * Follows the phone's GPS. Runs while location sharing is on, and always while walking directions
 * are active or a place is being saved (the user asked for those). Only shares the position with
 * the caregiver page when sharing is on.
 */
function startGPSWatch() {
    const coordsEl = document.getElementById("gps-coords");
    if (gpsWatchId !== null) return true;
    if (!("geolocation" in navigator)) {
        if (coordsEl) coordsEl.textContent = "Location not supported";
        return false;
    }
    let lastShared = 0;
    gpsWatchId = navigator.geolocation.watchPosition(
        pos => {
            const lat = pos.coords.latitude;
            const lon = pos.coords.longitude;
            currentUserCoords = [lat, lon];
            currentUserAccuracy = pos.coords.accuracy;
            currentUserFixAt = Date.now();
            if (typeof pos.coords.heading === "number" && !isNaN(pos.coords.heading) && (pos.coords.speed || 0) > 0.6) {
                gpsHeading = pos.coords.heading;
                gpsHeadingAt = Date.now();
            }
            if (coordsEl) coordsEl.textContent = `${lat.toFixed(5)}, ${lon.toFixed(5)} · ±${Math.round(pos.coords.accuracy)}m`;
            if (prefs.gps !== false && Date.now() - lastShared > 10000) {
                lastShared = Date.now();
                fetch("/api/location/update", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ latitude: lat, longitude: lon, accuracy: pos.coords.accuracy })
                }).catch(() => {});
            }
        },
        err => {
            if (coordsEl) coordsEl.textContent = err && err.code === 1 ? "Location permission denied" : "Location unavailable";
            if (err && err.code === 1) { navigator.geolocation.clearWatch(gpsWatchId); gpsWatchId = null; }
        },
        { enableHighAccuracy: true, maximumAge: 3000, timeout: 20000 }
    );
    return true;
}

/** Stops following GPS when location sharing is off and nothing needs it any more. */
function releaseGPSIfUnused() {
    if (prefs.gps === false && !isNavigating && gpsWatchId !== null) {
        navigator.geolocation.clearWatch(gpsWatchId);
        gpsWatchId = null;
    }
}

/** The user's position, waiting up to `waitMs` for a fix less than 30 s old. Resolves null if none. */
async function getUserPosition(waitMs = 12000) {
    if (!startGPSWatch()) return null;
    const fresh = () => currentUserCoords && Date.now() - currentUserFixAt < 30000;
    const t0 = Date.now();
    while (!fresh() && Date.now() - t0 < waitMs) {
        await new Promise(r => setTimeout(r, 250));
    }
    return fresh() ? currentUserCoords : null;
}

// Compass: the way the back camera points (phone held upright) or, lying flat, the way its top points
let compassHeading = null;
let compassAt = 0;
let compassStarted = false;
let compassSin = 0, compassCos = 1;

function onCompassReading(e) {
    let heading = null;
    if (typeof e.webkitCompassHeading === "number" && !isNaN(e.webkitCompassHeading)) {
        heading = e.webkitCompassHeading;                         // iPhone: already a compass heading
    } else if (e.absolute !== false && e.alpha !== null && e.beta !== null && e.gamma !== null) {
        const r = Math.PI / 180;
        const a = e.alpha * r, b = e.beta * r, g = e.gamma * r;
        // Direction out of the back of the phone, in east/north coordinates (W3C rotation matrix)
        const vx = Math.cos(a) * -Math.sin(g) - Math.sin(a) * Math.sin(b) * Math.cos(g);
        const vy = Math.sin(a) * -Math.sin(g) + Math.cos(a) * Math.sin(b) * Math.cos(g);
        if (Math.hypot(vx, vy) > 0.35) heading = (Math.atan2(vx, vy) / r + 360) % 360;
        else heading = (360 - e.alpha) % 360;                     // lying flat: the top of the phone
        const screenAngle = (screen.orientation && screen.orientation.angle) || window.orientation || 0;
        if (Math.hypot(vx, vy) <= 0.35) heading = (heading + screenAngle + 360) % 360;
    }
    if (heading === null) return;
    // Smooth the jitter (on the circle, so 359 and 1 average to 0)
    const k = 0.25;
    compassSin = compassSin * (1 - k) + Math.sin(heading * Math.PI / 180) * k;
    compassCos = compassCos * (1 - k) + Math.cos(heading * Math.PI / 180) * k;
    compassHeading = (Math.atan2(compassSin, compassCos) * 180 / Math.PI + 360) % 360;
    compassAt = Date.now();
}

async function startCompass() {
    if (compassStarted) return;
    try {
        // iPhone asks for permission (only from a tap; voice-started routes fall back to GPS heading)
        if (window.DeviceOrientationEvent && typeof DeviceOrientationEvent.requestPermission === "function") {
            const state = await DeviceOrientationEvent.requestPermission();
            if (state !== "granted") return;
        }
    } catch (e) { return; }
    compassStarted = true;
    if ("ondeviceorientationabsolute" in window) window.addEventListener("deviceorientationabsolute", onCompassReading);
    else window.addEventListener("deviceorientation", onCompassReading);
}

/** The way the user faces: compass when it is reporting, else the GPS direction of travel. */
function currentHeading() {
    if (compassHeading !== null && Date.now() - compassAt < 2000) return Math.round(compassHeading);
    if (gpsHeading !== null && Date.now() - gpsHeadingAt < 5000) return Math.round(gpsHeading);
    return null;
}

// ─────────────────────────────────────────────
// 9. SOS
// ─────────────────────────────────────────────
async function triggerSOS() {
    const curLang = langSelect ? langSelect.value : "en";
    speakBrowserFallback("Send emergency alert to your caregiver? Press confirm.", curLang);
    const ok = await AMCS.confirm({
        title: "Send emergency SOS?",
        message: "Your caregiver will see an alert with your live location.",
        icon: "fa-triangle-exclamation",
        danger: true,
        confirmText: "Send SOS"
    });
    if (!ok) return;
    try {
        const body = { message: "EMERGENCY: SOS button triggered in Assistive View!" };
        if (currentUserCoords) { body.latitude = currentUserCoords[0]; body.longitude = currentUserCoords[1]; }
        const res = await fetch("/api/sos/trigger", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(body)
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        playEarcon("hazard");
        speakBrowserFallback("Emergency alert sent. Your caregiver has your location.", curLang);
        notify("SOS sent", "Caregivers have received your location.", "success");
        if (window.AMCS) window.AMCS.checkSOS();
        if (data.whatsapp_url) {
            const sendWA = await AMCS.confirm({
                title: "Share on WhatsApp too?",
                message: "Open WhatsApp to send your Google Maps location to your family contact.",
                icon: "fa-share-from-square",
                confirmText: "Open WhatsApp",
                cancelText: "Not now"
            });
            if (sendWA) window.open(data.whatsapp_url, "_blank", "noopener");
        }
    } catch (e) {
        speakBrowserFallback("Could not send the emergency alert. Check the connection.", curLang);
        notify("SOS failed", "Network error. Call your caregiver directly if you need help.", "error");
    }
}

// ─────────────────────────────────────────────
// 10. Hands-Free Voice Command Recognition
// ─────────────────────────────────────────────
let speechRecognizer = null;
let isListeningForVoice = false;

function toggleVoiceRecognition() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    const micBtn = document.getElementById("mic-btn");

    const setMicUI = (on) => {
        if (!micBtn) return;
        micBtn.classList.toggle("listening", on);
        micBtn.setAttribute("aria-pressed", on ? "true" : "false");
    };

    if (!SpeechRecognition) {
        speakBrowserFallback("Voice recognition is not supported in this browser. Please use Chrome or Edge.", "en");
        notify("Voice commands unavailable", "Use Chrome or Edge for speech recognition.", "warning");
        return;
    }

    if (isListeningForVoice && speechRecognizer) {
        speechRecognizer.stop();
        isListeningForVoice = false;
        setMicUI(false);
        return;
    }

    try {
        speechRecognizer = new SpeechRecognition();
        speechRecognizer.continuous = false;
        speechRecognizer.interimResults = false;
        const RECOGNITION_LOCALES = { en: "en-IN", hi: "hi-IN", bn: "bn-IN", ta: "ta-IN", te: "te-IN", mr: "mr-IN", gu: "gu-IN", es: "es-ES", fr: "fr-FR", de: "de-DE", ja: "ja-JP", ar: "ar-SA", ru: "ru-RU", "zh-CN": "zh-CN" };
        const selected = langSelect ? langSelect.value : "en";
        speechRecognizer.lang = RECOGNITION_LOCALES[selected] || "en-IN";

        speechRecognizer.onstart = () => {
            isListeningForVoice = true;
            setMicUI(true);
            // Silence the guidance so the microphone hears the user, not the phone
            speechAudio.pause();
            AMCS.voice.stop();
            guidanceQuietUntil = Date.now() + 8000;
            playDirectionalSpatialChime("ahead", "caution");
        };

        speechRecognizer.onresult = (event) => {
            const transcript = event.results[0][0].transcript.toLowerCase().trim();
            if (!transcript) return;
            console.log("[AMCS Voice Command]:", transcript);
            handleVoiceCommand(transcript);
        };

        speechRecognizer.onerror = (e) => {
            console.warn("[AMCS] Voice recognition error:", e.error);
            isListeningForVoice = false;
            setMicUI(false);
            if (e.error === "not-allowed" || e.error === "service-not-allowed") {
                notify("Microphone blocked", "Allow microphone access to use voice commands.", "warning");
            } else if (e.error === "no-speech") {
                notify("Didn't catch that", "Tap the microphone and speak after the tone.");
            }
        };

        speechRecognizer.onend = () => {
            isListeningForVoice = false;
            setMicUI(false);
        };

        speechRecognizer.start();
    } catch (e) {
        console.error("SpeechRecognition error:", e);
        isListeningForVoice = false;
        setMicUI(false);
    }
}


// ─────────────────────────────────────────────
// Outdoor Suite: Auto-Torch, Fall Detection & Wayfinding
// ─────────────────────────────────────────────
let isTorchActive = false;
let autoTorchEnabled = true;
let isFallAlertActive = false;
let fallCountdownInterval = null;
let fallCountdownSeconds = 10;
let lastMotionTimestamp = 0;
let isNavigating = false;
let navigationTimer = null;
let currentNavigationRoute = null;

function checkAmbientLuminanceAndTorch() {
    if (!autoTorchEnabled || prefs.autoTorch === false || !cameraStream) return;
    try {
        const sample = captureCtx.getImageData(0, 0, 32, 24).data;
        let totalLum = 0;
        const count = sample.length / 4;
        for (let i = 0; i < sample.length; i += 4) {
            totalLum += 0.299 * sample[i] + 0.587 * sample[i + 1] + 0.114 * sample[i + 2];
        }
        const avgLum = totalLum / count;
        if (avgLum < 38 && !isTorchActive) {
            setTorchState(true);
            showIslandNotification("Night Torch Auto-On", "Flashlight active for low light", "fa-solid fa-lightbulb", false, 2800);
        } else if (avgLum > 75 && isTorchActive) {
            setTorchState(false);
            showIslandNotification("Night Torch Auto-Off", "Ambient lighting restored", "fa-solid fa-sun", false, 2400);
        }
    } catch (e) {}
}

async function setTorchState(enable) {
    if (!cameraStream) return false;
    const videoTrack = cameraStream.getVideoTracks()[0];
    if (!videoTrack) return false;
    try {
        const capabilities = videoTrack.getCapabilities ? videoTrack.getCapabilities() : {};
        if (capabilities.torch) {
            await videoTrack.applyConstraints({
                advanced: [{ torch: enable }]
            });
            isTorchActive = enable;
            const torchBtn = document.getElementById("torch-btn");
            if (torchBtn) {
                torchBtn.classList.toggle("torch-active", enable);
                torchBtn.setAttribute("aria-pressed", enable ? "true" : "false");
            }
            return true;
        }
    } catch (e) {
        console.warn("[AMCS] Torch constraint error:", e);
    }
    return false;
}

async function toggleTorch() {
    if (!cameraStream) {
        notify("Camera is off", "Enable the camera to use the flashlight.", "warning");
        return;
    }
    autoTorchEnabled = false;
    const target = !isTorchActive;
    const ok = await setTorchState(target);
    if (ok) {
        showIslandNotification(target ? "Flashlight On" : "Flashlight Off", "Manual torch toggle", "fa-solid fa-lightbulb", false, 2000);
    } else {
        notify("Flashlight not available", "This camera or browser does not support torch control.", "warning");
    }
}

function initFallDetection() {
    if (window.DeviceMotionEvent) {
        window.addEventListener("devicemotion", handleDeviceMotion, { passive: true });
        console.log("[AMCS] Fall detection accelerometer listener active.");
    }
}

function handleDeviceMotion(event) {
    if (isFallAlertActive) return;
    const acc = event.accelerationIncludingGravity || event.acceleration;
    if (!acc) return;

    const ax = acc.x || 0;
    const ay = acc.y || 0;
    const az = acc.z || 0;
    const totalAcc = Math.sqrt(ax * ax + ay * ay + az * az);

    if (totalAcc > 26.0) {
        const now = Date.now();
        if (now - lastMotionTimestamp > 3000) {
            lastMotionTimestamp = now;
            console.warn(`[AMCS] Sudden impact detected: ${totalAcc.toFixed(1)} m/s^2`);
            setTimeout(() => {
                triggerFallAlert();
            }, 600);
        }
    }
}

function triggerFallAlert() {
    if (isFallAlertActive) return;
    isFallAlertActive = true;
    fallCountdownSeconds = 10;

    const overlay = document.getElementById("fall-alert-overlay");
    const countNum = document.getElementById("fall-countdown-num");
    if (overlay) overlay.classList.remove("hidden");
    if (countNum) countNum.textContent = fallCountdownSeconds;

    playEarcon("hazard");
    triggerHapticFeedback("hazard");
    showIslandNotification("FALL DETECTED!", "Emergency alert in 10 seconds", "fa-solid fa-triangle-exclamation", true, 10000);

    const curLang = langSelect ? langSelect.value : "en";
    speakBrowserFallback("Fall detected! Are you okay? Tap screen or say 'I am fine' to cancel emergency alert.", curLang);

    if (fallCountdownInterval) clearInterval(fallCountdownInterval);
    fallCountdownInterval = setInterval(() => {
        fallCountdownSeconds--;
        if (countNum) countNum.textContent = fallCountdownSeconds;

        if (fallCountdownSeconds <= 0) {
            clearInterval(fallCountdownInterval);
            fallCountdownInterval = null;
            dispatchFallSOS();
        }
    }, 1000);
}

function cancelFallAlert(e) {
    if (e && e.stopPropagation) e.stopPropagation();
    if (!isFallAlertActive) return;

    if (fallCountdownInterval) {
        clearInterval(fallCountdownInterval);
        fallCountdownInterval = null;
    }
    isFallAlertActive = false;

    const overlay = document.getElementById("fall-alert-overlay");
    if (overlay) overlay.classList.add("hidden");

    playEarcon("start");
    showIslandNotification("Alert Cancelled", "Glad you are safe", "fa-solid fa-circle-check", false, 3000);

    const curLang = langSelect ? langSelect.value : "en";
    speakBrowserFallback("Emergency alert cancelled. Glad you are safe.", curLang);
}

async function dispatchFallSOS() {
    const overlay = document.getElementById("fall-alert-overlay");
    if (overlay) overlay.classList.add("hidden");
    isFallAlertActive = false;

    let batteryLvl = "unknown";
    try {
        if (navigator.getBattery) {
            const b = await navigator.getBattery();
            batteryLvl = Math.round(b.level * 100);
        }
    } catch (e) {}

    const curCoords = currentUserCoords || FALLBACK_COORDS;

    try {
        const resp = await fetch("/api/fall_detected", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                latitude: curCoords[0],
                longitude: curCoords[1],
                battery: batteryLvl
            })
        });
        const data = await resp.json();
        showIslandNotification("EMERGENCY DISPATCHED", "Caregiver alerted via GPS & WhatsApp", "fa-solid fa-triangle-exclamation", true, 6000);
        playEarcon("hazard");
    } catch (e) {
        console.error("[AMCS] Fall SOS dispatch error:", e);
    }
}

const NAV_POLL_MS = 2500;
const NAV_REMIND_MS = 30000;       // the same instruction is repeated after this long
let navAnnouncedKey = "";
let navAnnouncedAt = 0;
let navPollBusy = false;
let navSpeakingUntil = 0;          // a direction being spoken is not cut off by a routine update

function sayNavigation(text, lang, urgent = false) {
    if (!text) return;
    quietGuidanceFor(text, 2500);
    navSpeakingUntil = Date.now() + Math.min(20000, text.length * 75 + 800);
    speakText(text, lang, { interrupt: true, alert: urgent });
}

function navBanner(text, badge, relDeg) {
    const banner = document.getElementById("nav-turn-banner");
    const stepText = document.getElementById("nav-step-text");
    const distBadge = document.getElementById("nav-distance-badge");
    const arrow = document.querySelector("#nav-maneuver-icon i");
    if (banner) banner.classList.remove("hidden");
    if (stepText && text) stepText.textContent = text;
    if (distBadge && badge) distBadge.textContent = badge;
    // The arrow points the way to walk, relative to the way the phone faces (icon itself points 45deg)
    if (arrow) arrow.style.transform = typeof relDeg === "number" ? `rotate(${relDeg - 45}deg)` : "";
}

async function startNavigation(destinationQuery) {
    const curLang = langSelect ? langSelect.value : "en";
    destinationQuery = (destinationQuery || "").trim();
    if (!destinationQuery) { promptNavigation(); return; }
    startCompass();

    showIslandNotification("Finding you…", `Route to ${destinationQuery}`, "fa-solid fa-compass", false, 2500);
    const curCoords = await getUserPosition();
    if (!curCoords) {
        // Never route from a guessed position: it would send the user the wrong way
        const msg = "I cannot find your location. Turn on location for this site and try again.";
        sayNavigation(msg, "en");
        notify("Location needed", msg, "warning");
        return;
    }

    try {
        const resp = await fetch("/api/navigate/plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                destination: destinationQuery,
                origin_lat: curCoords[0],
                origin_lon: curCoords[1],
                heading: currentHeading(),
                lang: curLang
            })
        });
        const data = await resp.json();

        if (data.status === "success") {
            isNavigating = true;
            currentNavigationRoute = data;
            navAnnouncedKey = "";
            navAnnouncedAt = Date.now();

            navBanner(data.instruction_translated || data.initial_instruction,
                `${Math.round(data.total_distance_m)} m • ${data.destination}`, null);
            playEarcon("start");
            showIslandNotification("Route active", `${data.destination} • ${Math.round(data.total_distance_m)} m`, "fa-solid fa-location-arrow", false, 3500);
            sayNavigation(data.instruction_translated || data.initial_instruction, curLang);

            if (navigationTimer) clearInterval(navigationTimer);
            navigationTimer = setInterval(pollNavigationProgress, NAV_POLL_MS);
        } else {
            const msg = data.instruction_translated || data.message || `I could not find a route to ${destinationQuery}.`;
            sayNavigation(msg, curLang);
            notify("No route found", msg, "warning");
        }
    } catch (e) {
        console.error("[AMCS] Navigation error:", e);
        notify("Navigation failed", "Could not reach the routing service.", "error");
    }
}

async function pollNavigationProgress() {
    if (!isNavigating || navPollBusy || !currentUserCoords) return;
    const curLang = langSelect ? langSelect.value : "en";
    navPollBusy = true;
    try {
        const resp = await fetch("/api/navigate/update", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                lat: currentUserCoords[0],
                lon: currentUserCoords[1],
                accuracy: currentUserAccuracy,
                heading: currentHeading(),
                announced_key: navAnnouncedKey,
                lang: curLang
            })
        });
        const data = await resp.json();
        if (!isNavigating) return;

        if (data.status === "navigating") {
            const text = data.instruction_translated || data.instruction;
            navBanner(text, `${Math.round(data.dist_remaining_m)} m to go`, data.relative_deg);
            const isNew = data.announce_key !== navAnnouncedKey;
            const busy = !data.urgent && Date.now() < navSpeakingUntil;
            if (!busy && (isNew || Date.now() - navAnnouncedAt > NAV_REMIND_MS)) {
                navAnnouncedKey = data.announce_key;
                navAnnouncedAt = Date.now();
                if (data.urgent && prefs.haptics !== false && "vibrate" in navigator) {
                    // Turn now: two pulses for right, one long pulse for left
                    navigator.vibrate(data.maneuver && data.maneuver.includes("left") ? [400] : [120, 80, 120]);
                }
                sayNavigation(text, curLang, !!data.urgent);
            }
        } else if (data.status === "arrived") {
            cancelActiveNavigation(true);
            sayNavigation(data.instruction_translated || data.instruction, curLang);
            playEarcon("start");
            showIslandNotification("Arrived!", data.instruction, "fa-solid fa-flag-checkered", false, 4000);
        } else if (data.status === "no_active_route") {
            cancelActiveNavigation(true);
        }
    } catch (e) {
    } finally {
        navPollBusy = false;
    }
}

function cancelActiveNavigation(isSilent = false) {
    isNavigating = false;
    currentNavigationRoute = null;
    navAnnouncedKey = "";
    if (navigationTimer) {
        clearInterval(navigationTimer);
        navigationTimer = null;
    }
    releaseGPSIfUnused();

    const banner = document.getElementById("nav-turn-banner");
    if (banner) banner.classList.add("hidden");

    fetch("/api/navigate/cancel", { method: "POST" }).catch(() => {});

    if (!isSilent) {
        const curLang = langSelect ? langSelect.value : "en";
        speakBrowserFallback("Navigation cancelled.", curLang);
        showIslandNotification("Route Cancelled", "Wayfinding paused", "fa-solid fa-xmark", false, 2000);
    }
}

function requireCamera() {
    if (videoEl && videoEl.readyState >= 2) return true;
    notify("Camera not ready", "Enable the camera first, then try again.", "warning");
    if (!cameraStream) startCamera();
    return false;
}

let sceneQABusy = false;

async function askSceneQA(questionText) {
    questionText = (questionText || "").trim();
    if (!questionText || sceneQABusy) return;
    if (!requireCamera()) {
        speakText("The camera is not on yet. Enable the camera, then ask again.", "en");
        return;
    }
    const curLang = langSelect ? langSelect.value : "en";
    const base64Image = captureFrame();
    sceneQABusy = true;
    guidanceQuietUntil = Date.now() + 10000;          // no scene chatter while the answer is worked out

    showIslandNotification("Looking…", `"${questionText}"`, "fa-solid fa-brain", false, 3000);
    playEarcon("lock");

    try {
        const resp = await fetch("/api/scene_qa", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                question: questionText,
                image: base64Image,
                lang: curLang,
                camera_facing: cameraFacesUser ? "user" : "environment",
                quality: DETECTION_QUALITY,
                tts: !AMCS.voice.useDeviceVoice(curLang)
            })
        });
        const data = await resp.json();

        if (data.status === "success") {
            const ansText = data.answer_translated || data.answer_en;
            const mainTextEl = document.getElementById("speech-translated-text");
            const subTextEl = document.getElementById("speech-original-text");
            if (mainTextEl) mainTextEl.textContent = ansText;
            if (subTextEl) subTextEl.textContent = `You asked: ${questionText}`;

            guidanceQuietUntil = 0;
            quietGuidanceFor(ansText, 2500);
            speakText(ansText, curLang, { audioUrl: data.audio_url, interrupt: true });
            showIslandNotification("Answer", ansText.length > 48 ? ansText.substring(0, 46) + "…" : ansText, "fa-solid fa-comment-dots", false, 4000);
        } else {
            guidanceQuietUntil = 0;
            speakText("Sorry, I could not answer that. Please ask again.", "en");
        }
    } catch (e) {
        console.error("[AMCS] Scene Q&A error:", e);
        guidanceQuietUntil = 0;
        speakText("I could not reach the AMCS server. Check the connection.", "en");
    } finally {
        sceneQABusy = false;
    }
}

// ─────────────────────────────────────────────
// Saved places ("location setter") & where am I
// ─────────────────────────────────────────────
async function saveCurrentPlace(name) {
    const curLang = langSelect ? langSelect.value : "en";
    name = (name || "").trim();
    if (!name) {
        name = await AMCS.prompt({
            title: "Save this place",
            message: "Give this spot a name. Later say \"take me to\" and the name.",
            icon: "fa-location-dot",
            placeholder: "e.g. home, office, bus stop",
            label: "Place name",
            suggestions: ["home", "office", "bus stop", "college gate", "market"],
            confirmText: "Save place"
        });
        name = (name || "").trim();
        if (!name) return;
    }
    showIslandNotification("Saving place…", name, "fa-solid fa-location-dot", false, 2500);
    const pos = await getUserPosition();
    if (!pos) {
        const msg = "I cannot find your location. Turn on location for this site and try again.";
        speakText(msg, "en");
        notify("Location needed", msg, "warning");
        return;
    }
    try {
        const resp = await fetch("/api/places", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ name, lat: pos[0], lon: pos[1] })
        });
        const data = await resp.json();
        if (data.status !== "success") throw new Error(data.error || "not saved");
        let msg = data.message;
        if (currentUserAccuracy && currentUserAccuracy > 30) msg += ` Location is only accurate to about ${Math.round(currentUserAccuracy)} meters here.`;
        speakText(msg, "en");
        notify("Place saved", msg, "success");
    } catch (e) {
        speakText("Sorry, the place could not be saved.", "en");
        notify("Not saved", String(e.message || e), "error");
    } finally {
        releaseGPSIfUnused();
    }
}

async function sayWhereAmI() {
    const curLang = langSelect ? langSelect.value : "en";
    const pos = await getUserPosition();
    if (!pos) {
        speakText("I cannot find your location. Turn on location for this site and try again.", "en");
        return;
    }
    try {
        const resp = await fetch("/api/where_am_i", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ lat: pos[0], lon: pos[1], accuracy: currentUserAccuracy, heading: currentHeading(), lang: curLang })
        });
        const data = await resp.json();
        speakText(data.answer_translated || data.answer_en, curLang, { interrupt: true });
    } catch (e) {
        speakText("I could not reach the AMCS server.", "en");
    } finally {
        releaseGPSIfUnused();
    }
}


// ─────────────────────────────────────────────
// Item Finder & Acoustic Geiger Counter Radar
// ─────────────────────────────────────────────
let isItemRadarActive = false;
let radarTargetItem = "";
let radarPollInterval = null;
let radarPingInterval = null;
let currentRadarPitch = 440;
let currentRadarPan = 0.0;
let currentRadarRate = 1.0;

async function promptFindItem() {
    if (isItemRadarActive) { stopItemFinder(); return; }
    const item = await AMCS.prompt({
        title: "Find my item",
        message: "Sweep the camera slowly. Beeps speed up and pan toward the item.",
        icon: "fa-satellite-dish",
        placeholder: "e.g. keys, phone, bottle",
        label: "Item to find",
        suggestions: ["phone", "keys", "bottle", "backpack", "spectacles", "book", "chair"],
        confirmText: "Start radar"
    });
    if (item && item.trim()) {
        if (!cameraStream) await startCamera();
        startItemFinder(item.trim());
    }
}

async function promptSceneQuestion() {
    const q = await AMCS.prompt({
        title: "Ask about your surroundings",
        message: "AMCS looks through the camera and answers out loud.",
        icon: "fa-comment-dots",
        placeholder: "What is in front of me?",
        label: "Your question",
        suggestions: ["What is in front of me?", "What is this?", "Who is in front of me?", "Is the path clear?", "What does the sign say?"],
        confirmText: "Ask"
    });
    if (q) askSceneQA(q);
}

async function promptNavigation() {
    let suggestions = [];
    try {
        const [placesRes, landmarksRes] = await Promise.all([fetch("/api/places"), fetch("/api/navigate/landmarks")]);
        const saved = ((await placesRes.json()).places || []).map(p => p.name);
        const campus = ((await landmarksRes.json()).landmarks || []).map(l => l.name);
        suggestions = saved.concat(campus).slice(0, 10);
    } catch (e) {}
    startCompass();      // a tap: the moment an iPhone may ask for compass access
    const dest = await AMCS.prompt({
        title: "Walking directions",
        message: "Say or type a saved place, a campus building, or any address nearby. Hold the phone upright, camera facing forward, so turns are told from the way you face.",
        icon: "fa-route",
        placeholder: "e.g. home, Library, Jaipur Junction",
        label: "Destination",
        suggestions,
        confirmText: "Start route"
    });
    if (dest) startNavigation(dest);
}

function describeNow() {
    if (!requireCamera()) return;
    lastSpeechTimestamp = 0;
    processCurrentFrame(true);
}

function startItemFinder(targetItem) {
    if (!targetItem) return;
    radarTargetItem = targetItem.trim().toLowerCase();
    isItemRadarActive = true;

    const findBtn = document.getElementById("find-item-btn");
    const banner = document.getElementById("radar-live-banner");
    const titleEl = document.getElementById("radar-title");
    const subEl = document.getElementById("radar-sub");

    if (findBtn) findBtn.classList.add("radar-active");
    if (banner) banner.classList.remove("hidden");
    if (titleEl) titleEl.textContent = `RADAR: ${radarTargetItem.toUpperCase()}`;
    if (subEl) subEl.textContent = "Scanning... Sweep camera slowly";

    playEarcon("start");
    showIslandNotification(`Finding ${radarTargetItem}`, "Acoustic audio radar online", "fa-solid fa-satellite-dish", false, 3000);

    const curLang = langSelect ? langSelect.value : "en";
    speakBrowserFallback(`Acoustic radar active for ${radarTargetItem}. Sweep your camera slowly around the room.`, curLang);

    // Initial frame scan
    pollItemRadar();

    // Polling loop for vision tracking
    if (radarPollInterval) clearInterval(radarPollInterval);
    radarPollInterval = setInterval(pollItemRadar, 1400);

    // Schedule acoustic Geiger ping loop
    scheduleNextRadarPing();
}

function scheduleNextRadarPing() {
    if (!isItemRadarActive) return;
    playRadarPing(currentRadarPitch, currentRadarPan);

    const dot = document.getElementById("radar-ping-dot");
    if (dot) {
        dot.classList.add("ping-pulse");
        setTimeout(() => dot.classList.remove("ping-pulse"), 100);
    }

    const intervalMs = Math.max(90, Math.min(1800, Math.round(1000 / currentRadarRate)));
    radarPingInterval = setTimeout(scheduleNextRadarPing, intervalMs);
}

function playRadarPing(pitchHz, panVal) {
    if (isMuted) return;
    try {
        const actx = getAudioCtx();
        if (!actx) return;
        const osc = actx.createOscillator();
        const gain = actx.createGain();

        if (actx.createStereoPanner) {
            const panner = actx.createStereoPanner();
            panner.pan.setValueAtTime(Math.max(-0.9, Math.min(0.9, panVal)), actx.currentTime);
            osc.connect(panner);
            panner.connect(gain);
        } else {
            osc.connect(gain);
        }

        osc.type = "sine";
        osc.frequency.setValueAtTime(pitchHz, actx.currentTime);
        gain.gain.setValueAtTime(0.08, actx.currentTime);
        gain.gain.exponentialRampToValueAtTime(0.001, actx.currentTime + 0.07);
        gain.connect(actx.destination);

        osc.start();
        osc.stop(actx.currentTime + 0.08);
    } catch (e) {}
}

async function pollItemRadar() {
    if (!isItemRadarActive || !videoEl || videoEl.readyState < 2) return;
    const curLang = langSelect ? langSelect.value : "en";

    const base64Image = captureFrame();

    try {
        const resp = await fetch("/api/find_item", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                target: radarTargetItem,
                image: base64Image,
                lang: curLang,
                camera_facing: cameraFacesUser ? "user" : "environment",
                quality: DETECTION_QUALITY
            })
        });
        const data = await resp.json();

        if (data.status === "success" && data.radar) {
            const r = data.radar;
            currentRadarRate = r.ping_rate_hz || 1.0;
            currentRadarPitch = r.ping_pitch_hz || 440;
            currentRadarPan = r.spatial_pan || 0.0;

            const subEl = document.getElementById("radar-sub");
            if (subEl) subEl.textContent = r.guidance_translated || r.guidance;

            if (r.found) {
                if (r.is_centered) {
                    showIslandNotification("TARGET LOCKED!", `${r.target.toUpperCase()} • ${r.distance_m}m ahead`, "fa-solid fa-bullseye", true, 2000);
                    triggerHapticFeedback("caution");
                } else {
                    showIslandNotification(`Spotted: ${r.target}`, `${r.spatial.toUpperCase()} • ${r.distance_m}m`, "fa-solid fa-satellite-dish", false, 1500);
                }

                if (r.distance_m <= 0.5 && r.is_centered) {
                    speakBrowserFallback(`Target reached! ${r.target} is right in front of your hand.`, curLang);
                }
            }
        }
    } catch (e) {
        console.error("[AMCS] Item radar poll error:", e);
    }
}

function stopItemFinder() {
    isItemRadarActive = false;
    radarTargetItem = "";
    if (radarPollInterval) {
        clearInterval(radarPollInterval);
        radarPollInterval = null;
    }
    if (radarPingInterval) {
        clearTimeout(radarPingInterval);
        radarPingInterval = null;
    }

    const findBtn = document.getElementById("find-item-btn");
    const banner = document.getElementById("radar-live-banner");
    if (findBtn) findBtn.classList.remove("radar-active");
    if (banner) banner.classList.add("hidden");

    showIslandNotification("Radar Stopped", "Item search ended", "fa-solid fa-xmark", false, 2000);
    const curLang = langSelect ? langSelect.value : "en";
    speakBrowserFallback("Item finder stopped.", curLang);
}


async function scanDoorGuidance() {
    if (!requireCamera()) return;
    const curLang = langSelect ? langSelect.value : "en";
    const base64Image = captureFrame();
    showIslandNotification("Scanning Door...", "Analyzing doorway & handle position", "fa-solid fa-door-open", false, 2500);

    try {
        const resp = await fetch("/api/scan_door", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64Image, lang: curLang })
        });
        const data = await resp.json();
        if (data.status === "success") {
            const guidance = data.guidance_translated || data.guidance_en;
            lastSpokenText = guidance;
            lastSpokenAudioUrl = "";
            const mainTextEl = document.getElementById("speech-translated-text");
            if (mainTextEl) mainTextEl.textContent = guidance;
            speakBrowserFallback(guidance, curLang);
            showIslandNotification("Door Scanned", guidance, "fa-solid fa-door-closed", false, 3500);
        }
    } catch (e) {
        notify("Door scan failed", "Could not reach the server.", "error");
    }
}


// ─────────────────────────────────────────────
// Multimodal Audio AI: Acoustic Siren & Horn Monitor
// ─────────────────────────────────────────────
let acousticAudioCtx = null;
let acousticAnalyser = null;
let acousticMicStream = null;
let acousticMonitorTimer = null;
let isAcousticListening = false;

async function initAcousticHazardMonitor() {
    if (isAcousticListening) return;
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;

        acousticMicStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false });
        acousticAudioCtx = new AudioCtx();
        const source = acousticAudioCtx.createMediaStreamSource(acousticMicStream);
        acousticAnalyser = acousticAudioCtx.createAnalyser();
        acousticAnalyser.fftSize = 256;
        source.connect(acousticAnalyser);

        isAcousticListening = true;
        console.log("[AMCS] Multimodal Acoustic Horn & Siren monitor online.");

        if (acousticMonitorTimer) clearInterval(acousticMonitorTimer);
        acousticMonitorTimer = setInterval(pollAcousticSpectrum, 2400);
    } catch (e) {
        console.warn("[AMCS] Microphone access for acoustic monitoring deferred:", e);
    }
}

async function pollAcousticSpectrum() {
    if (!acousticAnalyser || !isAcousticListening) return;
    const freqData = new Uint8Array(acousticAnalyser.frequencyBinCount);
    acousticAnalyser.getByteFrequencyData(freqData);

    // Quick client-side threshold check before server roundtrip
    let maxVal = 0;
    for (let i = 0; i < freqData.length; i++) {
        if (freqData[i] > maxVal) maxVal = freqData[i];
    }
    if (maxVal < 110) return; // Ignore normal conversational quietness

    const curLang = langSelect ? langSelect.value : "en";
    const binsList = Array.from(freqData).map(v => parseFloat((v / 255.0).toFixed(3)));

    try {
        const resp = await fetch("/api/audio_hazard_detect", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frequency_bins: binsList, lang: curLang })
        });
        const data = await resp.json();

        if (data.status === "success" && data.result && data.result.detected) {
            const res = data.result;
            const adv = res.advisory_translated || res.advisory_en;
            playEarcon("hazard");
            triggerHapticFeedback("hazard");
            showIslandNotification("ACOUSTIC HAZARD!", res.hazard_type.replace("_", " ").toUpperCase(), "fa-solid fa-bullhorn", true, 4500);
            speakBrowserFallback(adv, curLang);
        }
    } catch (e) {}
}


async function analyzeGarmentClothing() {
    if (!requireCamera()) return;
    const curLang = langSelect ? langSelect.value : "en";
    const base64Image = captureFrame();
    showIslandNotification("Analyzing Clothes...", "Inspecting color shade & fabric pattern", "fa-solid fa-shirt", false, 2500);

    try {
        const resp = await fetch("/api/match_clothing", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image: base64Image, lang: curLang })
        });
        const data = await resp.json();
        if (data.status === "success" && data.analysis) {
            const an = data.analysis;
            const text = an.summary_translated || an.summary_en;
            lastSpokenText = text;
            lastSpokenAudioUrl = "";
            const mainTextEl = document.getElementById("speech-translated-text");
            if (mainTextEl) mainTextEl.textContent = text;
            speakBrowserFallback(text, curLang);
            showIslandNotification(`${an.primary_color}`, `${an.pattern} fabric`, "fa-solid fa-palette", false, 4000);
        }
    } catch (e) {}
}


// Tele-Assistance Frame Streamer & Intercom Listener
const INTERCOM_CHECK_MS = 1500;
let intercomCheckedAt = 0;
let intercomCallActive = false;

async function syncTeleAssistIntercom(base64Image) {
    try {
        // Check for caregiver messages; the reply also says whether a call is live
        const resp = await fetch("/api/teleassist/intercom", { cache: "no-store" });
        const data = await resp.json();
        intercomCallActive = !!data.call_active;

        // Only share camera frames while the caregiver has a call open
        if (data.call_active && base64Image) {
            fetch("/api/teleassist/frame", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: base64Image })
            }).catch(() => {});
        }
        if (data.status === "success" && data.messages && data.messages.length > 0) {
            const curLang = langSelect ? langSelect.value : "en";
            for (const m of data.messages) {
                playEarcon("lock");
                showIslandNotification("Caregiver Intercom", m.message, "fa-solid fa-headset", true, 5000);
                speakBrowserFallback(`Caregiver guidance: ${m.message}`, curLang);
            }
        }
    } catch (e) {}
}

/** "save this place as home" -> "home"; null when the command is not about saving a place. */
function savedPlaceName(cmd) {
    const m = cmd.match(/(?:save|remember|mark|set)\s+(?:this|the|my)?\s*(?:place|location|spot|position)(?:\s+(?:as|called|named))?\s*(.*)$/)
        || cmd.match(/^(?:save|remember)\s+(?:as\s+)?(.+?)\s+(?:here|location)$/)
        || cmd.match(/(?:is jagah ko|yeh jagah|ye jagah)\s+(.+?)\s+(?:naam se\s+)?(?:save|yaad)/);
    return m ? (m[1] || "").replace(/^(as|called|named)\s+/, "").replace(/^(my|the)\s+/, "").trim() : null;
}

/** Sounds like a question about the surroundings (English, Hinglish or another script). */
function looksLikeQuestion(cmd) {
    return /^(what|what's|whats|who|who's|where|which|how|is|are|can|do|does|tell me|kya|kaun|kahan|kitne|kitna)\b/.test(cmd)
        || /\b(kya hai|kaun hai|dikh raha|in front|ahead|saamne|samne|aage)\b/.test(cmd)
        || /[^\x00-\x7f]/.test(cmd) || cmd.endsWith("?");
}

function handleVoiceCommand(cmd) {
    const curLang = langSelect ? langSelect.value : "en";
    cmd = cmd.replace(/[.!]+$/, "").trim();

    // Places and walking directions first: they contain words like "where" and "place"
    const placeName = savedPlaceName(cmd);
    if (placeName !== null) {
        saveCurrentPlace(placeName);
        return;
    }
    if (/\b(where am i|my location|current location|main kahan hoon|mai kahan hu|main kaha hu)\b/.test(cmd)) {
        sayWhereAmI();
        return;
    }
    if (cmd.includes("stop navigation") || cmd.includes("cancel route") || cmd.includes("cancel navigation") || cmd.includes("stop directions")) {
        cancelActiveNavigation();
        return;
    }
    const navMatch = cmd.match(/(?:navigate to|take me to|directions to|guide me to|how do i get to|go to|le chalo|rasta dikhao)\s+(.+)$/)
        || cmd.match(/^(.+?)\s+(?:ka rasta|le chalo|ka rasta dikhao|chalna hai)$/);
    if (navMatch) {
        startNavigation(navMatch[1].replace(/^(the|my)\s+/, ""));
        return;
    }
    if (cmd.includes("save place") || cmd.includes("save location")) {
        saveCurrentPlace("");
        return;
    }
    if (isFallAlertActive && /\b(fine|ok|okay|safe|theek)\b/.test(cmd)) {
        cancelFallAlert();
        return;
    }

    if (cmd.includes("describe") || cmd.includes("what do you see") || cmd.includes("look around")) {
        describeNow();
        return;
    }

    // "Speak Hindi", "switch to Tamil", or just "English"
    const VOICE_LANGS = { english: "en", hindi: "hi", spanish: "es", french: "fr", german: "de", chinese: "zh-CN", japanese: "ja", arabic: "ar", russian: "ru", bengali: "bn", tamil: "ta", telugu: "te", marathi: "mr", gujarati: "gu" };
    const spokenLang = Object.keys(VOICE_LANGS).find(name => cmd.split(/\s+/).includes(name));
    if (spokenLang && langSelect && cmd.split(/\s+/).length <= 4) {
        langSelect.value = VOICE_LANGS[spokenLang];
        langSelect.dispatchEvent(new Event("change"));
        speakBrowserFallback(`Language set to ${spokenLang}.`, VOICE_LANGS[spokenLang]);
        return;
    }

    if (cmd.includes("cloth") || cmd.includes("shirt") || cmd.includes("pant") || cmd.includes("wardrobe") || cmd.includes("kapde") || cmd.includes("what color")) {
        analyzeGarmentClothing();
        return;
    }

    if (cmd.includes("find the door") || cmd.includes("find door") || cmd.includes("door handle") || cmd.includes("darwaza dhoondho") || cmd === "door") {
        scanDoorGuidance();
        return;
    }

    if (cmd.includes("find my") || cmd.includes("where is my") || cmd.includes("search for") || cmd.includes("dhoondho")) {
        const target = cmd.replace("find my", "").replace("where is my", "").replace("search for", "").replace("dhoondho", "").trim();
        startItemFinder(target || "bottle");
        return;
    }
    if (cmd.includes("stop radar") || cmd.includes("stop finder") || cmd.includes("stop searching")) {
        stopItemFinder();
        return;
    }

    if (cmd.includes("torch on") || cmd.includes("light on") || cmd.includes("flashlight on")) {
        setTorchState(true);
        speakBrowserFallback("Flashlight turned on.", curLang);
        return;
    }
    if (cmd.includes("torch off") || cmd.includes("light off") || cmd.includes("flashlight off")) {
        setTorchState(false);
        speakBrowserFallback("Flashlight turned off.", curLang);
        return;
    }
    if (cmd.startsWith("ask ")) {
        askSceneQA(cmd.slice(4));
        return;
    }
    if (cmd.includes("emergency") || cmd.includes("sos") || cmd.includes("bachao")) {
        triggerSOS();
        return;
    }
    // Any question about the surroundings: "what is in front of me", "who is there", "saamne kya hai"
    if (looksLikeQuestion(cmd)) {
        askSceneQA(cmd);
        return;
    }

    if (cmd.includes("medicine") || cmd.includes("tablet") || cmd.includes("dawa") || cmd.includes("pill")) {
        setMode("medicine");
        speakBrowserFallback("Switched to medicine and expiry date reader mode.", curLang);
    } else if (cmd.includes("currency") || cmd.includes("note") || cmd.includes("rupee") || cmd.includes("paisa") || cmd.includes("money")) {
        setMode("currency");
        speakBrowserFallback("Switched to Indian currency identification mode.", curLang);
    } else if (cmd.includes("sign") || cmd.includes("text") || cmd.includes("board") || cmd.includes("read")) {
        setMode("ocr");
        speakBrowserFallback("Switched to signboard reading mode.", curLang);
    } else if (cmd.includes("obstacle") || cmd.includes("walk") || cmd.includes("vision") || cmd.includes("navigation")) {
        setMode("vision");
        speakBrowserFallback("Switched to obstacle navigation mode.", curLang);
    } else if (cmd.includes("hybrid")) {
        setMode("hybrid");
        speakBrowserFallback("Switched to hybrid mode.", curLang);
    } else if (cmd.includes("start") || cmd.includes("begin") || cmd.includes("resume") || cmd.includes("chalo")) {
        if (!isAutoRunning) toggleAutoDetection();
        speakBrowserFallback("Assistant started.", curLang);
    } else if (cmd.includes("stop") || cmd.includes("pause") || cmd.includes("roko")) {
        if (isAutoRunning) toggleAutoDetection();
        speakBrowserFallback("Assistant paused.", curLang);
    } else if (cmd.includes("mute") || cmd.includes("silent") || cmd.includes("quiet")) {
        toggleMute();
    } else if (cmd.includes("help")) {
        triggerSOS();
    } else if (cmd.includes("repeat") || cmd.includes("again") || cmd.includes("fir se")) {
        replayAudio();
    } else {
        // Not a command: treat it as a question about what the camera sees
        askSceneQA(cmd);
    }
}

// ─────────────────────────────────────────────
// 11. Language Switch Listener
// ─────────────────────────────────────────────
if (langSelect) {
    // A language chosen in Settings overrides the server default
    if (prefs.lang && langSelect.querySelector(`option[value="${prefs.lang}"]`)) {
        langSelect.value = prefs.lang;
    }
    langSelect.addEventListener("change", () => {
        if (window.AMCS) window.AMCS.setPrefs({ lang: langSelect.value });
        if (isAutoRunning) processCurrentFrame(true);
    });
}

// Keyboard shortcuts (ignored while typing, on focused controls, or when a dialog is open)
document.addEventListener("keydown", (e) => {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    const t = e.target;
    if (t && (t.isContentEditable || ["INPUT", "TEXTAREA", "SELECT", "BUTTON", "A"].includes(t.tagName))) return;
    if (document.querySelector("dialog[open]")) return;
    const k = e.key.toLowerCase();
    if (k === " ") { e.preventDefault(); toggleAutoDetection(); }
    else if (k === "d") describeNow();
    else if (k === "r") replayAudio();
    else if (k === "v") toggleVoiceRecognition();
    else if (k === "m") toggleMute();
});

document.addEventListener("visibilitychange", () => {
    // Wake locks are released when the tab is hidden; re-acquire on return
    if (document.visibilityState === "visible" && isAutoRunning) acquireWakeLock();
});

// Double-tap anywhere on the camera view to ask a question by voice (easy to find without looking)
function initDoubleTapToAsk() {
    const stage = document.getElementById("video-wrapper");
    if (!stage) return;
    let lastTap = 0, lastX = 0, lastY = 0;
    stage.addEventListener("pointerup", e => {
        if (e.target.closest("button, a, input, select, textarea, label, .nav-turn-banner, .radar-live-banner")) return;
        const now = Date.now();
        if (now - lastTap < 400 && Math.hypot(e.clientX - lastX, e.clientY - lastY) < 40) {
            lastTap = 0;
            e.preventDefault();
            toggleVoiceRecognition();
            return;
        }
        lastTap = now; lastX = e.clientX; lastY = e.clientY;
    });
}

// Start camera on load and initialize 60 FPS animation loops
window.addEventListener("DOMContentLoaded", () => {
    initDoubleTapToAsk();
    startCamera();
    initGPSTracker();
    initFallDetection();
    if (prefs.acousticMonitor !== false) initAcousticHazardMonitor();
    requestAnimationFrame(renderARHUDLoop);
    requestAnimationFrame(renderFluidAudioOrb);
    updateIslandTelemetry();
});
