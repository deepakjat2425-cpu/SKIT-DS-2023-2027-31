/**
 * AMCS shared UI layer: theme, preferences, toasts, dialogs and global SOS polling.
 * Loaded on every page before page-specific scripts.
 * Automatic Multilanguage Communication System - SKIT Jaipur
 */
(function () {
    "use strict";

    // ─────────────────────────────────────────────
    // Safe storage (private windows can throw)
    // ─────────────────────────────────────────────
    const store = {
        get(key, fallback = null) {
            try {
                const v = localStorage.getItem(key);
                return v === null ? fallback : v;
            } catch (e) { return fallback; }
        },
        set(key, value) {
            try { localStorage.setItem(key, value); } catch (e) { /* ignore */ }
        }
    };

    // ─────────────────────────────────────────────
    // User preferences (client-side, per device)
    // ─────────────────────────────────────────────
    const PREF_KEY = "amcs_prefs";
    // Version 2 added "auto" quality. Earlier saves stored "balanced" even when it was never chosen.
    const PREFS_VERSION = 2;
    // Phones and small tablets: touch screen, small display. They get lighter visuals and the
    // fast detection profile unless the user picks something else in Settings.
    const IS_PHONE = (() => {
        try {
            const coarse = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
            const small = Math.min(screen.width, screen.height) <= 820;
            return /Android|iPhone|iPad|iPod|Mobi/i.test(navigator.userAgent) || (coarse && small);
        } catch (e) { return false; }
    })();
    const DEFAULT_PREFS = {
        v: PREFS_VERSION,
        lang: null,             // null = use server default
        speechRate: 1.0,        // Web Speech rate
        voiceEngine: "auto",    // auto = natural device voice when available, else server voice
        voiceName: "",          // "" = pick the best voice automatically
        speechIntervalSec: 7.5, // minimum gap between routine announcements
        haptics: true,
        earcons: true,
        autoTorch: true,
        acousticMonitor: true,
        gps: true,
        quality: "auto",        // detection quality: auto | fast | balanced | high
        liteMode: "auto"        // lighter visuals: auto (phones) | on | off
    };

    function getPrefs() {
        try {
            const raw = store.get(PREF_KEY);
            const saved = raw ? JSON.parse(raw) : {};
            if (!(saved.v >= PREFS_VERSION) && saved.quality === "balanced") delete saved.quality;
            return Object.assign({}, DEFAULT_PREFS, saved, { v: PREFS_VERSION });
        } catch (e) {
            return Object.assign({}, DEFAULT_PREFS);
        }
    }
    function setPrefs(patch) {
        const next = Object.assign(getPrefs(), patch || {});
        store.set(PREF_KEY, JSON.stringify(next));
        return next;
    }
    // "auto" quality: fast on phones (smaller uploads, quicker answers), balanced elsewhere
    function resolveQuality(p) {
        const q = (p || getPrefs()).quality;
        if (["fast", "mobile", "balanced", "high"].includes(q)) return q;
        // Phones: the light model at 640 px finds more than the full model at 512 px, in less time
        return IS_PHONE ? "mobile" : "balanced";
    }
    function liteActive(p) {
        const mode = (p || getPrefs()).liteMode;
        return mode === "on" || (mode !== "off" && IS_PHONE);
    }
    function applyLite(p) {
        document.body.classList.toggle("lite", liteActive(p));
    }

    function resetPrefs() {
        store.set(PREF_KEY, JSON.stringify(DEFAULT_PREFS));
        return Object.assign({}, DEFAULT_PREFS);
    }

    // ─────────────────────────────────────────────
    // HTML escaping for any server/user text placed in innerHTML
    // ─────────────────────────────────────────────
    function escapeHtml(value) {
        return String(value === undefined || value === null ? "" : value)
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;")
            .replace(/"/g, "&quot;")
            .replace(/'/g, "&#39;");
    }

    // ─────────────────────────────────────────────
    // Theme (system / light / dark) + high contrast
    // ─────────────────────────────────────────────
    const THEME_ICONS = { system: "fa-circle-half-stroke", light: "fa-sun", dark: "fa-moon" };
    const THEME_ORDER = ["system", "light", "dark"];

    function applyTheme(mode) {
        const root = document.documentElement;
        let resolved = mode;
        if (mode === "system") {
            resolved = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark";
        }
        root.setAttribute("data-theme", resolved);
        root.setAttribute("data-theme-mode", mode);
        const btn = document.getElementById("theme-toggle");
        if (btn) {
            const icon = btn.querySelector("i");
            if (icon) icon.className = "fa-solid " + THEME_ICONS[mode];
            const label = "Theme: " + mode.charAt(0).toUpperCase() + mode.slice(1);
            btn.setAttribute("title", label);
            btn.setAttribute("aria-label", label + ". Activate to change.");
        }
        const meta = document.querySelector('meta[name="theme-color"]');
        if (meta) meta.setAttribute("content", resolved === "light" ? "#f3f4f8" : "#06070b");
    }
    function cycleTheme() {
        const current = store.get("amcs_theme", "system");
        const next = THEME_ORDER[(THEME_ORDER.indexOf(current) + 1) % THEME_ORDER.length];
        store.set("amcs_theme", next);
        applyTheme(next);
        toast({ title: "Theme changed", message: next === "system" ? "Following your device setting" : next.charAt(0).toUpperCase() + next.slice(1) + " theme", type: "info", duration: 1800 });
    }
    function applyContrast(on) {
        document.body.classList.toggle("high-contrast", on);
        const btn = document.getElementById("contrast-toggle");
        if (btn) btn.setAttribute("aria-pressed", on ? "true" : "false");
    }

    // ─────────────────────────────────────────────
    // Toasts
    // ─────────────────────────────────────────────
    const TOAST_ICONS = { info: "fa-circle-info", success: "fa-circle-check", warning: "fa-triangle-exclamation", error: "fa-circle-xmark" };

    function toast(opts) {
        const o = typeof opts === "string" ? { message: opts } : (opts || {});
        const region = document.getElementById("toast-region");
        if (!region) return;
        const type = o.type || "info";
        const el = document.createElement("div");
        el.className = "toast " + type;
        el.setAttribute("role", type === "error" || type === "warning" ? "alert" : "status");
        el.innerHTML =
            '<div class="toast-icon"><i class="fa-solid ' + TOAST_ICONS[type] + '" aria-hidden="true"></i></div>' +
            '<div class="toast-body">' +
                (o.title ? '<div class="toast-title">' + escapeHtml(o.title) + "</div>" : "") +
                (o.message ? '<div class="toast-msg">' + escapeHtml(o.message) + "</div>" : "") +
            "</div>" +
            '<button class="toast-close" type="button" aria-label="Dismiss notification"><i class="fa-solid fa-xmark"></i></button>';
        const remove = () => {
            if (!el.isConnected) return;
            el.classList.add("leaving");
            setTimeout(() => el.remove(), 220);
        };
        el.querySelector(".toast-close").addEventListener("click", remove);
        region.appendChild(el);
        while (region.children.length > 4) region.firstElementChild.remove();
        setTimeout(remove, o.duration || 4200);
    }

    // ─────────────────────────────────────────────
    // Accessible modal dialogs (replace alert / confirm / prompt)
    // ─────────────────────────────────────────────
    function openDialog({ title, message = "", icon = "fa-circle-question", danger = false, input = null, suggestions = [], confirmText = "OK", cancelText = "Cancel" }) {
        return new Promise(resolve => {
            const dlg = document.getElementById("app-dialog");
            if (!dlg || typeof dlg.showModal !== "function") {
                // Very old browsers: fall back to native dialogs
                if (input) return resolve(window.prompt(title + (message ? "\n" + message : ""), input.value || ""));
                return resolve(window.confirm(title + (message ? "\n" + message : "")));
            }
            dlg.classList.toggle("danger", !!danger);
            dlg.querySelector(".dialog-icon i").className = "fa-solid " + icon;
            dlg.querySelector(".dialog-title").textContent = title;
            dlg.querySelector(".dialog-message").textContent = message;

            const inputWrap = dlg.querySelector(".dialog-input-wrap");
            const inputEl = dlg.querySelector(".dialog-input");
            const sugWrap = dlg.querySelector(".dialog-suggestions");
            inputWrap.hidden = !input;
            sugWrap.innerHTML = "";
            if (input) {
                inputEl.value = input.value || "";
                inputEl.placeholder = input.placeholder || "";
                dlg.querySelector(".dialog-input-label").textContent = input.label || title;
                suggestions.forEach(s => {
                    const b = document.createElement("button");
                    b.type = "button";
                    b.className = "btn btn-xs btn-outline";
                    b.textContent = s;
                    b.addEventListener("click", () => { inputEl.value = s; inputEl.focus(); });
                    sugWrap.appendChild(b);
                });
            }
            sugWrap.hidden = !suggestions.length;

            const okBtn = dlg.querySelector(".dialog-ok");
            const cancelBtn = dlg.querySelector(".dialog-cancel");
            okBtn.textContent = confirmText;
            okBtn.className = "btn dialog-ok " + (danger ? "btn-danger" : "btn-primary");
            cancelBtn.textContent = cancelText;

            const finish = (result) => {
                dlg.removeEventListener("close", onClose);
                okBtn.removeEventListener("click", onOk);
                cancelBtn.removeEventListener("click", onCancel);
                if (dlg.open) dlg.close();
                resolve(result);
            };
            const onOk = (e) => {
                e.preventDefault();
                if (input) {
                    const v = inputEl.value.trim();
                    if (!v) { inputEl.focus(); return; }
                    finish(v);
                } else {
                    finish(true);
                }
            };
            const onCancel = (e) => { e.preventDefault(); finish(input ? null : false); };
            const onClose = () => finish(input ? null : false);

            okBtn.addEventListener("click", onOk);
            cancelBtn.addEventListener("click", onCancel);
            dlg.addEventListener("close", onClose);
            dlg.showModal();
            setTimeout(() => (input ? inputEl : okBtn).focus(), 30);
        });
    }
    const confirmDialog = (opts) => openDialog(Object.assign({ confirmText: "Confirm" }, opts));
    const promptDialog = (opts) => openDialog(Object.assign({ confirmText: "Continue", input: { value: opts.value || "", placeholder: opts.placeholder || "", label: opts.label } }, opts));

    // ─────────────────────────────────────────────
    // Camera selection
    // Browsers open their "default" camera for {video: true}. On many PCs that is a
    // virtual camera (OBS, Snap, ManyCam…) which fails with "Could not start video
    // source" when its app is closed. We rank real cameras first, fall back through
    // every device, and remember the one that worked.
    // ─────────────────────────────────────────────
    const VIRTUAL_CAM = /virtual|obs|snap ?camera|manycam|xsplit|droidcam|iriun|epoccam|broadcast|mmhmm|vcam|camo|splitcam|e2esoft|youcam/i;
    const CAM_PREF_KEY = "amcs_camera_id";

    function isVirtualCamera(label) { return VIRTUAL_CAM.test(label || ""); }

    async function listCameras() {
        if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return [];
        const all = await navigator.mediaDevices.enumerateDevices();
        return all.filter(d => d.kind === "videoinput");
    }

    // Real cameras only, unless the machine has nothing but virtual ones.
    function usableCameras(cams) {
        const real = cams.filter(c => !isVirtualCamera(c.label));
        return real.length ? real : cams;
    }

    function rankCameras(cams, facingMode) {
        const saved = store.get(CAM_PREF_KEY);
        const wantBack = facingMode === "environment";
        const score = (c) => {
            let s = 0;
            if (saved && c.deviceId === saved) s += 100;
            const l = (c.label || "").toLowerCase();
            if (wantBack && /back|rear|environment/.test(l)) s += 20;
            if (/integrated|built-?in|webcam|hd camera|front|user|face/.test(l)) s += 10;
            return s;
        };
        return usableCameras(cams).slice().sort((a, b) => score(b) - score(a));
    }

    function stopStream(s) { if (s) s.getTracks().forEach(t => t.stop()); }
    const wait = (ms) => new Promise(r => setTimeout(r, ms));

    function rememberCamera(deviceId, label) {
        // Never remember a virtual camera as the preferred one
        if (deviceId && !isVirtualCamera(label)) store.set(CAM_PREF_KEY, deviceId);
    }

    /**
     * Opens the best available camera. Virtual cameras (OBS, Snap, ManyCam...) are skipped
     * whenever a real camera exists, because they show nothing unless their app is running.
     * opts: { facingMode, width, height, frameRate, deviceId }  (deviceId forces a specific camera)
     * Returns { stream, label, deviceId, cameras } or throws the last error (with .attempts).
     */
    async function openCamera(opts = {}) {
        if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
            const e = new Error("Camera API unavailable (needs HTTPS or localhost)");
            e.name = "InsecureContextError";
            throw e;
        }
        const base = {
            width: { ideal: opts.width || 1280 },
            height: { ideal: opts.height || 720 }
        };
        if (opts.frameRate) base.frameRate = { ideal: opts.frameRate };

        const attempts = [];
        const tryOpen = async (videoConstraints, label) => {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ video: videoConstraints, audio: false });
                return { stream, track: stream.getVideoTracks()[0] };
            } catch (err) {
                attempts.push(`${label}: ${err.name}`);
                throw err;
            }
        };
        const isFatal = (err) => err && (err.name === "NotAllowedError" || err.name === "SecurityError");

        let cams = await listCameras().catch(() => []);
        let lastErr = null;

        // Device labels stay hidden until permission is granted. Ask once with a plain request
        // so the browser shows its prompt, then decide based on the real device list.
        if (!opts.deviceId && (!cams.length || cams.every(c => !c.label))) {
            try {
                const { stream, track } = await tryOpen(Object.assign({}, base), "permission request");
                cams = await listCameras().catch(() => []);
                const realExists = cams.some(c => c.label && !isVirtualCamera(c.label));
                if (!isVirtualCamera(track.label) || !realExists) {
                    const id = (track.getSettings ? track.getSettings() : {}).deviceId;
                    rememberCamera(id, track.label);
                    return { stream, label: track.label, deviceId: id, cameras: cams };
                }
                stopStream(stream);      // the browser picked a virtual camera; open a real one below
                await wait(300);
            } catch (err) {
                lastErr = err;
                if (isFatal(err)) throw err;
                cams = await listCameras().catch(() => []);
            }
        }

        const ordered = opts.deviceId
            ? [cams.find(c => c.deviceId === opts.deviceId) || { deviceId: opts.deviceId, label: "selected camera" }]
            : rankCameras(cams, opts.facingMode);

        for (const cam of ordered) {
            // A camera that was just released (page reload, camera switch, another app closing)
            // can report "busy" for a moment on Windows, so give each one a second chance.
            for (let round = 0; round < 2; round++) {
                try {
                    const { stream, track } = await tryOpen(Object.assign({}, base, { deviceId: { exact: cam.deviceId } }), cam.label || "camera");
                    const label = track.label || cam.label;
                    rememberCamera(cam.deviceId, label);
                    return { stream, label, deviceId: cam.deviceId, cameras: cams };
                } catch (err) {
                    lastErr = err;
                    if (isFatal(err)) throw err;
                    if (err.name !== "NotReadableError" && err.name !== "AbortError") break;
                    await wait(800);
                }
            }
        }

        // Last resort when the device list is empty or unlabeled: let the browser choose.
        // Skipped when real cameras exist, because the browser default is often a virtual camera.
        const realKnown = cams.some(c => c.label && !isVirtualCamera(c.label));
        if (!realKnown) {
            try {
                const { stream, track } = await tryOpen(true, "any camera");
                return { stream, label: track.label, deviceId: null, cameras: cams };
            } catch (err) {
                lastErr = err;
            }
        }
        lastErr = lastErr || new Error("No camera found");
        if (!cams.length && lastErr.name !== "NotAllowedError") lastErr = Object.assign(new Error("No camera found"), { name: "NotFoundError" });
        lastErr.attempts = attempts;
        throw lastErr;
    }

    /**
     * Some Windows camera drivers drop a stream a second or so after it opens (the track
     * fires "ended"). Reconnect automatically a few times before telling the user.
     * handlers: { reopen() -> Promise, onReconnecting(), onGiveUp() }
     */
    const cameraDrops = [];
    function guardCameraStream(stream, handlers) {
        const track = stream && stream.getVideoTracks()[0];
        if (!track) return;
        track.addEventListener("ended", () => {
            const now = Date.now();
            while (cameraDrops.length && now - cameraDrops[0] > 30000) cameraDrops.shift();
            cameraDrops.push(now);
            console.warn(`[AMCS] Camera stream ended (${track.label}); drop ${cameraDrops.length} in 30s`);
            if (cameraDrops.length <= 4) {
                if (handlers.onReconnecting) handlers.onReconnecting();
                setTimeout(() => handlers.reopen(), 600 * cameraDrops.length);
            } else if (handlers.onGiveUp) {
                handlers.onGiveUp();
            }
        });
    }

    function describeCameraError(err) {
        const name = err && err.name;
        if (name === "InsecureContextError") return { title: "Camera unavailable", text: "Browsers only allow the camera on HTTPS or localhost. Open http://localhost:5000 on this PC, or start the server with AMCS_HTTPS=1 for phones." };
        if (name === "NotAllowedError" || name === "SecurityError") return { title: "Camera permission blocked", text: "Click the camera icon in the address bar, choose Allow, then press Try again." };
        if (name === "NotFoundError" || name === "OverconstrainedError") return { title: "No camera found", text: "Connect a camera, or open AMCS on a phone." };
        if (name === "NotReadableError" || name === "AbortError" || name === "TrackStartError") return { title: "Camera is busy", text: "The laptop camera could not start. Close other apps that may be using it (Zoom, Teams, the Camera app), then press Try again." };
        return { title: "Camera could not start", text: (err && err.message) || "Unknown camera error. Press Try again." };
    }

    // ─────────────────────────────────────────────
    // Global SOS banner & server connectivity pill
    // ─────────────────────────────────────────────
    let serverOnline = true;
    function setServerStatus(online) {
        if (online === serverOnline) return;
        serverOnline = online;
        const pill = document.getElementById("server-status");
        if (!pill) return;
        pill.classList.toggle("offline", !online);
        const label = pill.querySelector(".status-label");
        if (label) label.textContent = online ? "Online" : "Offline";
        pill.setAttribute("title", online ? "Server connected" : "Server unreachable");
        if (!online) toast({ title: "Connection lost", message: "Trying to reconnect to the AMCS server…", type: "warning" });
        else toast({ title: "Reconnected", message: "Server connection restored.", type: "success", duration: 2200 });
    }

    async function checkSOSGlobal() {
        try {
            const res = await fetch("/api/location/status", { cache: "no-store" });
            if (!res.ok) throw new Error("HTTP " + res.status);
            const data = await res.json();
            setServerStatus(true);
            const banner = document.getElementById("global-sos-alert");
            if (!banner) return data;
            if (data.sos_active) {
                banner.hidden = false;
                document.getElementById("global-sos-text").textContent = "Emergency alert: " + (data.sos_message || "Assistance requested!");
            } else {
                banner.hidden = true;
            }
            return data;
        } catch (e) {
            setServerStatus(false);
            return null;
        }
    }

    async function resolveSOSGlobal() {
        try {
            await fetch("/api/sos/resolve", { method: "POST" });
            const banner = document.getElementById("global-sos-alert");
            if (banner) banner.hidden = true;
            toast({ title: "SOS cleared", message: "The emergency alert has been resolved.", type: "success" });
        } catch (e) {
            toast({ title: "Could not clear SOS", message: "Check the server connection and try again.", type: "error" });
        }
    }

    // ─────────────────────────────────────────────
    // Boot
    // ─────────────────────────────────────────────
    applyTheme(store.get("amcs_theme", "system"));
    if (window.matchMedia) {
        const mq = window.matchMedia("(prefers-color-scheme: light)");
        const onChange = () => { if (store.get("amcs_theme", "system") === "system") applyTheme("system"); };
        if (mq.addEventListener) mq.addEventListener("change", onChange);
    }

    document.addEventListener("DOMContentLoaded", () => {
        applyTheme(store.get("amcs_theme", "system"));
        applyContrast(store.get("amcs_contrast") === "true");
        applyLite();

        const themeBtn = document.getElementById("theme-toggle");
        if (themeBtn) themeBtn.addEventListener("click", cycleTheme);
        const contrastBtn = document.getElementById("contrast-toggle");
        if (contrastBtn) contrastBtn.addEventListener("click", () => {
            const on = !document.body.classList.contains("high-contrast");
            applyContrast(on);
            store.set("amcs_contrast", on ? "true" : "false");
        });
        const sosResolve = document.getElementById("global-sos-resolve");
        if (sosResolve) sosResolve.addEventListener("click", resolveSOSGlobal);

        checkSOSGlobal();
        setInterval(checkSOSGlobal, 5000);

        if ("serviceWorker" in navigator) {
            navigator.serviceWorker.register("/sw.js").catch(() => {});
        }
    });

    // ─────────────────────────────────────────────
    // Voice (text-to-speech)
    // Modern browsers ship natural neural voices (Edge: "Microsoft ... Online (Natural)",
    // Chrome: "Google ...", Safari: Siri / Enhanced). They sound far better and start instantly,
    // unlike server-generated gTTS audio. We pick the best voice for the language and fall back
    // to the server voice only when the device has nothing decent.
    // ─────────────────────────────────────────────
    const LANG_LOCALES = {
        en: "en-IN", hi: "hi-IN", es: "es-ES", fr: "fr-FR", de: "de-DE", "zh-CN": "zh-CN", ja: "ja-JP",
        ar: "ar-SA", ru: "ru-RU", bn: "bn-IN", ta: "ta-IN", te: "te-IN", mr: "mr-IN", gu: "gu-IN"
    };
    const hasTTS = typeof window !== "undefined" && "speechSynthesis" in window;
    let voiceCache = [];

    function loadVoices() {
        if (!hasTTS) return [];
        const v = window.speechSynthesis.getVoices();
        if (v && v.length) voiceCache = v;
        return voiceCache;
    }
    if (hasTTS) {
        loadVoices();
        if (window.speechSynthesis.addEventListener) window.speechSynthesis.addEventListener("voiceschanged", loadVoices);
    }

    const baseLang = (code) => String(code || "en").toLowerCase().replace("_", "-").split("-")[0];

    function voiceScore(v) {
        const n = (v.name || "").toLowerCase();
        let s = 0;
        if (/natural/.test(n)) s += 60;
        if (/neural|premium|enhanced|wavenet|studio/.test(n)) s += 45;
        if (/online/.test(n)) s += 20;
        if (/^google/.test(n)) s += 35;
        if (/siri/.test(n)) s += 30;
        if (/espeak|robot|novelty|bad news|bells|boing|bubbles|cellos|whisper|zarvox|trinoids/.test(n)) s -= 80;
        const preferred = (navigator.languages || [navigator.language || ""]).map(x => String(x).toLowerCase());
        if (preferred.includes(String(v.lang || "").toLowerCase().replace("_", "-"))) s += 10;
        return s;
    }

    /** Voices for a language, best first. */
    function voicesFor(lang) {
        const want = baseLang(lang);
        return loadVoices()
            .filter(v => baseLang(v.lang) === want)
            .sort((a, b) => voiceScore(b) - voiceScore(a));
    }

    function pickVoice(lang) {
        const list = voicesFor(lang);
        const chosen = getPrefs().voiceName;
        if (chosen) {
            const match = list.find(v => v.name === chosen);
            if (match) return match;
        }
        return list[0] || null;
    }

    const isNaturalVoice = (v) => !!v && voiceScore(v) >= 30;

    /** True when the browser voice should be used instead of the server (gTTS) voice. */
    function useDeviceVoice(lang) {
        if (!hasTTS) return false;
        const engine = getPrefs().voiceEngine || "auto";
        if (engine === "server") return false;
        const v = pickVoice(lang);
        if (engine === "device") return !!v || voicesFor("en").length > 0;
        return isNaturalVoice(v) || (!!v && getPrefs().voiceName === v.name);
    }

    let activeUtterance = null;   // keep a reference: Chrome drops callbacks of collected utterances
    /** Speaks text with the best device voice. opts: { interrupt=true, onstart, onend } */
    function speak(text, lang, opts = {}) {
        if (!hasTTS || !text) { if (opts.onend) opts.onend(); return false; }
        const synth = window.speechSynthesis;
        if (opts.interrupt !== false) synth.cancel();
        const u = new SpeechSynthesisUtterance(text);
        const v = pickVoice(lang);
        if (v) { u.voice = v; u.lang = v.lang; } else { u.lang = LANG_LOCALES[lang] || lang || "en-IN"; }
        const p = getPrefs();
        u.rate = Math.min(2, Math.max(0.5, Number(p.speechRate) || 1));
        u.pitch = 1;
        u.volume = 1;
        u.onstart = () => { if (opts.onstart) opts.onstart(); };
        u.onend = u.onerror = () => { if (activeUtterance === u) activeUtterance = null; if (opts.onend) opts.onend(); };
        activeUtterance = u;
        if (synth.paused) synth.resume();
        synth.speak(u);
        return true;
    }

    function stopSpeaking() { if (hasTTS) window.speechSynthesis.cancel(); }
    const isSpeaking = () => hasTTS && (window.speechSynthesis.speaking || window.speechSynthesis.pending);

    window.AMCS = {
        store, getPrefs, setPrefs, resetPrefs, DEFAULT_PREFS,
        isPhone: IS_PHONE, resolveQuality, liteActive, applyLite,
        setTheme(mode) { store.set("amcs_theme", mode); applyTheme(mode); },
        setContrast(on) { applyContrast(on); store.set("amcs_contrast", on ? "true" : "false"); },
        escapeHtml, toast, confirm: confirmDialog, prompt: promptDialog,
        openCamera, listCameras, usableCameras, isVirtualCamera, guardCameraStream, describeCameraError,
        voice: { speak, stop: stopSpeaking, isSpeaking, voicesFor, pickVoice, useDeviceVoice, isNaturalVoice, voiceScore, LANG_LOCALES },
        checkSOS: checkSOSGlobal, resolveSOS: resolveSOSGlobal
    };
    // Backwards compatibility with inline handlers in older templates
    window.resolveSOSGlobal = resolveSOSGlobal;
})();
