/**
 * Caregiver Dashboard & Live Folium Map Controller
 * Automatic Multilanguage Communication System - SKIT Jaipur
 */

const esc = (s) => (window.AMCS ? window.AMCS.escapeHtml(s) : String(s));
const notify = (title, message, type = "info") => window.AMCS && window.AMCS.toast({ title, message, type });

let lastLocationLat = null;
let lastLocationLon = null;
let lastMapReload = 0;
let wasSosActive = false;
let lastSosChime = 0;
const hiddenFeedKeys = new Set();   // entries the caregiver cleared locally
const STALE_AFTER_SEC = 60;         // no update for a minute = device idle

async function pollCaregiverStatus() {
    try {
        const response = await fetch("/api/location/status", { cache: "no-store" });
        if (!response.ok) return;
        const data = await response.json();
        updateCaregiverHUD(data);
    } catch (e) {
        console.warn("[AMCS Caregiver] Status poll error:", e);
    }
}

function formatAgo(seconds) {
    if (seconds < 5) return "just now";
    if (seconds < 60) return `${Math.round(seconds)} s ago`;
    if (seconds < 3600) return `${Math.round(seconds / 60)} min ago`;
    return `${Math.round(seconds / 3600)} h ago`;
}

function updateCaregiverHUD(data) {
    const lat = Number(data.latitude);
    const lon = Number(data.longitude);

    // 1. Tracker freshness
    const statusEl = document.getElementById("user-status-text");
    const pingEl = document.getElementById("last-ping-time");
    if (data.last_updated) {
        const age = Math.max(0, Date.now() / 1000 - Number(data.last_updated));
        const live = age < STALE_AFTER_SEC;
        if (statusEl) {
            statusEl.textContent = live ? "Active · Live" : "Idle";
            statusEl.className = live ? "text-success" : "text-warning";
        }
        if (pingEl) pingEl.textContent = `Last update ${formatAgo(age)}`;
    }

    // 2. Position
    const coordEl = document.getElementById("coord-display");
    const accEl = document.getElementById("accuracy-display");
    if (coordEl && isFinite(lat) && isFinite(lon)) coordEl.textContent = `${lat.toFixed(4)}, ${lon.toFixed(4)}`;
    if (accEl) accEl.textContent = `Accuracy ±${Math.round(data.accuracy || 0)} m · ${data.location_name || "SKIT Jaipur"}`;

    const mapsUrl = `https://www.google.com/maps?q=${lat},${lon}`;
    const gmaps = document.getElementById("gmaps-link");
    if (gmaps) gmaps.href = mapsUrl;

    // 3. Geofence
    const geoText = document.getElementById("geofence-status-text");
    const geoDist = document.getElementById("geofence-distance");
    const geoIconBox = document.getElementById("geofence-icon-container");
    if (data.geofence) {
        const breached = !!data.geofence.breached;
        if (geoText) {
            geoText.textContent = breached ? "Outside safe zone" : "Inside safe zone";
            geoText.className = breached ? "text-danger" : "text-success";
        }
        if (geoIconBox) geoIconBox.className = breached ? "kpi-icon kpi-red" : "kpi-icon kpi-amber";
        if (geoDist) geoDist.textContent = `${Math.round(data.geofence.distance_from_center_m)} m from SKIT centre`;
    }

    // 4. SOS
    const sosText = document.getElementById("sos-status-text");
    const sosIconBox = document.getElementById("sos-icon-container");
    const sosIcon = document.getElementById("sos-kpi-icon");
    const resolveBtn = document.getElementById("kpi-resolve-btn");
    if (data.sos_active) {
        if (sosText) { sosText.textContent = "Emergency SOS active"; sosText.className = "text-danger"; }
        if (sosIconBox) sosIconBox.className = "kpi-icon kpi-red";
        if (sosIcon) sosIcon.className = "fa-solid fa-triangle-exclamation fa-beat";
        if (resolveBtn) resolveBtn.style.display = "inline-flex";
        // Alarm immediately, then every 10 s while unresolved (not on every poll)
        const now = Date.now();
        if (!wasSosActive || now - lastSosChime > 10000) {
            playSOSChime();
            lastSosChime = now;
        }
        if (!wasSosActive) notify("Emergency SOS", data.sos_message || "The user requested help.", "error");
    } else {
        if (sosText) { sosText.textContent = "Safe · Normal"; sosText.className = "text-success"; }
        if (sosIconBox) sosIconBox.className = "kpi-icon kpi-green";
        if (sosIcon) sosIcon.className = "fa-solid fa-shield-heart";
        if (resolveBtn) resolveBtn.style.display = "none";
    }
    wasSosActive = !!data.sos_active;

    // 5. WhatsApp share carries the live location
    const wa = document.getElementById("caregiver-wa-btn");
    if (wa) {
        const state = data.sos_active ? "EMERGENCY SOS ACTIVE" : "Safe";
        const msg = `AMCS status: ${state}. Location: ${data.location_name || "SKIT Jaipur"} ${mapsUrl}`;
        wa.href = "https://api.whatsapp.com/send?text=" + encodeURIComponent(msg);
    }

    // 6. Activity feed
    if (data.activity_log) renderActivityFeed(data.activity_log);

    // 7. Reload the map when the user moved (~10 m) — at most every 8 s
    const moved = lastLocationLat === null ||
        Math.abs(lastLocationLat - lat) > 0.0001 ||
        Math.abs(lastLocationLon - lon) > 0.0001;
    if (moved && Date.now() - lastMapReload > 8000) {
        lastLocationLat = lat;
        lastLocationLon = lon;
        reloadFoliumMap();
    }
}

const feedKey = (item) => `${item.timestamp}|${item.type}|${item.description}`;

function renderActivityFeed(logEntries) {
    const feedContainer = document.getElementById("activity-feed");
    const badge = document.getElementById("log-count-badge");
    if (!feedContainer) return;

    const visible = (logEntries || []).filter(item => !hiddenFeedKeys.has(feedKey(item)));
    if (badge) badge.textContent = visible.length === 1 ? "1 event" : `${visible.length} events`;

    if (visible.length === 0) {
        feedContainer.innerHTML = '<div class="activity-empty"><i class="fa-solid fa-satellite" aria-hidden="true"></i><p>No new activity.</p></div>';
        return;
    }

    const html = visible.map(item => {
        let icon = "fa-circle-info";
        let itemClass = "item-info";
        if (item.severity === "danger" || item.type === "sos") {
            icon = "fa-triangle-exclamation";
            itemClass = "item-danger";
        } else if (item.severity === "warning" || item.type === "geofence") {
            icon = "fa-circle-exclamation";
            itemClass = "item-warning";
        } else if (item.type === "ocr") {
            icon = "fa-language";
            itemClass = "item-success";
        } else if (item.type === "detection") {
            icon = "fa-eye";
        } else if (item.type === "social") {
            icon = "fa-user-check";
            itemClass = "item-success";
        }
        return `
            <div class="activity-item ${itemClass}">
                <div class="activity-icon" aria-hidden="true"><i class="fa-solid ${icon}"></i></div>
                <div class="activity-body">
                    <span class="activity-time">${esc(item.timestamp)}</span>
                    <div class="activity-desc">${esc(item.description)}</div>
                </div>
            </div>`;
    }).join("");

    // Avoid re-rendering (and re-announcing) an unchanged feed
    if (feedContainer.dataset.sig !== html.length + "|" + visible.length + "|" + feedKey(visible[0])) {
        feedContainer.innerHTML = html;
        feedContainer.dataset.sig = html.length + "|" + visible.length + "|" + feedKey(visible[0]);
    }
}

function reloadFoliumMap() {
    const iframe = document.getElementById("folium-map-iframe");
    if (iframe) {
        lastMapReload = Date.now();
        iframe.src = "/api/map?t=" + Date.now();
    }
}

function toggleMapFullscreen() {
    const mapBox = document.getElementById("folium-frame-box");
    if (!document.fullscreenElement) {
        if (mapBox.requestFullscreen) {
            mapBox.requestFullscreen().catch(err => notify("Full screen unavailable", err.message, "warning"));
        }
    } else {
        document.exitFullscreen();
    }
}

async function resolveCaregiverSOS() {
    const ok = await AMCS.confirm({
        title: "Mark the SOS as resolved?",
        message: "Only do this after you have reached the user or confirmed they are safe.",
        icon: "fa-shield-heart",
        confirmText: "Mark resolved"
    });
    if (!ok) return;
    try {
        await fetch("/api/sos/resolve", { method: "POST" });
        notify("SOS resolved", "The emergency alert has been cleared.", "success");
        pollCaregiverStatus();
        if (window.AMCS) window.AMCS.checkSOS();
    } catch (e) {
        notify("Could not resolve alert", "Check the server connection and try again.", "error");
    }
}

async function sendCaregiverChime() {
    // Delivered through the intercom channel; the user's device speaks it aloud.
    try {
        const res = await fetch("/api/teleassist/intercom", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: "Your caregiver is checking in. Say help if you need assistance." })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        notify("Check-in sent", "The user's device will speak it while their assistant is running.", "success");
    } catch (e) {
        notify("Check-in failed", "Could not reach the server.", "error");
    }
}

function clearCaregiverFeed() {
    const feedContainer = document.getElementById("activity-feed");
    if (!feedContainer) return;
    // Remember what was cleared so the next poll does not bring it back
    fetch("/api/location/status", { cache: "no-store" })
        .then(r => r.json())
        .then(data => {
            (data.activity_log || []).forEach(item => hiddenFeedKeys.add(feedKey(item)));
            renderActivityFeed(data.activity_log || []);
        })
        .catch(() => {});
    feedContainer.innerHTML = '<div class="activity-empty"><i class="fa-solid fa-broom" aria-hidden="true"></i><p>Feed cleared. New events will appear here.</p></div>';
    feedContainer.dataset.sig = "";
}

let sosAudioCtx = null;
function playSOSChime() {
    try {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (!AudioCtx) return;
        if (!sosAudioCtx) sosAudioCtx = new AudioCtx();
        if (sosAudioCtx.state === "suspended") sosAudioCtx.resume().catch(() => {});
        const t = sosAudioCtx.currentTime;
        [0, 0.38].forEach(offset => {
            const osc = sosAudioCtx.createOscillator();
            const gain = sosAudioCtx.createGain();
            osc.type = "sawtooth";
            osc.frequency.setValueAtTime(880, t + offset);
            osc.frequency.exponentialRampToValueAtTime(440, t + offset + 0.3);
            gain.gain.setValueAtTime(0.25, t + offset);
            gain.gain.exponentialRampToValueAtTime(0.01, t + offset + 0.3);
            osc.connect(gain);
            gain.connect(sosAudioCtx.destination);
            osc.start(t + offset);
            osc.stop(t + offset + 0.35);
        });
    } catch (e) {}
}

// ─────────────────────────────────────────────
// Caregiver "See What I See" Live Tele-Assistance
// ─────────────────────────────────────────────
let isTeleAssistActive = false;
let teleAssistPollTimer = null;

async function toggleTeleAssistCall() {
    const btn = document.getElementById("teleassist-btn");
    const badge = document.getElementById("teleassist-badge");
    const viewport = document.getElementById("teleassist-viewport");

    if (isTeleAssistActive) {
        isTeleAssistActive = false;
        clearInterval(teleAssistPollTimer);
        teleAssistPollTimer = null;
        if (btn) { btn.innerHTML = '<i class="fa-solid fa-video" aria-hidden="true"></i> Start video call'; btn.className = "btn btn-primary btn-block"; }
        if (badge) { badge.textContent = "Standby"; badge.className = "badge badge-inactive"; }
        if (viewport) viewport.style.display = "none";
        await fetch("/api/teleassist/call", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "stop" })
        }).catch(() => {});
    } else {
        isTeleAssistActive = true;
        if (btn) { btn.innerHTML = '<i class="fa-solid fa-phone-slash" aria-hidden="true"></i> End call'; btn.className = "btn btn-danger btn-block"; }
        if (badge) { badge.textContent = "Live"; badge.className = "badge badge-danger badge-live"; }
        if (viewport) viewport.style.display = "block";
        await fetch("/api/teleassist/call", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ action: "start" })
        }).catch(() => {});

        pollTeleAssistVideo();
        teleAssistPollTimer = setInterval(pollTeleAssistVideo, 800);
        const input = document.getElementById("caregiver-intercom-input");
        if (input) input.focus();
    }
}

async function pollTeleAssistVideo() {
    if (!isTeleAssistActive) return;
    try {
        const resp = await fetch("/api/teleassist/frame", { cache: "no-store" });
        const data = await resp.json();
        const imgEl = document.getElementById("teleassist-feed-img");
        const placeholder = document.getElementById("teleassist-placeholder");
        if (data.has_frame && data.frame && imgEl && String(data.frame).startsWith("data:image")) {
            imgEl.src = data.frame;
            imgEl.style.display = "block";
            if (placeholder) placeholder.style.display = "none";
        }
    } catch (e) {}
}

async function sendCaregiverIntercomText() {
    const input = document.getElementById("caregiver-intercom-input");
    if (!input || !input.value.trim()) return;
    const msg = input.value.trim();
    try {
        const res = await fetch("/api/teleassist/intercom", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ message: msg })
        });
        if (!res.ok) throw new Error("HTTP " + res.status);
        input.value = "";
        notify("Message sent", `“${msg}” will be spoken to the user.`, "success");
    } catch (e) {
        notify("Message not sent", "Could not reach the server.", "error");
    }
}

window.addEventListener("DOMContentLoaded", () => {
    pollCaregiverStatus();
    setInterval(pollCaregiverStatus, 3500);
});
