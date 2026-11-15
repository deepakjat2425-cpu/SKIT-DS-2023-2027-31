/**
 * AMCS live panels for the Assist page: telemetry tiles, proximity radar, objects in view and
 * the guidance history. main.js feeds them through window.AMCSPanels; nothing here touches the
 * camera, speech or the server.
 * Automatic Multilanguage Communication System - SKIT Jaipur
 */
(function () {
    "use strict";

    const $ = (id) => document.getElementById(id);
    const esc = (v) => (window.AMCS && window.AMCS.escapeHtml ? window.AMCS.escapeHtml(v) : String(v));

    const NAMES = {
        "dining table": "table", "tv": "TV", "cell phone": "phone", "couch": "sofa", "potted plant": "plant",
        "wall switch": "light switch", "trash can": "dustbin", "wine glass": "glass", "handbag": "bag"
    };
    const friendly = (d) => (d.is_user ? "you" : (NAMES[d.class] || d.class || "object"));
    const severityOf = (d) => (["hazard", "caution"].includes(d.severity) ? d.severity : "safe");
    const SIDE = { left: "Left", right: "Right", ahead: "Ahead" };

    // ─────────────────────────────────────────────
    // Proximity radar: a top-down view, the user at the bottom, distance rings going outwards
    // ─────────────────────────────────────────────
    const RADAR_MAX_M = 8;            // objects further away sit on the outer ring
    const RADAR_RINGS = [1.3, 2.8, 5, 8];
    const ANGLE_SPREAD = 1.8;         // the camera only sees ~70 degrees; widen it to use the panel
    const radar = { canvas: null, ctx: null, w: 0, h: 0, blips: [], live: false, sweep: 0, colors: null };

    function readColors() {
        const cs = getComputedStyle(document.body);
        const v = (name, fallback) => (cs.getPropertyValue(name) || "").trim() || fallback;
        radar.colors = {
            line: v("--line-strong", "rgba(255,255,255,0.13)"),
            faint: v("--line", "rgba(255,255,255,0.07)"),
            text: v("--text", "#f3f5f9"),
            muted: v("--text-3", "#717a8c"),
            primary: v("--primary", "#8794ff"),
            accent: v("--accent", "#45d9e8"),
            hazard: v("--danger", "#ff6b6b"),
            caution: v("--warning", "#f6b84a"),
            safe: v("--success", "#3ddc97"),
            surface: v("--bg-elev", "#0e1016")
        };
    }

    function sizeRadar() {
        const c = radar.canvas;
        if (!c) return;
        const dpr = Math.min(2, window.devicePixelRatio || 1);
        const w = Math.round(c.clientWidth), h = Math.round(c.clientHeight);
        if (!w || !h) return;
        if (c.width !== Math.round(w * dpr) || c.height !== Math.round(h * dpr)) {
            c.width = Math.round(w * dpr);
            c.height = Math.round(h * dpr);
        }
        radar.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        radar.w = w;
        radar.h = h;
    }

    /** Polar position (bearing in degrees, distance in metres) -> canvas point. */
    function project(bearingDeg, distanceM) {
        const ox = radar.w / 2, oy = radar.h - 16;
        const maxR = Math.min(radar.h - 34, radar.w / 2 - 8);
        const r = maxR * Math.sqrt(Math.min(RADAR_MAX_M, Math.max(0.2, distanceM)) / RADAR_MAX_M);
        const a = (Math.max(-40, Math.min(40, bearingDeg)) * ANGLE_SPREAD) * Math.PI / 180;
        return { x: ox + Math.sin(a) * r, y: oy - Math.cos(a) * r, ox, oy, maxR };
    }

    function setBlips(detections) {
        const now = performance.now();
        radar.blips.forEach(b => { b.seen = false; });
        (detections || []).forEach(d => {
            if (d.is_user || typeof d.distance_m !== "number") return;
            const bearing = typeof d.bearing_deg === "number"
                ? d.bearing_deg
                : (d.spatial === "left" ? -24 : d.spatial === "right" ? 24 : 0);
            const key = d.track_id !== undefined ? "t" + d.track_id : d.class + ":" + d.spatial;
            let b = radar.blips.find(x => x.key === key && !x.seen);
            if (!b) {
                b = { key, bearing, distance: d.distance_m, alpha: 0, born: now };
                radar.blips.push(b);
            }
            b.seen = true;
            b.targetBearing = bearing;
            b.targetDistance = d.distance_m;
            b.severity = severityOf(d);
            b.label = friendly(d);
        });
    }

    // Lite mode (phones by default) redraws the radar 12 times a second instead of every frame
    const LITE = !!(window.AMCS && window.AMCS.liteActive && window.AMCS.liteActive());
    let lastDraw = 0;
    const STEP = LITE ? 5 : 1;          // animation steps per drawn frame, so motion keeps its speed

    function drawRadar(t) {
        const g = radar.ctx;
        if (!g || !radar.canvas.isConnected) return;
        if (LITE && t - lastDraw < 83) { requestAnimationFrame(drawRadar); return; }
        lastDraw = t;
        sizeRadar();
        if (!radar.w) { requestAnimationFrame(drawRadar); return; }
        if (!radar.colors) readColors();
        const C = radar.colors;
        const { ox, oy, maxR } = project(0, RADAR_MAX_M);
        const half = 35 * ANGLE_SPREAD * Math.PI / 180;
        g.clearRect(0, 0, radar.w, radar.h);

        // Field of view, tinted by zone: hazard distance, caution distance, the rest
        const zone = (m, color, alpha) => {
            const r = maxR * Math.sqrt(m / RADAR_MAX_M);
            g.beginPath();
            g.moveTo(ox, oy);
            g.arc(ox, oy, r, -Math.PI / 2 - half, -Math.PI / 2 + half);
            g.closePath();
            g.globalAlpha = alpha;
            g.fillStyle = color;
            g.fill();
            g.globalAlpha = 1;
        };
        zone(RADAR_MAX_M, C.primary, 0.07);
        zone(2.8, C.caution, 0.08);
        zone(1.3, C.hazard, 0.13);

        // Distance rings and their labels
        g.lineWidth = 1;
        g.font = "600 10px Inter, system-ui, sans-serif";
        g.textAlign = "left";
        RADAR_RINGS.forEach(m => {
            const r = maxR * Math.sqrt(m / RADAR_MAX_M);
            g.beginPath();
            g.arc(ox, oy, r, -Math.PI / 2 - half, -Math.PI / 2 + half);
            g.strokeStyle = m === RADAR_MAX_M ? C.line : C.faint;
            g.stroke();
            const lx = ox + Math.sin(half) * r + 5, ly = oy - Math.cos(half) * r + 3;
            g.fillStyle = C.muted;
            g.fillText(m + " m", Math.min(lx, radar.w - 28), ly);
        });
        [-half, 0, half].forEach(a => {
            g.beginPath();
            g.moveTo(ox, oy);
            g.lineTo(ox + Math.sin(a) * maxR, oy - Math.cos(a) * maxR);
            g.strokeStyle = a === 0 ? C.faint : C.line;
            g.setLineDash(a === 0 ? [3, 5] : []);
            g.stroke();
        });
        g.setLineDash([]);

        // Sweep while the assistant is running
        if (radar.live) {
            radar.sweep = (t / 2600) % 1;
            const r = maxR * radar.sweep;
            g.beginPath();
            g.arc(ox, oy, r, -Math.PI / 2 - half, -Math.PI / 2 + half);
            g.strokeStyle = C.accent;
            g.globalAlpha = 0.55 * (1 - radar.sweep);
            g.lineWidth = 2;
            g.stroke();
            g.globalAlpha = 1;
            g.lineWidth = 1;
        }

        // Objects
        const labelled = [];
        for (let i = radar.blips.length - 1; i >= 0; i--) {
            const b = radar.blips[i];
            if (b.seen) {
                const k = 1 - Math.pow(1 - 0.18, STEP);
                b.bearing += (b.targetBearing - b.bearing) * k;
                b.distance += (b.targetDistance - b.distance) * k;
                b.alpha = Math.min(1, b.alpha + 0.1 * STEP);
            } else {
                b.alpha -= 0.06 * STEP;
                if (b.alpha <= 0) { radar.blips.splice(i, 1); continue; }
            }
            const p = project(b.bearing, b.distance);
            const color = C[b.severity] || C.safe;
            g.globalAlpha = b.alpha;
            if (b.severity !== "safe") {
                const pulse = 1 + 0.35 * Math.sin(t / 260 + i);
                g.beginPath();
                g.arc(p.x, p.y, 9 * pulse, 0, Math.PI * 2);
                g.fillStyle = color;
                g.globalAlpha = b.alpha * 0.2;
                g.fill();
                g.globalAlpha = b.alpha;
            }
            g.beginPath();
            g.arc(p.x, p.y, b.severity === "hazard" ? 5.5 : 4.5, 0, Math.PI * 2);
            g.fillStyle = color;
            g.fill();
            g.lineWidth = 2;
            g.strokeStyle = C.surface;
            g.stroke();
            g.lineWidth = 1;
            labelled.push({ b, p });
        }
        // Name the nearest few; more labels than that just overlap
        labelled.sort((a, c) => a.b.distance - c.b.distance).slice(0, 4).forEach(({ b, p }) => {
            g.globalAlpha = b.alpha;
            g.font = "600 11px Inter, system-ui, sans-serif";
            const text = b.label.charAt(0).toUpperCase() + b.label.slice(1);
            const tw = g.measureText(text).width;
            const right = p.x + 10 + tw < radar.w - 4;
            const tx = p.x + (right ? 10 : -10), ty = Math.max(12, p.y + 4);
            // A backing plate keeps the name readable over rings and their distance labels
            g.fillStyle = C.surface;
            g.globalAlpha = b.alpha * 0.82;
            g.beginPath();
            const plateX = right ? tx - 4 : tx - tw - 4;
            if (g.roundRect) g.roundRect(plateX, ty - 11, tw + 8, 15, 5);
            else g.rect(plateX, ty - 11, tw + 8, 15);
            g.fill();
            g.globalAlpha = b.alpha;
            g.textAlign = right ? "left" : "right";
            g.fillStyle = C.text;
            g.fillText(text, tx, ty);
        });
        g.globalAlpha = 1;

        // The user
        g.beginPath();
        g.arc(ox, oy, 6, 0, Math.PI * 2);
        g.fillStyle = C.primary;
        g.fill();
        g.beginPath();
        g.arc(ox, oy, 10, 0, Math.PI * 2);
        g.strokeStyle = C.primary;
        g.globalAlpha = 0.4;
        g.stroke();
        g.globalAlpha = 1;

        requestAnimationFrame(drawRadar);
    }

    // ─────────────────────────────────────────────
    // Telemetry tiles
    // ─────────────────────────────────────────────
    function modelLabel(stats) {
        if (!stats || !stats.model) return null;
        return stats.model.replace(/\.pt$/, "").replace(/^yolo/i, "YOLO");
    }

    function updateTelemetry(data, info) {
        const dets = (data.detections || []).filter(d => !d.is_user);
        const objects = $("tele-objects"), objectsSub = $("tele-objects-sub");
        if (objects) objects.textContent = String(dets.length);
        if (objectsSub) {
            const hazards = dets.filter(d => d.severity === "hazard").length;
            const cautions = dets.filter(d => d.severity === "caution").length;
            objectsSub.textContent = hazards ? `${hazards} hazard${hazards > 1 ? "s" : ""} close by`
                : cautions ? `${cautions} to be careful of` : (dets.length === 1 ? "object detected" : "objects detected");
        }

        const nearest = dets[0];
        const tile = $("tele-nearest-tile"), near = $("tele-nearest"), nearSub = $("tele-nearest-sub");
        if (near) {
            if (nearest) near.innerHTML = `${esc(nearest.distance_m)}<span class="unit">m</span>`;
            else near.textContent = "—";
        }
        if (nearSub) nearSub.textContent = nearest ? `${friendly(nearest)} · ${(SIDE[nearest.spatial] || "ahead").toLowerCase()}` : "nothing close";
        if (tile) {
            tile.classList.toggle("is-hazard", !!nearest && nearest.severity === "hazard");
            tile.classList.toggle("is-caution", !!nearest && nearest.severity === "caution");
        }

        const lat = $("tele-latency"), latSub = $("tele-latency-sub");
        if (lat && info && typeof info.latency === "number") lat.innerHTML = `${Math.round(info.latency)}<span class="unit">ms</span>`;
        if (latSub && data.stats && typeof data.stats.detect_ms === "number") {
            latSub.textContent = `detector ${Math.round(data.stats.detect_ms)} ms · ${data.stats.image_size || 640} px`;
        }

        const model = $("tele-model"), modelSub = $("tele-model-sub");
        const label = modelLabel(data.stats);
        if (model && label) model.textContent = label;
        if (modelSub && data.stats && label) modelSub.textContent = data.stats.extended_model ? "+ extended vocabulary" : "80 object classes";

        const chip = $("low-light-chip");
        if (chip) chip.hidden = !(data.stats && data.stats.low_light);
    }

    // ─────────────────────────────────────────────
    // Objects in view
    // ─────────────────────────────────────────────
    function updateObjectList(detections) {
        const list = $("object-list");
        if (!list) return;
        const dets = (detections || []).filter(d => !d.is_user).slice(0, 12);
        if (!dets.length) {
            list.innerHTML = '<li class="object-empty">Nothing detected right now.</li>';
            return;
        }
        list.innerHTML = dets.map(d => {
            const sev = severityOf(d);
            const pct = Math.round(Math.max(0, Math.min(1, Number(d.confidence) || 0)) * 100);
            const extra = d.source === "extended" ? "<small>extended</small>" : "";
            return `<li class="object-row sev-row-${sev}">
                <span class="object-dot" aria-hidden="true"></span>
                <span class="object-name">${esc(friendly(d))}${extra}</span>
                <span class="object-meta">${esc(SIDE[d.spatial] || "Ahead")} · ${esc(d.distance_m)} m</span>
                <span class="object-conf" aria-hidden="true"><span class="object-conf-bar" style="width:${pct}%"></span></span>
                <span class="object-pct"><span class="sr-only">confidence </span>${pct}%</span>
            </li>`;
        }).join("");
    }

    // ─────────────────────────────────────────────
    // Guidance history
    // ─────────────────────────────────────────────
    const history = [];
    const HISTORY_MAX = 20;

    function logGuidance(text, isAlert) {
        text = (text || "").trim();
        if (!text) return;
        if (history.length && history[0].text === text && Date.now() - history[0].at < 4000) return;
        history.unshift({ text, alert: !!isAlert, at: Date.now() });
        if (history.length > HISTORY_MAX) history.pop();
        const list = $("guidance-history"), count = $("history-count");
        if (count) count.textContent = `${history.length} spoken`;
        if (!list) return;
        list.innerHTML = history.map(h => {
            const time = new Date(h.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
            return `<li class="history-item${h.alert ? " is-alert" : ""}"><span class="history-time">${esc(time)}</span><span class="history-text">${esc(h.text)}</span></li>`;
        }).join("");
    }

    // ─────────────────────────────────────────────
    // Public API
    // ─────────────────────────────────────────────
    function update(data, info) {
        if (!data) return;
        setBlips(data.detections);
        updateTelemetry(data, info);
        updateObjectList(data.detections);
        const stage = $("video-wrapper");
        if (stage) stage.classList.toggle("is-hazard", !!data.is_urgent);
    }

    function setLive(on) {
        radar.live = !!on;
        const stage = $("video-wrapper");
        if (stage) {
            stage.classList.toggle("is-live", !!on);
            if (!on) stage.classList.remove("is-hazard");
        }
    }

    function reset() {
        radar.blips.forEach(b => { b.seen = false; });
        const list = $("object-list");
        if (list) list.innerHTML = '<li class="object-empty">Start the assistant to list what the camera sees.</li>';
        const chip = $("low-light-chip");
        if (chip) chip.hidden = true;
        const tile = $("tele-nearest-tile");
        if (tile) tile.classList.remove("is-hazard", "is-caution");
        const near = $("tele-nearest"), nearSub = $("tele-nearest-sub"), objects = $("tele-objects");
        if (near) near.textContent = "—";
        if (nearSub) nearSub.textContent = "nothing close";
        if (objects) objects.textContent = "0";
    }

    document.addEventListener("DOMContentLoaded", () => {
        radar.canvas = $("proximity-radar");
        if (!radar.canvas) return;
        radar.ctx = radar.canvas.getContext("2d");
        readColors();
        // Theme and high-contrast switches change the palette
        const refresh = () => setTimeout(readColors, 0);
        new MutationObserver(refresh).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
        new MutationObserver(refresh).observe(document.body, { attributes: true, attributeFilter: ["class"] });
        requestAnimationFrame(drawRadar);
    });

    window.AMCSPanels = { update, setLive, reset, logGuidance };
})();
