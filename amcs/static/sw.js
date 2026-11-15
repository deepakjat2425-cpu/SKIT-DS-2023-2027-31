/**
 * AMCS service worker (served at /sw.js so it controls every page).
 *
 * Strategy: network first, cache as fallback. Users always get the latest
 * UI when online, and the app shell still opens when the connection drops.
 * API calls and cached speech audio are never intercepted.
 */

const CACHE_NAME = "amcs-v3.2.0";
const APP_SHELL = [
  "/",
  "/ocr",
  "/static/css/style.css",
  "/static/js/ui.js",
  "/static/js/panels.js",
  "/static/js/main.js",
  "/static/js/caregiver.js",
  "/static/icons/icon.svg",
  "/static/manifest.json"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => Promise.all(APP_SHELL.map((url) => cache.add(url).catch(() => null))))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(keys.filter((k) => k !== CACHE_NAME).map((k) => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  const url = new URL(req.url);

  if (req.method !== "GET" || url.origin !== self.location.origin) return;
  if (url.pathname.startsWith("/api/") || url.pathname.startsWith("/static/audio_cache/")) return;

  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.ok && res.type === "basic") {
          const copy = res.clone();
          // Store without the ?v= query so the shell matches offline
          caches.open(CACHE_NAME).then((cache) => cache.put(url.pathname, copy)).catch(() => {});
        }
        return res;
      })
      .catch(() =>
        caches.match(url.pathname)
          .then((cached) => cached || (req.mode === "navigate" ? caches.match("/") : null))
          .then((res) => res || Response.error())
      )
  );
});
