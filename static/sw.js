const CACHE_NAME = "undo-cache-v1";
const OFFLINE_URL = "/static/offline.html";
const PRECACHE = [
  "/",
  "/static/style.css",
  OFFLINE_URL,
  "/static/icons/icon-192.png",
  "/static/icons/icon-512.png",
  "/static/manifest.webmanifest"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(PRECACHE))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("fetch", (event) => {
  const req = event.request;

  // Navigationsanfragen: Network-first, Fallback offline
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req).catch(() => caches.match(OFFLINE_URL))
    );
    return;
  }

  // Statische Assets: Cache-first
  event.respondWith(
    caches.match(req).then((hit) => hit || fetch(req))
  );
});