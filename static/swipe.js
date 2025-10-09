// Minimaler Swipe-Handler für horizontales Wischen (DOM-getrieben, kein Jinja)
(() => {
  const SWIPE_THRESHOLD = 60;     // Mindestweg in px
  const MAX_ANGLE_DEG   = 30;     // wie „gerade“ der Swipe sein muss

  let startX = 0, startY = 0, moved = false, ignore = false;

  function isFormEl(el) {
    return el && el.closest && el.closest('input, textarea, select, button, a, [contenteditable="true"]');
  }

  function angleOK(dx, dy) {
    const adx = Math.abs(dx), ady = Math.abs(dy);
    if (adx === 0) return false;
    const angle = Math.atan2(ady, adx) * 180 / Math.PI;
    return angle <= MAX_ANGLE_DEG; // nahezu horizontal
  }

  function goto(url) { if (url) window.location.assign(url); }

  // Tabs dynamisch aus der Navbar lesen
  const navLinks = Array.from(document.querySelectorAll('nav.undo-nav a'));
  let tabs = navLinks.map(a => a.href);

  // Fallback, falls Navbar nicht vorhanden:
  if (!tabs.length) {
    const origin = window.location.origin;
    tabs = ["/", "/reflections", "/profile"].map(p => origin + p);
  }

  function normalizePath(href) {
    try { return new URL(href, window.location.origin).pathname; }
    catch { return null; }
  }

  function tabIndexOf(href) {
    const path = normalizePath(href);
    if (!path) return -1;
    return tabs.findIndex(t => normalizePath(t) === path);
  }

  function nextTab() {
    const i = tabIndexOf(window.location.href);
    if (i >= 0) goto(tabs[(i + 1) % tabs.length]);
  }
  function prevTab() {
    const i = tabIndexOf(window.location.href);
    if (i >= 0) goto(tabs[(i - 1 + tabs.length) % tabs.length]);
  }

  const root = document; // global

  root.addEventListener('touchstart', (e) => {
    const t = e.touches[0];
    startX = t.clientX; startY = t.clientY;
    moved = false;
    ignore = !!isFormEl(e.target); // In Formularen/Links nicht reagieren
  }, {passive: true});

  root.addEventListener('touchmove', () => { moved = true; }, {passive: true});

  root.addEventListener('touchend', (e) => {
    if (ignore || !moved) return;
    const t = e.changedTouches[0];
    const dx = t.clientX - startX;
    const dy = t.clientY - startY;
    if (Math.abs(dx) < SWIPE_THRESHOLD) return;
    if (!angleOK(dx, dy)) return;

    if (dx < 0) nextTab();
    else       prevTab();
  }, {passive: true});
})();