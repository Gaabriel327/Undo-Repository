# flask_app.py — Rekonstruiert aus unserem Chat (Drop-in)

from __future__ import annotations
import os
from datetime import datetime, timedelta, date
import io
import numpy as np

from PIL import Image, ImageDraw, ImageFont
from flask import (
    Flask, request, redirect, url_for, render_template,
    jsonify, make_response, session, g, send_file
)
from flask_login import (
    LoginManager, login_required, login_user, logout_user, current_user
)
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import func, or_
from models import db 

from math import pi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytz
from matplotlib import transforms
# ===== Models (ggf. Pfad anpassen) =====
from models import db, User, Reflection, Group, PromoCode

# ===== Pro/KI-Engine (du hast diese Datei) =====
from pro_feedback_engine import (
    is_pro, ai_generate_feedback, update_streak_and_grant_tokens, ai_generate_group_question
)
import random
from pro_feedback_engine import ai_generate_question

import re
from dotenv import load_dotenv
load_dotenv(override=False)

def _to_second_person(text: str) -> str:
    """Weiche Korrektur in 2. Person (du). Kein perfektes NLP – aber verhindert 'ich'-Ausreißer."""
    if not text:
        return text
    s = text.strip()

    # grobe Ich→Du-Glättungen
    s = re.sub(r"\bIch\b", "Du", s)
    s = re.sub(r"\bich\b", "du", s)
    s = s.replace(" mein ", " dein ").replace(" meine ", " deine ").replace(" meinen ", " deinen ")
    # Wir→Ihr (falls Gruppenfluss fälschlich ins Solo gerät)
    s = s.replace(" wir ", " ihr ").replace(" unser ", " euer ").replace(" uns ", " euch ")

    if not s.endswith("?") and ("?" in s or len(s) < 140):
        # Fragen beenden wir mit '?', kurze Sätze ebenfalls freundlich abrunden
        s = s.rstrip(". ") + "?"
    return s

def _to_plural_second_person(text: str) -> str:
    """Für WeDo: in der Gruppe konsequent 'ihr'/'euer'."""
    if not text:
        return text
    s = text.strip()
    s = s.replace(" wir ", " ihr ").replace(" unser ", " euer ").replace(" uns ", " euch ")
    # Falls KI mal 'du' gebaut hat, minimal in 'ihr' drehen (sehr vorsichtig)
    s = re.sub(r"\bDu\b", "Ihr", s)
    s = re.sub(r"\bdu\b", "ihr", s)
    return s

def _load_font(size: int) -> ImageFont.FreeTypeFont:
    """Versucht system fonts; fallback auf default Bitmap-Font."""
    candidates = [
        # macOS SF Pro (variabel je OS-Version)
        "/System/Library/Fonts/SFNS.ttf",
        "/System/Library/Fonts/SFNSRounded.ttf",
        "/System/Library/Fonts/HelveticaNeueDeskInterface.ttc",
        # DejaVu (Linux/Brew)
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/Library/Fonts/Arial.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except Exception:
            pass
    # Fallback (immer vorhanden, aber nicht hübsch)
    return ImageFont.load_default()

def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, max_width: int) -> list[str]:
    """Bricht Text so um, dass er in max_width passt."""
    lines = []
    for raw_line in (text or "").splitlines() or [""]:
        words = raw_line.split(" ")
        line = ""
        for w in words:
            test = (line + " " + w).strip()
            if draw.textlength(test, font=font) <= max_width:
                line = test
            else:
                if line:
                    lines.append(line)
                line = w
        lines.append(line)
    return lines

import re

# Optional: du hast ähnliche Listen schon – wir nutzen sie hier mit
_ACTION_WORDS = (
    "starten", "beginnen", "anpacken", "umsetzen", "planen", "entscheiden",
    "Schritt", "konkret", "heute", "morgen", "Woche", "Ziel", "Zeitfenster"
)

def _quality_tokens(answer: str) -> int:
    """
    Vergibt 0–3 Tokens basierend auf Antwort-Qualität.
    Ziel: 1 Token fast immer (bei solider Antwort), 2 oft, 3 selten.

    Heuristik:
      - Wortzahl: >= 20 → Basis 1 Token
      - „Reichtum“: Wortvielfalt / Wiederholungen (Richness)
      - Zukunft/Handlung: _ACTION_WORDS / _future greifen
      - Kohärenz: >=2 Sätze und vernünftige Satzlängen
      - 2 Tokens: ab ~40 Wörtern + (Richness und Action)
      - 3 Tokens: ab ~60 Wörtern + (Richness und Action und Kohärenz)
    """
    a = (answer or "").strip()
    if not a:
        return 0

    # Wörter & Sätze
    words = re.findall(r"\w+", a, flags=re.UNICODE)
    wc = len(words)
    sentences = re.split(r"[.!?]\s+", a)
    sentences = [s for s in sentences if s.strip()]
    sc = len(sentences)

    # Basistoken ab 40 Wörtern
    has_base = wc >= 40

    # Richness: Anteil einzigartiger Wörter
    unique = len(set(w.lower() for w in words))
    richness = (unique / max(1, wc))  # 0..1
    is_rich = richness >= 0.45  # relativ großzügig, aber nicht trivial

    # Zukunft/Handlung
    lower = a.lower()
    action_hits = sum(1 for w in _ACTION_WORDS) + 0
    action_hits = sum(1 for w in _ACTION_WORDS if w.lower() in lower)
    future_hits = sum(1 for w in _future if w.lower() in lower) if "_future" in globals() else 0
    has_action = (action_hits + future_hits) >= 2

    # Kohärenz: >1 Satz, mittlere Satzlänge plausibel
    avg_len = wc / max(1, sc)
    coherent = (sc >= 2) and (6 <= avg_len <= 35)

    # Schwellen:
    # 1 Token: fast sicher ab 40 Wörtern
    if not has_base:
        return 0

    tokens = 1

    # 2 Tokens: ab 90+ Wörtern, wenn „reich“ und „Handlung/Zukunft“ sichtbar
    if wc >= 90 and is_rich and has_action:
        tokens = 2

    # 3 Tokens: ab 180+ Wörtern und alle drei Bedingungen
    if wc >= 180 and is_rich and has_action and coherent:
        tokens = 3

    return tokens

# ===== Zeitzone & Helper (inline) =====
import pytz
APP_TZ = pytz.timezone("Europe/Berlin")

def today_bounds_utc(tz=APP_TZ):
    """(now_local, start_utc, end_utc) für den heutigen Lokaltag."""
    now_local = datetime.now(tz)
    start_local = tz.localize(datetime(now_local.year, now_local.month, now_local.day, 0, 0, 0))
    end_local = start_local + timedelta(days=1)
    return now_local, start_local.astimezone(pytz.UTC), end_local.astimezone(pytz.UTC)

def _user_answered_solo_today(user_id: int, mode: str) -> bool:
    """Prüft, ob der User heute bereits die Solo-Frage im gegebenen Modus beantwortet hat."""
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    exists = (Reflection.query
              .filter_by(user_id=user_id, category="solo", mode=mode)
              .filter(Reflection.timestamp >= start_utc,
                      Reflection.timestamp < end_utc)
              .first())
    return exists is not None

def user_groups(user_id):
    """Alle Gruppen eines Users (Besitzer ODER Mitglied, robust gematcht)."""
    from models import Group
    uid_s = str(user_id)
    q = Group.query.filter(
        (Group.created_by == uid_s) |
        (Group.group_members == uid_s) |
        (Group.group_members.like(f"{uid_s},%")) |
        (Group.group_members.like(f"%,{uid_s},%")) |
        (Group.group_members.like(f"%,{uid_s}"))
    )
    return q.all()

def create_app():
    app = Flask(__name__, instance_relative_config=True)

    # Basis-Konfiguration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-change-me')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
    app.config['SESSION_COOKIE_SECURE'] = False

    # SQLite-Fallback, wenn keine DB-URL gesetzt
    os.makedirs(app.instance_path, exist_ok=True)
    db_path = os.path.join(app.instance_path, "undo.db")
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv(
        'DATABASE_URL',
        f"sqlite:///{db_path}"
    )

    # Datenbank an App binden
    db.init_app(app)

    # Tabellen erstellen (nur beim ersten Start)
    with app.app_context():
        db.create_all()

    return app

# WSGI-Einstiegspunkt
app = create_app()

@app.get("/healthz")
def healthz():
    try:
        db.session.execute(db.text("SELECT 1"))
        return "ok", 200
    except Exception as e:
        return f"db error: {e}", 500

# Tabellen nur auf Befehl anlegen, nicht beim Import!
@app.cli.command("init-db")
def init_db():
    db.create_all()
    print("DB initialisiert.") 

# ===== Login =====
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id: str):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# ===== Konstanten =====
NUDGE_INTERVAL_DAYS = 7

# =========================
# Sprache
# =========================
@app.before_request
def inject_lang():
    if current_user.is_authenticated and getattr(current_user, "language", None):
        g.lang = current_user.language
        session["lang"] = current_user.language
    else:
        g.lang = session.get("lang", "de")

@app.post("/settings/language", endpoint="settings_language")
@login_required
def settings_language():
    lang = (request.form.get("language") or request.form.get("lang") or "de").strip().lower()
    if lang not in ("de", "en"):
        lang = "de"
    current_user.language = lang
    db.session.commit()
    session["lang"] = lang
    return redirect(request.referrer or url_for("index"))

# ---- i18n (very light) ----
TRANSLATIONS = {
    "de": {
        "nav.home": "Home",
        "nav.reflections": "Reflections",
        "nav.profile": "Profile",
        "cta.todays_undo": "TODAY'S UNDO",
        "cta.extra_q": "Extra-Frage (1 Token)",
        "streak.days": "Tage Streak",
        "groups.create": "Gruppe erstellen",
        "groups.overview": "Gruppen-Übersicht",
        "progress.title": "Dein Fortschritt",
    },
    "en": {
        "nav.home": "Home",
        "nav.reflections": "Reflections",
        "nav.profile": "Profile",
        "cta.todays_undo": "TODAY’S UNDO",
        "cta.extra_q": "Extra question (1 token)",
        "streak.days": "days streak",
        "groups.create": "Create group",
        "groups.overview": "Group overview",
        "progress.title": "Your progress",
    }
}

@app.context_processor
def inject_t():
    def t(key):
        lang = getattr(g, "lang", "de")
        return TRANSLATIONS.get(lang, TRANSLATIONS["de"]).get(key, key)
    return {"t": t}
# Tokens
# ---- Helper: nächstes Streak-Ziel für Banner in feedback.html ----
def _next_reward_info(user):
    """
    Nächste Belohnungsstufe relativ zur *aktuellen* Streak.
    Deine Streak-Logik: Tag 3 → +1, Tag 5 → +2, Tag 7 → +3, dann Reset auf 0.
    Gibt dict oder None zurück.
    """
    s = int(getattr(user, "streak", 0) or 0)

    # Falls gerade Tag 7 belohnt wurde, setzt du die Streak in update_streak... auf 0.
    # Dann ist als nächstes wieder Tag 3 dran.
    milestones = [(3, 1), (5, 2), (7, 3)]

    for day, tokens in milestones:
        if s < day:
            return {
                "day": day,
                "tokens": tokens,
                "remaining": day - s
            }
    # s >= 7 (oder gleich danach wieder 0) → nächste Stufe ist Tag 3
    return {"day": 3, "tokens": 1, "remaining": 3 - (s % 7)}
@app.get("/share/card/<int:rid>.png", endpoint="share_card_png")
@login_required
def share_card_png(rid: int):
    # Reflection laden
    r = Reflection.query.get_or_404(rid)
    if r.user_id != current_user.id:
        return "Nicht erlaubt", 403

    # Canvas
    W, H = 1200, 628  # Social Card Format
    M = 80            # Außen-Margin
    BG = (255, 255, 255)
    FG = (10, 10, 10)
    SUB = (106, 106, 106)
    ACCENT = (0, 0, 0)

    img = Image.new("RGB", (W, H), BG)
    draw = ImageDraw.Draw(img)

    # Fonts
    f_title = _load_font(36)
    f_q     = _load_font(30)
    f_a     = _load_font(28)
    f_brand = _load_font(24)

    # Titelzeile
    title = "UNDO · Reflection"
    draw.text((M, M), title, fill=SUB, font=f_title)

    # Frage
    q_y = M + 60
    q_maxw = W - 2*M
    q_lines = _wrap_text(draw, r.question or "", f_q, q_maxw)
    for i, line in enumerate(q_lines):
        draw.text((M, q_y + i*42), line, fill=FG, font=f_q)

    # Antwort-Block
    a_y = q_y + len(q_lines)*42 + 28
    draw.rectangle([M, a_y, W-M, a_y+4], fill=ACCENT)  # dünne Trennlinie
    a_y += 26

    answer = (r.answer or "").strip()
    if not answer:
        answer = "—"

    a_lines = _wrap_text(draw, answer, f_a, q_maxw)
    for i, line in enumerate(a_lines):
        draw.text((M, a_y + i*40), line, fill=FG, font=f_a)

    # Footer/Brand
    brand_text = "undo.app"
    bt_w = draw.textlength(brand_text, font=f_brand)
    draw.text((W - M - bt_w, H - M - 10), brand_text, fill=SUB, font=f_brand)

    # Output → Bytes
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    buf.seek(0)
    return send_file(buf, mimetype="image/png")

@app.get("/tokens", endpoint="tokens_info")
@login_required
def tokens_info():
    # Optional: Preise o.ä. aus ENV lesen (falls du später kaufst)
    price_eur = os.getenv("TOKEN_PRICE_EUR")
    return render_template("tokens.html", price_eur=price_eur)

# =========================
# Nudge/Snooze Motiv/Chance
# =========================
def _nudge_resp(target_url: str):
    now_local, _, _ = today_bounds_utc(APP_TZ)
    today_local = now_local.date()
    snooze_until = (today_local + timedelta(days=NUDGE_INTERVAL_DAYS)).isoformat()
    resp = make_response(redirect(target_url))
    resp.set_cookie(
        "motive_snooze_until", snooze_until,
        max_age=60*60*24*(NUDGE_INTERVAL_DAYS+1),
        httponly=True, samesite="Lax"
    )
    return resp

def _nudge_db_mark_today():
    try:
        now_local, _, _ = today_bounds_utc(APP_TZ)
        current_user.last_motive_check = now_local.date().isoformat()
        db.session.commit()
    except Exception:
        db.session.rollback()

# --- Helpers für "heute schon beantwortet?" ---

def _user_answered_solo_today(user_id: int, mode: str) -> bool:
    """Prüft, ob der User heute bereits die Solo-Frage im gegebenen Modus beantwortet hat."""
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    exists = (Reflection.query
              .filter_by(user_id=user_id, category="solo", mode=mode)
              .filter(Reflection.timestamp >= start_utc,
                      Reflection.timestamp < end_utc)
              .first())
    return exists is not None


def _user_answered_group_today(user_id: int, group_id: str | int, mode: str) -> bool:
    """Prüft, ob der User heute bereits für diese Gruppe (WeDo) im Modus geantwortet hat."""
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    exists = (Reflection.query
              .filter_by(user_id=user_id, category="wedo", subcategory=str(group_id), mode=mode)
              .filter(Reflection.timestamp >= start_utc,
                      Reflection.timestamp < end_utc)
              .first())
    return exists is not None

@app.get("/progress", endpoint="progress")
@login_required
def progress():
    # Radar-Chart für User (letzte 30 Tage)
    radar_url = url_for("radar_user_png", days=30)
    return render_template("progress.html", radar_url=radar_url)

def _to_db_bounds(start_utc, end_utc):
    """Falls deine DB naive UTC-Datetimes speichert, machen wir die Bounds naiv."""
    return start_utc.replace(tzinfo=None), end_utc.replace(tzinfo=None)


# ===== Radar config =====
RADAR_AXES = [
    "Selbstbild",
    "Emotionale Intelligenz",
    "Entscheidungsmuster",
    "Perspektivwechsel",
    "Kreativität & Vision",
    "Zukunft",
]

def _bounds_utc(days, tz):
    now_local, _, _ = today_bounds_utc(tz)
    start_local = now_local - timedelta(days=days)
    return start_local.astimezone(pytz.UTC), now_local.astimezone(pytz.UTC)

def _to_db_bounds(start_utc, end_utc):
    """Falls deine DB naive UTC-Datetimes speichert, machen wir die Bounds naiv."""
    return start_utc.replace(tzinfo=None), end_utc.replace(tzinfo=None)

def _collect_refs(user_id, start_utc, end_utc, *, category=None, subcategory=None):
    start_db, end_db = _to_db_bounds(start_utc, end_utc)
    q = Reflection.query.filter(
        Reflection.user_id == user_id,
        Reflection.timestamp >= start_db,
        Reflection.timestamp <  end_db
    )
    if category:
        q = q.filter(Reflection.category == category)
    if subcategory is not None:
        q = q.filter(Reflection.subcategory == str(subcategory))
    return q.order_by(Reflection.timestamp.asc()).all()

_feelings = ("froh", "dankbar", "ruhig", "gelassen", "stolz", "traurig",
             "wütend", "ängstlich", "unsicher", "entspannt", "zuversichtlich")
_future   = ("heute", "morgen", "bald", "nächste", "planen", "vorhaben",
             "Start", "beginnen", "Ziel", "Schritt", "Woche")

_feelings = ("froh", "dankbar", "ruhig", "gelassen", "stolz", "traurig",
             "wütend", "ängstlich", "unsicher", "entspannt", "zuversichtlich")
_future   = ("heute", "morgen", "bald", "nächste", "planen", "vorhaben",
             "start", "beginnen", "ziel", "schritt", "woche")

def _safe_len(s: str) -> int:
    return len((s or "").strip())

def _compute_six_scores(refs) -> dict[str, float]:
    """
    Gibt Scores als Dict {Axis: float 0..1} zurück – genau für RADAR_AXES.
    Robust, heuristisch. Keine externen Felder nötig.
    """
    if not refs:
        base = 0.35
        return {axis: base for axis in RADAR_AXES}

    answers  = [(r.answer or "").strip() for r in refs]
    answersL = [a.lower() for a in answers]
    modes    = [getattr(r, "mode", None) for r in refs]

    # 0 Selbstbild – Frequenz (Tage mit Eintrag / Fenster) + mittlere Länge
    days_map = {}
    for r in refs:
        d = r.timestamp.date()
        days_map[d] = days_map.get(d, 0) + 1
    day_keys = sorted(days_map.keys())
    window_days = max(7, (day_keys[-1] - day_keys[0]).days + 1)
    freq = min(1.0, len(day_keys) / window_days)

    avg_len = sum(_safe_len(a) for a in answers) / max(1, len(answers))
    len_norm = min(1.0, avg_len / 350.0)
    v0 = (freq * 0.6 + len_norm * 0.4)

    # 1 Emotionale Intelligenz – einfache Gefühlswörter
    feel_hits = sum(any(w in a for w in _feelings) for a in answersL)
    v1 = min(1.0, feel_hits / max(1, len(answers)) * 1.2)

    # 2 Entscheidungsmuster – Morgen/Abend-Balance + Kürze
    m = sum(1 for x in modes if x == "morning")
    e = sum(1 for x in modes if x == "evening")
    balance = 1.0 - abs(m - e) / max(1.0, (m + e))
    short_ratio = sum(1 for a in answers if _safe_len(a) <= 220) / max(1, len(answers))
    v2 = max(0.0, min(1.0, balance * 0.6 + short_ratio * 0.4))

    # 3 Perspektivwechsel – Wortvielfalt
    tokens = re.findall(r"[a-zäöüß]+", " ".join(answersL))
    if tokens:
        unique = len(set(tokens))
        total  = len(tokens)
        v3 = min(1.0, (unique / total) * 4.0)
    else:
        v3 = 0.35

    # 4 Kreativität & Vision – Anteil seltener Wörter (nicht Top10)
    if tokens:
        from collections import Counter
        cnt = Counter(tokens)
        common = set([w for w, _ in cnt.most_common(10)])
        rare_tokens = [w for w in tokens if w not in common]
        v4 = min(1.0, len(rare_tokens) / max(1, len(tokens)) * 2.0)
    else:
        v4 = 0.35

    # 5 Zukunft – Zukunfts-/Handlungswörter
    fut_hits = sum(any(w in a for w in _future) for a in answersL)
    v5 = min(1.0, fut_hits / max(1, len(answers)) * 1.5)

    vals = [max(0.05, min(1.0, v)) for v in [v0, v1, v2, v3, v4, v5]]
    return {axis: val for axis, val in zip(RADAR_AXES, vals)}

RADAR_LABEL_TWEAKS = {
    # Beispiele – passe nach Bedarf an:
    "Entscheidungsmuster": {"dr": 0.06, "dtheta": 0.01, "ha": "left"},
    "Selbstbild": {"dr": -0.10, "ha": "center"},
    "Kreativität & Vision": {"dr": 0.02},
    "Zukunft": {"dr": 0.03, "ha": "right"},
    "Perspektivwechsel": {"dr": 0.02, "ha": "center"},
    "Emotionale Intelligenz": {"dr": 0.04},
}

def _render_radar(scores_by_axis: dict[str, float],
                  axes: list[str],
                  title: str | None = None) -> io.BytesIO:
    labels = list(axes)
    values = [max(0.0, min(1.0, float(scores_by_axis.get(k, 0.0)))) for k in labels]

    n = len(labels)
    if n < 3:
        while len(labels) < 3:
            labels.append(f"Axis {len(labels)+1}")
            values.append(0.0)
        n = len(labels)

    # Polygon schließen
    angles = np.linspace(0, 2*np.pi, n, endpoint=False).tolist()
    angles_closed = angles + angles[:1]
    values_closed = values + values[:1]

    fig = plt.figure(figsize=(6.6, 5.8), dpi=160)
    ax = plt.subplot(111, polar=True)

        # keine Grad-Labels (0°,45°, …)
    ax.set_xticks([])          # entfernt die Winkel-Tick-Labels
    ax.set_thetagrids([])      # extra-sicher: keine Gridraster-Labels

    # --- Layout ---
    ax.set_theta_offset(np.pi/2)      # Start oben
    ax.set_theta_direction(-1)        # Uhrzeigersinn

    max_r = 1.20                      # <-- MEHR RAUM nach außen
    ax.set_ylim(0.0, max_r)

    # dezente Ringe bis 1.0
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels([])            # keine Zahlen
    ax.grid(True, linewidth=0.7, alpha=0.22)

    # Standard-Rand aus und eigener Außenkreis bei r=1.0
    for spine in ax.spines.values():
        spine.set_visible(False)
    outer = plt.Circle((0, 0), 1.0, transform=ax.transData._b,
                       fill=False, linewidth=1.1, alpha=0.55)
    ax.add_artist(outer)

    # feine Speichen
    for a in angles:
        ax.plot([a, a], [0, 1.0], linewidth=0.6, color="#000000", alpha=0.15)

    # --- Daten ---
    ax.plot(angles_closed, values_closed, linewidth=1.3, color="#111111")
    ax.fill(angles_closed, values_closed, alpha=0.12, color="#111111")

    # Titel optional
    if title:
        ax.set_title(title, va="bottom", fontweight="bold", fontsize=12, pad=12)

    # --- Labels OUTSIDE ---
    radius_label = 1.22  # Grundabstand (kannst du global „enger/weiter“ machen)

    for ang, lab in zip(angles, labels):
    # Basis-Ausrichtung abhängig vom Quadranten
        cosv = np.cos(ang)
        if -0.15 < cosv < 0.15:
            ha_base = "center"    # oben/unten
        elif cosv > 0:
            ha_base = "left"      # rechte Seite
        else:
            ha_base = "right"     # linke Seite

    # Individuelle Tweaks laden (falls vorhanden)
        t = RADAR_LABEL_TWEAKS.get(lab, {})
        dr = float(t.get("dr", 0.0))             # radialer Offset
        dtheta = float(t.get("dtheta", 0.0))     # Winkel-Offset (Radiant)
        ha = str(t.get("ha", ha_base))           # Ausrichtung überschreiben?

    # finale Position (leicht außerhalb des Außenkreises)
        r = radius_label + dr
        theta = ang + dtheta

    # Sicherheits-Clamps, damit nix abgeschnitten wird
        r = max(1.05, min(1.35, r))

        ax.text(theta, r, lab,
                ha=ha, va="center",
                fontsize=11, fontweight="bold",
                clip_on=False, zorder=10)

    # genug Rand, damit nichts abgeschnitten wird
    fig.subplots_adjust(top=0.9, bottom=0.05, left=0.05, right=0.95)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", transparent=True, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    buf.seek(0)
    return buf

RADAR_AXES = [
    "Selbstbild",
    "Emotionale Intelligenz",
    "Entscheidungsmuster",
    "Perspektivwechsel",
    "Kreativität & Vision",
    "Zukunft",
]

# ——— USER-RADAR ———
@app.get("/radar/user.png", endpoint="radar_user_png")
@login_required
def radar_user_png():
    try:
        days = request.args.get("days", default=30, type=int)
        start_utc, end_utc = _bounds_utc(days, APP_TZ)
        refs = _collect_refs(current_user.id, start_utc, end_utc)
        vals = _compute_six_scores(refs)
        buf = _render_radar(vals, RADAR_AXES, None)  # <— no inner title
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        print("[radar_user_png] ERROR:", e)
        fig = plt.figure(figsize=(5, 1.6)); ax = plt.gca(); ax.axis("off")
        ax.text(0.5, 0.5, "Radar nicht verfügbar", ha="center", va="center", fontsize=14)
        tmp = io.BytesIO(); fig.savefig(tmp, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig); tmp.seek(0)
        return send_file(tmp, mimetype="image/png")


# ——— GRUPPEN-RADAR ———
@app.get("/radar/group/<group_id>.png", endpoint="radar_group_png")
@login_required
def radar_group_png(group_id):
    try:
        days = request.args.get("days", default=30, type=int)
        start_utc, end_utc = _bounds_utc(days, APP_TZ)
        refs = _collect_refs(current_user.id, start_utc, end_utc,
                             category="wedo", subcategory=str(group_id))
        vals = _compute_six_scores(refs)
        buf = _render_radar(vals, RADAR_AXES, None)  # <— no inner title
        return send_file(buf, mimetype="image/png")
        return send_file(buf, mimetype="image/png")
    except Exception as e:
        print("[radar_group_png] ERROR:", e)
        fig = plt.figure(figsize=(5, 1.6)); ax = plt.gca(); ax.axis("off")
        ax.text(0.5, 0.5, "Radar nicht verfügbar", ha="center", va="center", fontsize=14)
        tmp = io.BytesIO(); fig.savefig(tmp, format="png", dpi=160, bbox_inches="tight")
        plt.close(fig); tmp.seek(0)
        return send_file(tmp, mimetype="image/png")
# ===== /Radar =====

@app.get("/nudge/motive/review-now", endpoint="motive_review_now")
@login_required
def motive_review_now():
    _nudge_db_mark_today()
    return _nudge_resp(url_for("personal"))  # oder deine Einstellungsseite

@app.get("/nudge/motive/snooze", endpoint="motive_snooze")
@login_required
def motive_snooze():
    _nudge_db_mark_today()
    return _nudge_resp(request.referrer or url_for("index"))

# Alt: dein ursprünglicher POST-ACK
@app.post("/nudge/motive-ack", endpoint="motive_ack")
@login_required
def motive_ack():
    _nudge_db_mark_today()
    return _nudge_resp(request.referrer or url_for("index"))

def _compute_motive_due(user):
    now_local, _, _ = today_bounds_utc(APP_TZ)
    today_local = now_local.date()
    cookie_snooze = request.cookies.get("motive_snooze_until")
    if cookie_snooze:
        try:
            return today_local > date.fromisoformat(cookie_snooze)
        except Exception:
            pass
    last_value = getattr(user, "last_motive_check", None)
    last_date = None
    if last_value:
        if isinstance(last_value, datetime):
            last_date = last_value.date()
        elif isinstance(last_value, date):
            last_date = last_value
        else:
            s = str(last_value)
            try: last_date = date.fromisoformat(s)
            except Exception:
                try: last_date = datetime.fromisoformat(s).date()
                except Exception: last_date = None
    return (last_date is None) or ((today_local - (last_date or today_local)).days >= NUDGE_INTERVAL_DAYS)

# =========================
# Registrierung mit Promo
# =========================
@app.route("/register", methods=["GET", "POST"], endpoint="register")
def register_route():
    if request.method == "POST":
        username  = (request.form.get("username") or "").strip()
        email     = (request.form.get("email") or "").strip().lower()
        password  = request.form.get("password") or ""
        promo_in  = (request.form.get("promo_code") or "").strip()

        if not username or not email or not password or not promo_in:
            return render_template("register.html",
                                   username=username, email=email, promo_code=promo_in,
                                   error="Bitte alle Felder inkl. Promo-Code ausfüllen."), 200

        if User.query.filter((User.username == username) | (User.email == email)).first():
            return render_template("register.html",
                                   username=username, email=email, promo_code=promo_in,
                                   error="Benutzername oder E-Mail vergeben."), 200

        norm_in = promo_in.lower().replace("-", "").replace(" ", "")
        pc = (PromoCode.query
              .filter(func.replace(func.replace(func.lower(PromoCode.code), "-", ""), " ", "") == norm_in)
              .first())

        if not pc or not pc.active:
            return render_template("register.html",
                                   username=username, email=email, promo_code=promo_in,
                                   error="Ungültiger Promo-Code."), 200
        if pc.expires_at and pc.expires_at < datetime.utcnow():
            return render_template("register.html",
                                   username=username, email=email, promo_code=promo_in,
                                   error="Promo-Code abgelaufen."), 200
        if pc.max_uses is not None and int(pc.used_count or 0) >= pc.max_uses:
            return render_template("register.html",
                                   username=username, email=email, promo_code=promo_in,
                                   error="Promo-Code bereits aufgebraucht."), 200

        user = User(
            username=username,
            email=email,
            password=generate_password_hash(password),
            subscription="pro",
            pro_until=datetime.utcnow() + timedelta(days=pc.duration_days or 30),
            promo_locked=True,
            promo_code=pc,
        )
        db.session.add(user)
        pc.used_count = int(pc.used_count or 0) + 1
        db.session.commit()

        login_user(user, remember=True, fresh=True)
        return redirect(url_for("personal", onboarding=1))

    return render_template("register.html")

# PROMO CODES



# =========================
# >>> DEINE ORIGINAL-INDEX-ROUTE <<<
# =========================
# --- Gruß-Helfer (auf Modulebene!) ---
def make_greeting(now_local, lang: str = "de") -> str:
    # 03–10:59 = Morgen, 11–17:59 = Tag, 18–02:59 = Abend
    h = now_local.hour
    if 3 <= h < 11:
        return "Guten Morgen" if lang == "de" else "Good morning"
    elif 11 <= h < 18:
        return "Guten Tag" if lang == "de" else "Good afternoon"
    else:
        return "Guten Abend" if lang == "de" else "Good evening"


@app.route("/", endpoint="index")
@login_required
def index():
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    today_local = now_local.date()
    hour = now_local.hour

    lang = getattr(current_user, "language", "de")
    greeting = make_greeting(now_local, lang)

    # Abendmodus: ab 18 Uhr bis < 03 Uhr
    evening_open = (hour >= 18) or (hour < 3)
    current_mode = "evening" if evening_open else "morning"

    # Heutige Reflections …
    todays = (Reflection.query
              .filter(Reflection.user_id == current_user.id,
                      Reflection.timestamp >= start_utc,
                      Reflection.timestamp < end_utc)
              .all())
    morning_done = any(r.mode == "morning" for r in todays)
    evening_done = any(r.mode == "evening" for r in todays)

    disable_today_button = False
    show_extra = False
    next_unlock_label = ""
    if not evening_open:
        if morning_done:
            disable_today_button = True
            show_extra = True
            next_unlock_label = "Nächstes UNDO verfügbar um 18:00"
    else:
        if evening_done:
            disable_today_button = True
            show_extra = True
            next_unlock_label = "Nächstes UNDO verfügbar morgen 06:00"

    # Nudge-Snooze …
    cookie_snooze = request.cookies.get("motive_snooze_until")
    if cookie_snooze:
        try:
            snooze_date = date.fromisoformat(cookie_snooze)
            motive_due = today_local > snooze_date
        except Exception:
            motive_due = True
    else:
        last_value = getattr(current_user, "last_motive_check", None)
        last_date = None
        if last_value:
            if isinstance(last_value, datetime):
                last_date = last_value.date()
            elif isinstance(last_value, date):
                last_date = last_value
            else:
                try:
                    last_date = date.fromisoformat(str(last_value))
                except Exception:
                    try:
                        last_date = datetime.fromisoformat(str(last_value)).date()
                    except Exception:
                        last_date = None
        motive_due = (last_date is None) or ((today_local - (last_date or today_local)).days >= NUDGE_INTERVAL_DAYS)

    # WeDo-Infos vorbereiten (robuster)
    # --- WeDo-Infos vorbereiten ---
    groups = user_groups(current_user.id) or []
    wedo_count = len(groups)
    first_group_id = groups[0].id if wedo_count >= 1 else None
    single_group_already = False
    if first_group_id:
        single_group_already = _user_answered_group_today(current_user.id, first_group_id, current_mode)

    wedo_info = {
        "has_groups": wedo_count > 0,
        "count": wedo_count,
        "single_group_id": first_group_id if wedo_count == 1 else None,
        "single_group_already": single_group_already if wedo_count == 1 else False,
    }

    return render_template(
        "index.html",
        username=current_user.username,
        greeting=greeting,              # falls du den Gruß nutzt
        current_mode=current_mode,
        disable_today_button=disable_today_button,
        show_extra=show_extra,
        next_unlock_label=next_unlock_label,
        wedo=wedo_info,                 # <<<< WICHTIG
        motive_due=motive_due
    )     
    
# =========================
# Prompt (Solo)
# =========================
def _user_answered_solo_today(user_id: int, mode: str) -> bool:
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    return (Reflection.query
            .filter(Reflection.user_id == user_id,
                    Reflection.category == "solo",
                    Reflection.mode == mode,
                    Reflection.timestamp >= start_utc,
                    Reflection.timestamp < end_utc)
            .first()) is not None

@app.route("/prompt", methods=["GET", "POST"], endpoint="prompt")
@login_required
def prompt():
    import random, re

    # --- Helpers ---
    def to_int(x, default=0):
        try:
            return int(str(x).strip())
        except Exception:
            return default

    def enforce_du(txt: str) -> str:
        t = txt or ""
        # sehr einfache Normalisierung – falls KI „ich“ benutzt
        t = re.sub(r"\bIch\b", "Du", t)
        t = re.sub(r"\bich\b", "du", t)
        t = t.replace(" mein ", " dein ").replace(" Meine ", " Deine ").replace(" meine ", " deine ")
        return t

    now_local, _, _ = today_bounds_utc(APP_TZ)
    current_mode = "evening" if (now_local.hour >= 18 or now_local.hour < 3) else "morning"

    # ===== Extra-Status =====
    # GET kann extra=1 setzen. Wir puffern die Info in der Session.
    if request.method == "GET":
        if request.args.get("extra") == "1":
            session["pending_extra"] = True
        elif request.args.get("extra") == "0":
            session.pop("pending_extra", None)

    is_extra = (request.values.get("extra") == "1") or bool(session.get("pending_extra"))
    user_tokens = to_int(getattr(current_user, "tokens", 0))

    # ===== POST: Antwort speichern =====
    if request.method == "POST":
        # 1) Frage/Antwort einsammeln (mit Session-Fallback für Frage)
        answer = (request.form.get("answer") or "").strip()
        shown_text = (request.form.get("question_text") or "").strip()
        if not shown_text:
            shown_text = (session.get("prompt_q_text") or "").strip()

        if not answer or not shown_text:
            # Zur Sicherheit Frage neu holen
            session.pop("prompt_q_text", None)
            return redirect(url_for("prompt", extra=("1" if is_extra else None)))

        # 2) Normale Tagesfrage nur 1× pro Tag/Modus
        if (not is_extra) and _user_answered_solo_today(current_user.id, current_mode):
            last = (Reflection.query
                    .filter_by(user_id=current_user.id, category="solo", mode=current_mode)
                    .order_by(Reflection.timestamp.desc())
                    .first())
            session.pop("pending_extra", None)
            session.pop("prompt_q_text", None)
            if last:
                return redirect(url_for("feedback_view", rid=last.id, compact=1))
            return redirect(url_for("index"))

        # 3) Extra kostet IMMER 1 Token – harter Server-Check
        if is_extra:
            user_tokens = to_int(getattr(current_user, "tokens", 0))
            if user_tokens < 1:
                # Kein Zugriff – Flag entfernen und zurück
                session.pop("pending_extra", None)
                session.pop("prompt_q_text", None)
                return redirect(url_for("index"))
            # Abbuchen & commit VOR dem Speichern der Reflection
            current_user.tokens = user_tokens - 1
            try:
                db.session.commit()
            except Exception:
                db.session.rollback()
                # Bei Fehler Extra abbrechen
                session.pop("pending_extra", None)
                session.pop("prompt_q_text", None)
                return redirect(url_for("index"))
            # Verbrauchtes Extra-Flag löschen
            session.pop("pending_extra", None)

        # 4) Feedback erzeugen (KI/Fallback), in Du-Form
        try:
            if is_pro(current_user):
                fb_text = ai_generate_feedback(
                    shown_text, answer,
                    current_user.motive or "", current_user.chance or "",
                    mode=current_mode
                )
            else:
                fb_text = random.choice([
                    "Klein halten und sichtbar machen – heute zählt ein kleiner, klarer Schritt.",
                    "Ein kurzes Zeitfenster heute reicht – 10 Minuten können den Knoten lösen.",
                    "Greif dir eine Sache, die leicht bleibt, und zieh sie leise durch."
                ])
        except Exception:
            fb_text = "Klein halten und sichtbar machen – heute zählt ein kleiner, klarer Schritt."
        fb_text = enforce_du(fb_text)

        # 5) Reflection speichern
        r = Reflection(
            user_id=current_user.id,
            question=shown_text,
            answer=answer,
            feedback=fb_text,
            category="solo",
            subcategory=None,
            mode=current_mode,
            timestamp=datetime.utcnow(),
        )
        db.session.add(r)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            # Wenn das Speichern fehlschlägt, nicht stillschweigend verlieren:
            return redirect(url_for("index"))

        # --- QUALITÄTSTOKENS (1–3) ---
        earned_quality = _quality_tokens(answer)
        if earned_quality > 0:
            current_user.tokens = int(current_user.tokens or 0) + earned_quality
            db.session.commit()

        # --- STREAK-BELONUNG (0/1/2/3 je nach 3/5/7) ---
        before = int(current_user.tokens or 0)
        update_streak_and_grant_tokens(db, current_user)
        after = int(current_user.tokens or 0)
        earned_streak = max(0, after - before)

        # Gesamt an Feedback-View übergeben
        total_earned = earned_quality + earned_streak
        return redirect(url_for("feedback_view", rid=r.id, compact=1, earned=total_earned))

    # ===== GET: Frage anzeigen =====
    already_today = _user_answered_solo_today(current_user.id, current_mode)
    if already_today and (not is_extra):
        # Tagesfrage bereits beantwortet → zurück zur Startseite
        session.pop("prompt_q_text", None)
        return redirect(url_for("index"))

    # GET-Guard: Extra ohne Token gar nicht erst anzeigen
    if is_extra and user_tokens < 1:
        session.pop("pending_extra", None)
        session.pop("prompt_q_text", None)
        return redirect(url_for("index"))

    # KI-Frage (Du-Perspektive) mit Fallback
    q = None
    try:
        q = ai_generate_question(
            motive=current_user.motive or "",
            chance=current_user.chance or "",
            mode=current_mode,
            seed_texts=[
                "Worum geht es dir heute wirklich – in einem Satz?",
                "Welche kleine Entscheidung hebt deinen Tag leise an?",
                "Was braucht dich heute für 10 ruhige Minuten?",
                "Womit beginnst du, damit es sich stimmig anfühlt?"
            ]
        )
    except Exception:
        q = None

    if not q:
        pool_morning = [
            "Worauf richtest du heute deinen Blick – ganz bewusst?",
            "Welche kleine Entscheidung macht deinen Tag heute leichter?",
            "Was ist heute eine Sache, die du sichtbar machst?",
            "Womit beginnst du, damit es sich stimmig anfühlt?"
        ]
        pool_evening = [
            "Was tat heute gut – und wovon möchtest du mehr?",
            "Welche kleine Sache hat heute einen echten Unterschied gemacht?",
            "Wofür willst du dir heute Anerkennung geben?",
            "Was darf heute leise zu Ende gehen?"
        ]
        base = pool_evening if current_mode == "evening" else pool_morning
        q = random.choice(base)

    q = enforce_du(q)
    if not q.endswith("?"):
        q = q.rstrip(". ") + "?"

    # Die angezeigte Frage für den POST sicherheitshalber puffern
    session["prompt_q_text"] = q

    # Rendern
    return render_template(
        "prompt.html",
        mode=current_mode,
        is_extra=is_extra,
        question_text=q,      # dein Template nutzt 'question_text'
        display_text=q,
        already_today=already_today
    )
# =========================
# Feedback / Reflection
# =========================
@app.get("/feedback/<int:rid>", endpoint="feedback_view")
@login_required
def feedback_view(rid):
    r = Reflection.query.get_or_404(rid)
    if r.user_id != current_user.id:
        return "Nicht erlaubt", 403

    compact = request.args.get("compact", type=int) == 1
    earned  = request.args.get("earned",  type=int)

    def _next_reward_info(streak: int | None):
        s = int(streak or 0)
        # Beispiel-Staffel: Tag 1=+1, Tag 3=+2, Tag 7=+3
        checkpoints = [(1, 1), (3, 2), (7, 3)]
        for day, tokens in checkpoints:
            if s < day:
                return {"day": day, "tokens": tokens, "remaining": day - s}
        return None

    next_reward = _next_reward_info(getattr(current_user, "streak", 0))

    return render_template(
        "feedback.html",
        r=r,
        compact=compact,
        earned=earned,
        next_reward=next_reward
    )

# ---- Reflection: mit & ohne rid unter EINEM Endpoint ----
@app.get("/reflection", defaults={"rid": None}, endpoint="reflection")
@app.get("/reflection/<int:rid>", endpoint="reflection")
@login_required
def reflection_route(rid):
    if rid is None:
        # zur letzten eigenen Reflexion oder Home
        last = (Reflection.query
                .filter_by(user_id=current_user.id)
                .order_by(Reflection.timestamp.desc())
                .first())
        if last:
            return redirect(url_for("feedback_view", rid=last.id), code=307)
        return redirect(url_for("index"))

    # vorhandene Query-Params weiterreichen (z.B. earned=…, compact=…)
    params = request.args.to_dict(flat=True)
    # falls nicht mitgegeben, compact erzwingen
    params.setdefault("compact", 1)

    return redirect(url_for("feedback_view", rid=rid, **params), code=307)

@app.get("/reflections", endpoint="reflections")
@login_required
def reflections_list():
    reflections = (Reflection.query
                   .filter_by(user_id=current_user.id)
                   .order_by(Reflection.timestamp.desc())
                   .all())
    return render_template("reflections.html", reflections=reflections)

# macht 'user' in ALLEN Templates verfügbar -> verweist auf current_user
@app.context_processor
def inject_user():
    return {"user": current_user}

# /profile – echte Seite, kein Redirect
@app.route("/profile", methods=["GET", "POST"], endpoint="profile")
@login_required
def profile():
    # Falls du später Profileinstellungen speichern willst, kannst du hier POST verarbeiten.
    # if request.method == "POST":
    #     ... (Updates) ...
    #     db.session.commit()
    return render_template("profile.html", user=current_user)

# Share-Link
@app.post("/api/share_link/<int:rid>")
@login_required
def api_share_link(rid):
    r = Reflection.query.get_or_404(rid)
    if r.user_id != current_user.id:
        return jsonify({"error": "forbidden"}), 403
    link = url_for("feedback_view", rid=rid, _external=True)
    return jsonify({"url": link})
# =========================
# WEDO / Groups
# =========================
# ---------------------------------------------
# Hilfen für CSV-Mitgliedschaften + Limit 3
# ---------------------------------------------
def _csv_has_member(csv_str: str | None, uid_s: str) -> bool:
    if not csv_str:
        return False
    parts = [p.strip() for p in csv_str.split(",") if p.strip()]
    return uid_s in parts

def _user_total_groups_count(uid) -> int:
    uid_s = str(uid)
    owned = Group.query.filter(Group.created_by == uid_s).count()
    others = Group.query.filter(Group.created_by != uid_s).all()
    member_count = sum(1 for g in others if _csv_has_member(g.group_members, uid_s))
    return owned + member_count

def _user_groups(uid):
    """Alle Gruppen (Owner oder Mitglied) – robust via CSV-Scan."""
    uid_s = str(uid)
    owned = Group.query.filter(Group.created_by == uid_s).all()
    others = Group.query.filter(Group.created_by != uid_s).all()
    member_of = [g for g in others if _csv_has_member(g.group_members, uid_s)]
    return owned + member_of


# ---------------------------------------------
# „ihr“-Form erzwingen – sehr defensives Cleanup
# ---------------------------------------------
def _to_plural_second_person(text: str) -> str:
    if not text:
        return text
    t = text.strip()

    # Häufige Fehlstarts des Modells „Ich/Wir/...“ → auf „ihr/du“ drehen
    # (für WeDo nehmen wir bevorzugt „ihr“)
    repls = [
        (r"\bIch\b", "Ihr"),
        (r"\bich\b", "ihr"),
        (r"\bWir\b", "Ihr"),
        (r"\bwir\b", "ihr"),
        (r"\bUns\b", "Euch"),
        (r"\buns\b", "euch"),
        (r"\bEuer\b", "Euer"),  # passt schon
        (r"\beuer\b", "euer"),
        (r"\bDu\b", "Ihr"),
        (r"\bdu\b", "ihr"),
        (r"\bDir\b", "Euch"),
        (r"\bdir\b", "euch"),
        (r"\bDich\b", "Euch"),
        (r"\bdich\b", "euch"),
    ]

    import re as _re
    for pat, sub in repls:
        t = _re.sub(pat, sub, t)

    # Fragezeichen sicherstellen
    if not t.endswith("?"):
        t = t.rstrip(". ") + "?"
    return t


# ---------------------------------------------
# WeDo: Liste
# ---------------------------------------------
# ---------------------------------------------
# Hilfen für CSV-Mitgliedschaften + Limit 3
# ---------------------------------------------
def _csv_has_member(csv_str: str | None, uid_s: str) -> bool:
    if not csv_str:
        return False
    parts = [p.strip() for p in csv_str.split(",") if p.strip()]
    return uid_s in parts

def _user_total_groups_count(uid) -> int:
    uid_s = str(uid)
    owned = Group.query.filter(Group.created_by == uid_s).count()
    others = Group.query.filter(Group.created_by != uid_s).all()
    member_count = sum(1 for g in others if _csv_has_member(g.group_members, uid_s))
    return owned + member_count

def _user_groups(uid):
    """Alle Gruppen (Owner oder Mitglied) – robust via CSV-Scan."""
    uid_s = str(uid)
    owned = Group.query.filter(Group.created_by == uid_s).all()
    others = Group.query.filter(Group.created_by != uid_s).all()
    member_of = [g for g in others if _csv_has_member(g.group_members, uid_s)]
    return owned + member_of


# ---------------------------------------------
# „ihr“-Form erzwingen – sehr defensives Cleanup
# ---------------------------------------------
def _to_plural_second_person(text: str) -> str:
    if not text:
        return text
    t = text.strip()
    import re as _re
    repls = [
        (r"\bIch\b", "Ihr"), (r"\bich\b", "ihr"),
        (r"\bWir\b", "Ihr"), (r"\bwir\b", "ihr"),
        (r"\bUns\b", "Euch"), (r"\buns\b", "euch"),
        (r"\bDu\b", "Ihr"),  (r"\bdu\b", "ihr"),
        (r"\bDir\b", "Euch"), (r"\bdir\b", "euch"),
        (r"\bDich\b", "Euch"), (r"\bdich\b", "euch"),
    ]
    for pat, sub in repls:
        t = _re.sub(pat, sub, t)
    if not t.endswith("?"):
        t = t.rstrip(". ") + "?"
    return t


# ---------------------------------------------
# WeDo: Liste
# ---------------------------------------------
@app.route("/wedo", methods=["GET"], endpoint="groups_list")
@login_required
def groups_list():
    uid = str(current_user.id)
    groups = Group.query.filter(
        (Group.created_by == uid) | (Group.group_members.like(f"%{uid}%"))
    ).all()

    own_count = Group.query.filter(Group.created_by == uid).count()
    can_create = own_count < 3
    return render_template("group.html", groups=groups, can_create=can_create, own_count=own_count)


# ---------------------------------------------
# WeDo: Gruppe erstellen (max. 3 insgesamt)
# ---------------------------------------------
@app.post("/wedo/create", endpoint="create_group")
@login_required
def create_group():
    total = _user_total_groups_count(current_user.id)
    if total >= 3:
        return "Du kannst in maximal 3 Gruppen gleichzeitig sein.", 400

    name = (request.form.get("name") or "Meine Gruppe").strip()
    # WICHTIG: Hier NICHT künstlich int erzwingen; deine DB hat UUIDs.
    g = Group(name=name, created_by=str(current_user.id), group_members="")
    db.session.add(g)
    db.session.commit()
    return redirect(url_for("group_open", group_id=g.id))


# ---------------------------------------------
# WeDo: Gruppe öffnen (Dashboard)
#  (group_id als STRING – UUIDs funktionieren jetzt)
# ---------------------------------------------
@app.route("/wedo/<group_id>", methods=["GET"], endpoint="group_overview")
@login_required
def group_overview(group_id):
    g = Group.query.get_or_404(group_id)
    now_local, _, _ = today_bounds_utc(APP_TZ)
    mode = "evening" if (now_local.hour >= 18) else "morning"
    already_today = _user_answered_group_today(current_user.id, g.id, mode)
    radar_url = url_for("radar_group_png", group_id=g.id, days=30)

    # WICHTIG: hier NICHT group.html rendern, sondern deine Detail-Template-Seite!
    return render_template(
        "group_overview.html",
        group=g,
        mode=mode,
        already_today=already_today,
        radar_url=radar_url
    )


# ---------------------------------------------
# WeDo: Prompt (GET: Frage zeigen, POST: speichern)
#  - Extra-Frage ist deaktiviert (immer False)
#  - group_id als STRING – keine int-Casts mehr
# ---------------------------------------------
@app.route("/wedo/<group_id>/prompt", methods=["GET", "POST"], endpoint="group_prompt")
@login_required
def group_prompt(group_id):
    g = Group.query.get_or_404(group_id)

    # Berechtigung: Owner oder Mitglied
    uid_s = str(current_user.id)
    is_member = (uid_s == str(g.created_by)) or _csv_has_member(g.group_members, uid_s)
    if not is_member:
        return "Nicht erlaubt", 403

    now_local, _, _ = today_bounds_utc(APP_TZ)
    current_mode = "evening" if now_local.hour >= 18 else "morning"

    # Extra ist bei WeDo deaktiviert
    is_extra = False

    if request.method == "POST":
        answer = (request.form.get("answer") or "").strip()
        shown_text = (request.form.get("question_text") or "").strip()
        if not answer or not shown_text:
            return redirect(url_for("group_prompt", group_id=group_id))

        # Nur 1× pro Tag/Modus
        if _user_answered_group_today(current_user.id, group_id, current_mode):
            return redirect(url_for("group_open", group_id=group_id))

        # Feedback
        if is_pro(current_user):
            fb = ai_generate_feedback(
                shown_text, answer, current_user.motive, current_user.chance, mode=current_mode
            ).replace("UNDO-Impuls:", "WeDo-Impuls:")
        else:
            fb = "Klar und machbar halten – ein kleiner Schritt, den ihr heute sichtbar macht."

        # Speichern
        r = Reflection(
            user_id=current_user.id,
            question=shown_text,
            answer=answer,
            feedback=fb,
            category="wedo",
            subcategory=str(g.id),
            mode=current_mode,
            timestamp=datetime.utcnow(),
        )
        db.session.add(r)
        db.session.commit()

                # --- QUALITÄTSTOKENS (1–3) ---
        earned_quality = _quality_tokens(answer)
        if earned_quality > 0:
            current_user.tokens = int(current_user.tokens or 0) + earned_quality
            db.session.commit()

        # --- STREAK-BELONUNG ---
        before = int(current_user.tokens or 0)
        update_streak_and_grant_tokens(db, current_user)
        after = int(current_user.tokens or 0)
        earned_streak = max(0, after - before)

        total_earned = earned_quality + earned_streak
        return redirect(url_for("feedback_view", rid=r.id, compact=1, earned=total_earned))

    # ---------- GET: Frage generieren ----------
    try:
        q = ai_generate_group_question(
            motive=getattr(g, "motive", None),
            chance=getattr(g, "chance", None),
            mode=current_mode
        )
    except Exception:
        q = None

    if not q:
        q = "Womit wollt ihr heute beginnen, damit es sich leicht in Bewegung anfühlt?"
    q = _to_plural_second_person(q)

    return render_template(
        "group_prompt.html",
        group=g,
        mode=current_mode,
        question=q,
        question_text=q,  # wichtig: Hidden-Feld im Formular
    )


# ---------------------------------------------
# WeDo: Gruppe bearbeiten
# ---------------------------------------------
@app.route("/wedo/<group_id>/edit", methods=["GET", "POST"], endpoint="group_edit")
@login_required
def group_edit(group_id):
    g = Group.query.get_or_404(group_id)
    if str(current_user.id) != (g.created_by or ""):
        return "Nur Ersteller können bearbeiten.", 403
    if request.method == "POST":
        g.name = (request.form.get("name") or g.name).strip()
        # Optional: motive/chance übernehmen, wenn du Felder hast
        # g.motive = (request.form.get("motive") or g.motive or "").strip()
        # g.chance = (request.form.get("chance") or g.chance or "").strip()
        db.session.commit()
        return redirect(url_for("group_open", group_id=g.id))
    return render_template("group_edit.html", group=g)

#WeDo: Gruppe öffnen
@app.get("/wedo/<group_id>/open", endpoint="group_open")
@login_required
def group_open_alias(group_id):
    return redirect(url_for("group_overview", group_id=group_id), code=307)

# ---------------------------------------------
# WeDo: Gruppe löschen (nur Owner)
# ---------------------------------------------
@app.route("/wedo/<group_id>/delete", methods=["POST"], endpoint="group_delete")
@login_required
def group_delete(group_id):
    g = Group.query.get_or_404(group_id)
    if str(current_user.id) != (g.created_by or ""):
        return "Nur Ersteller können löschen.", 403
    db.session.delete(g)
    db.session.commit()
    return redirect(url_for("groups_list"))

# =========================
# Stubs für fehlende Seiten
# =========================
# --- Settings: Ansicht ---
@app.get("/settings", endpoint="settings")
@login_required
def settings_view():
    # Optional: kleine Statusmeldung via Query-Param
    msg = request.args.get("msg")
    return render_template("settings.html", user=current_user, msg=msg)

# --- Settings: Pro kündigen ---
@app.post("/settings/cancel_pro", endpoint="settings_cancel_pro")
@login_required
def settings_cancel_pro():
    # Promo-gebundenes Pro darf nicht gekündigt werden
    if getattr(current_user, "promo_locked", False):
        return redirect(url_for(
            "settings",
            msg="Dein Pro ist an einen Promo-Code gebunden und kann hier nicht beendet werden."
        ))

    # Pro beenden
    current_user.subscription = "free"
    current_user.pro_until = None
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return redirect(url_for("settings", msg="Konnte Pro nicht beenden. Bitte später erneut versuchen."))

    return redirect(url_for("settings", msg="Pro wurde beendet."))

@app.route("/personal", methods=["GET", "POST"], endpoint="personal")
@login_required
def personal():
    if request.method == "POST":
        # Felder robust einsammeln (nutze, was du im Template hast)
        current_user.first_name = (request.form.get("first_name") or "").strip() or current_user.first_name
        current_user.motive     = (request.form.get("motive") or request.form.get("why") or "").strip()
        current_user.chance     = (request.form.get("chance") or request.form.get("goal") or "").strip()
        lang = (request.form.get("language") or request.form.get("lang") or "").strip().lower()
        if lang in ("de", "en"):
            current_user.language = lang

        # Profil als ausgefüllt markieren (falls du das Feld hast)
        try:
            current_user.profile_completed = True
        except Exception:
            pass

        # Optional: letzter Motiv-Check auf heute (hilft beim Nudge)
        try:
            from datetime import datetime
            current_user.last_motive_check = datetime.utcnow().date().isoformat()
        except Exception:
            pass

        # Speichern
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()
            return render_template("personal.html", user=current_user, error="Konnte nicht speichern."), 400

        # Weiter zur gewünschten Seite (z. B. Home)
        return redirect(url_for("index"))

    # GET – Seite anzeigen
    show_hint = (request.args.get("onboarding") == "1") and not bool(getattr(current_user, "profile_completed", False))
    return render_template("personal.html", user=current_user)

@app.route("/buy-tokens", endpoint="buy_tokens")
@login_required
def buy_tokens():
    return render_template("buy_tokens.html") if os.path.exists("templates/buy_tokens.html") else "Token-Kauf", 200

# =========================
# Minimaler Login/Logout (Platzhalter)
# =========================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = (request.form.get("email") or "").strip().lower()
        password = request.form.get("password") or ""
        u = User.query.filter_by(email=email).first()
        if u and check_password_hash(u.password, password):
            login_user(u, remember=True, fresh=True)
            return redirect(url_for("index"))
        return render_template("login.html", error="Login fehlgeschlagen.")
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

@app.cli.command("migrate-groups-columns")
def migrate_groups_columns():
    """
    Fügt fehlende Spalten zu 'groups' hinzu (idempotent).
    Nutzt SQLite ALTER TABLE ... ADD COLUMN.
    """
    from sqlalchemy import text
    from models import db
    with db.engine.begin() as conn:
        cols = {row[1] for row in conn.execute(text("PRAGMA table_info('groups');")).fetchall()}

        def add(col, ddl):
            if col not in cols:
                conn.execute(text(f"ALTER TABLE groups ADD COLUMN {ddl};"))
                print(f"Added column: {col}")

        add("created_at",   "created_at DATETIME")
        add("motive",       "motive TEXT")
        add("chance",       "chance TEXT")
        add("last_q_text",  "last_q_text TEXT")
        add("last_q_day",   "last_q_day TEXT")
        add("last_q_mode",  "last_q_mode TEXT")

        print("Migration done.")

# =========================
# Start
# =========================
if __name__ == "__main__":
    app.run(debug=True)