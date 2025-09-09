# flask_app.py
import os, uuid, random, re, sqlite3
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager, login_user, logout_user, login_required, current_user
)
from sqlalchemy import text

# --- Models ---
from models import (
    db, User, Reflection, Group,
    Question, UserQuestionHistory, UserCategoryScore, PromoCode
)

# --- Pro / Tokens / KI ---
from pro_feedback_engine import (
    is_pro,
    require_feature_or_charge, FEATURE,
    update_streak_and_grant_tokens,
    ai_generate_feedback,
)

# ===========================
# App/Timezone
# ===========================
APP_TZ = ZoneInfo(os.getenv("APP_TZ", "Europe/Berlin"))
UTC = ZoneInfo("UTC")

# ===========================
# App Setup
# ===========================
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "supersecret")

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "undo.db"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///" + str(DB_PATH)
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
db.init_app(app)

@app.before_first_request
def bootstrap():
    with app.app_context():
        ensure_user_columns()
        ensure_groups_columns()
        db.create_all()
        seed_questions_if_empty()

# ===========================
# DB Column Helpers
# ===========================
def ensure_user_columns():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(user)")
    cols = {row[1] for row in cur.fetchall()}
    def add(sql):
        try:
            cur.execute(sql)
        except Exception:
            pass
    if "first_name" not in cols: add("ALTER TABLE user ADD COLUMN first_name TEXT")
    if "birth_date" not in cols: add("ALTER TABLE user ADD COLUMN birth_date TEXT")
    if "motive" not in cols: add("ALTER TABLE user ADD COLUMN motive TEXT")
    if "chance" not in cols: add("ALTER TABLE user ADD COLUMN chance TEXT")
    if "profile_completed" not in cols: add("ALTER TABLE user ADD COLUMN profile_completed INTEGER DEFAULT 0")
    if "pro_until" not in cols: add("ALTER TABLE user ADD COLUMN pro_until TEXT")
    conn.commit()
    conn.close()

def ensure_groups_columns():
    conn = sqlite3.connect(str(DB_PATH))
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(groups)")
    cols = {row[1] for row in cur.fetchall()}
    def add(sql):
        try:
            cur.execute(sql)
        except Exception:
            pass
    if "group_members" not in cols: add("ALTER TABLE groups ADD COLUMN group_members TEXT")
    if "created_by" not in cols: add("ALTER TABLE groups ADD COLUMN created_by TEXT")
    if "name" not in cols: add("ALTER TABLE groups ADD COLUMN name TEXT")
    conn.commit()
    conn.close()

# ===========================
# Flask-Login
# ===========================
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

@login_manager.user_loader
def load_user(user_id):
    try:
        return User.query.get(int(user_id))
    except Exception:
        return None

# ===========================
# Onboarding-Gate
# ===========================
@app.before_request
def require_profile_completion():
    allowed = {"login", "logout", "register", "personal", "static", "share_reflection"}
    if getattr(current_user, "is_authenticated", False):
        ep = (getattr(request, "endpoint", None) or "").split(".")[-1]
        if not (current_user.profile_completed or False) and ep not in allowed:
            return redirect(url_for("personal"))

# ===========================
# Registrierung / Login / Logout
# ===========================
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form["username"].strip()
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        if User.query.filter((User.username == username) | (User.email == email)).first():
            return "Benutzername oder E-Mail vergeben.", 400
        user = User(username=username, email=email, password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        login_user(user)
        return redirect(url_for("personal"))
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"].strip().lower()
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for("index"))
        return "E-Mail oder Passwort falsch", 401
    return render_template("login.html")

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("login"))

# ===========================
# Promo-Code einlösen
# ===========================
def _extend_pro_until(user, days: int):
    now = datetime.utcnow()
    # falls pro_until als ISO-Text gespeichert ist:
    base = now
    if user.pro_until:
        try:
            base_dt = datetime.fromisoformat(user.pro_until) if isinstance(user.pro_until, str) else user.pro_until
            if base_dt > now:
                base = base_dt
        except Exception:
            pass
    user.pro_until = (base + timedelta(days=max(1, days))).isoformat()
    user.subscription = "pro"

@app.route("/redeem_code", methods=["POST"])
@login_required
def redeem_code():
    code = (request.form.get("code") or "").strip()
    if not code:
        return "Kein Code übergeben.", 400

    promo = PromoCode.query.filter_by(code=code).first()
    if not promo:
        return "Ungültiger Code.", 400

    now = datetime.utcnow()
    if promo.expires_at and promo.expires_at < now:
        return "Dieser Code ist abgelaufen.", 400
    if (promo.uses_left or 0) < 1:
        return "Dieser Code wurde bereits verwendet.", 400

    plan = (promo.plan or "pro").lower()
    if plan == "pro":
        _extend_pro_until(current_user, int(promo.days or 30))

    grant = int(promo.token_grant or 0)
    if grant > 0:
        current_user.tokens = int(current_user.tokens or 0) + grant

    promo.uses_left = int(promo.uses_left or 0) - 1

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return "Serverfehler beim Einlösen des Codes.", 500

    return redirect(url_for("profile"))

# ===========================
# Personal (Onboarding)
# ===========================
@app.route("/personal", methods=["GET", "POST"])
@login_required
def personal():
    if request.method == "POST":
        current_user.first_name = request.form.get("first_name")
        current_user.birth_date = request.form.get("birth_date")
        current_user.motive = request.form.get("motive")
        current_user.chance = request.form.get("chance")
        current_user.profile_completed = True
        db.session.commit()
        return redirect(url_for("index"))
    return render_template("personal.html", user=current_user)

# ===========================
# Zeit-Utils
# ===========================
def today_bounds_utc(tz: ZoneInfo):
    """
    Liefert (now_local, start_utc_naiv, end_utc_naiv) für den aktuellen lokalen Tag.
    In der DB speichern wir naive UTC-Timestamps, daher tzinfo entfernen.
    """
    now_local = datetime.now(tz)
    sod_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
    eod_local = sod_local + timedelta(days=1)
    sod_utc = sod_local.astimezone(UTC).replace(tzinfo=None)
    eod_utc = eod_local.astimezone(UTC).replace(tzinfo=None)
    return now_local, sod_utc, eod_utc

# ===========================
# Basic Pages
# ===========================
@app.route("/")
@login_required
def index():
    now_local, start_utc, end_utc = today_bounds_utc(APP_TZ)
    evening_open = now_local.hour >= 18  # ab 18:00 lokal

    todays = (Reflection.query
              .filter(Reflection.user_id == current_user.id,
                      Reflection.timestamp >= start_utc,
                      Reflection.timestamp < end_utc)
              .all())

    morning_done = any(r.mode == "morning" for r in todays)
    evening_done = any(r.mode == "evening" for r in todays)

    # Default-Werte
    current_mode = "morning"
    disable_today_button = False
    show_extra = False
    next_unlock_label = ""

    if not evening_open:
        # Vor 18:00 -> MORNING
        current_mode = "morning"
        if morning_done:
            disable_today_button = True
            show_extra = True
            next_unlock_label = "Nächstes UNDO verfügbar um 18:00"
    else:
        # Ab 18:00 -> EVENING
        current_mode = "evening"
        if evening_done:
            disable_today_button = True
            show_extra = True  # Extra-Frage erlauben
            next_unlock_label = "Nächstes UNDO verfügbar morgen 06:00"

    return render_template(
        "index.html",
        username=current_user.username,
        current_mode=current_mode,
        disable_today_button=disable_today_button,
        show_extra=show_extra,
        next_unlock_label=next_unlock_label,
    )

@app.route("/profile")
@login_required
def profile():
    return render_template("profile.html", user=current_user, is_pro=is_pro(current_user))

@app.route("/settings", methods=["GET"])
@login_required
def settings():
    return render_template("settings.html", user=current_user)

@app.post("/settings/subscribe")
@login_required
def settings_subscribe_pro():
    current_user.subscription = "pro"
    # Optional auch pro_until setzen:
    current_user.pro_until = (datetime.utcnow() + timedelta(days=30)).isoformat()
    db.session.commit()
    return redirect(url_for("settings") + "#pro")

@app.post("/settings/cancel")
@login_required
def settings_cancel_pro():
    current_user.subscription = "free"
    current_user.pro_until = None
    db.session.commit()
    return redirect(url_for("settings") + "#pro")

@app.route("/buy_tokens", methods=["GET", "POST"])
@login_required
def buy_tokens():
    if request.method == "POST":
        amount = int(request.form.get("amount", "0"))
        current_user.tokens = (current_user.tokens or 0) + amount
        db.session.commit()
        return redirect(url_for("profile"))
    return render_template("buy_tokens.html")

# Öffentliche Share-Ansicht (ohne Login)
@app.route("/share/<int:rid>", endpoint="share_reflection")
def share_reflection(rid):
    r = Reflection.query.get_or_404(rid)
    return render_template("share_reflection.html", reflection=r)

# ===========================
# Radar Diagram (8 Kategorien) – mit sanfter Sättigung
# ===========================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import math

@app.route("/progress")
@login_required
def progress():
    label_map = {
        "selbstbild": "Selbstbild & Bewusstsein",
        "emotion":    "Emotionen & Regulation",
        "gewohnheit": "Gewohnheiten & Entscheidungen",
        "beziehung":  "Beziehungen & Empathie",
        "mindset":    "Perspektive & Mindset",
        "vision":     "Kreativität & Vision",
        "zukunft":    "Zukunft & Ziele",
        "koerper":    "Körper & Energie",
    }
    keys = list(label_map.keys())
    labels = list(label_map.values())

    # Versuche echte Scores (0..100)
    scores = []
    has_score = False
    for k in keys:
        sc = UserCategoryScore.query.filter_by(user_id=current_user.id, category=k).first()
        val = sc.score if sc else None
        if val is not None:
            has_score = True
        scores.append(val)

    if has_score:
        values = [int(v or 0) for v in scores]
    else:
        # Sättigende Prozent-Skala basierend auf jüngsten Einträgen
        N_RECENT = 30
        tau = 7.0  # 10 Antworten ~ 76%
        recent = (Reflection.query
                  .filter_by(user_id=current_user.id)
                  .order_by(Reflection.timestamp.desc())
                  .limit(N_RECENT)
                  .all())
        counts = {k: 0 for k in keys}
        for r in recent:
            k = (r.category or "").lower()
            if k in counts:
                counts[k] += 1
        values = []
        for k in keys:
            c = counts[k]
            percent = 100.0 * (1.0 - math.exp(-c / tau))
            values.append(int(round(percent)))

    N = len(values)
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
    values_closed = values + values[:1]
    angles_closed = angles + angles[:1]

    fig, ax = plt.subplots(figsize=(6.8, 6.8), subplot_kw=dict(polar=True))
    ax.set_theta_offset(np.pi/2)
    ax.set_theta_direction(-1)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8)
    ax.grid(True, linewidth=0.6, alpha=0.5)
    ax.plot(angles_closed, values_closed, linewidth=2.2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    ax.scatter(angles, values, s=22)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=9)

    filename = f"static/radar_{current_user.username}.png"
    fig.savefig(filename, bbox_inches="tight", dpi=160)
    plt.close(fig)
    return render_template("progress.html", image_url="/" + filename)

# ===========================
# Groups (WeDo) – Nur Pro
# ===========================
@app.route("/groups", endpoint="groups_page")
@login_required
def groups_page():
    ok, _ = require_feature_or_charge(db, current_user, FEATURE.WEDO)
    if not ok:
        return "WeDo ist nur in UNDO Pro verfügbar.", 403
    uid = str(current_user.id)
    groups = (Group.query
              .filter(Group.group_members.isnot(None))
              .filter(Group.group_members.contains(uid))
              .all())
    return render_template("group.html", groups=groups)

@app.route("/create_group", methods=["POST"])
@login_required
def create_group():
    ok, _ = require_feature_or_charge(db, current_user, FEATURE.WEDO)
    if not ok:
        return "WeDo ist nur in UNDO Pro verfügbar.", 403
    gid = str(uuid.uuid4())
    name = request.form.get("name", f"Gruppe {gid[:4]}")
    g = Group(id=gid, name=name, created_by=str(current_user.id), group_members=str(current_user.id))
    db.session.add(g)
    db.session.commit()
    return redirect(url_for("groups_page"))

@app.route("/join/<group_id>")
@login_required
def join_group(group_id):
    ok, _ = require_feature_or_charge(db, current_user, FEATURE.WEDO)
    if not ok:
        return "WeDo ist nur in UNDO Pro verfügbar.", 403
    g = Group.query.get(group_id)
    if not g:
        return "Gruppe nicht gefunden", 404
    members = (g.group_members or "").split(",") if g.group_members else []
    uid = str(current_user.id)
    if uid not in members:
        members.append(uid)
        g.group_members = ",".join(filter(None, members))
        db.session.commit()
    return redirect(url_for("groups_page"))

# ===========================
# Fragen Engine
# ===========================
CATS = ["selbstbild", "emotion", "gewohnheit", "beziehung", "mindset", "vision", "zukunft", "koerper"]
MODE_WEIGHT = {
    "morning": {"gewohnheit": 3, "zukunft": 3, "koerper": 3, "vision": 2, "selbstbild": 2, "emotion": 2, "mindset": 2, "beziehung": 1},
    "evening": {"selbstbild": 3, "emotion": 3, "mindset": 2, "beziehung": 2, "koerper": 2, "gewohnheit": 1, "vision": 1, "zukunft": 1},
}
AFFINITY = {
    "selbstbild": ["selbstbewusstsein", "stärke", "wert", "vertrauen", "identität"],
    "emotion": ["stress", "ruhe", "gefühle", "angst", "resilienz"],
    "gewohnheit": ["routine", "disziplin", "sport", "produktivität"],
    "beziehung": ["freund", "partner", "familie", "netzwerk", "empathie"],
    "mindset": ["glaubenssatz", "optimismus", "zweifel", "denken"],
    "vision": ["kreativität", "traum", "projekt", "gründung"],
    "zukunft": ["ziel", "karriere", "planung", "studium"],
    "koerper": ["schlaf", "energie", "körper", "ernährung", "bewegung"],
}

def tokenize(text: str):
    return re.findall(r"[a-zäöüß]+", (text or "").lower())

def affinity_weights_for_user(user):
    t = tokenize((user.motive or "") + " " + (user.chance or ""))
    base = {c: 1 for c in CATS}
    for c, keys in AFFINITY.items():
        for k in keys:
            if k in t:
                base[c] += 2
    return base

def need_weights_for_user(uid: int):
    w = {}
    for c in CATS:
        sc = UserCategoryScore.query.filter_by(user_id=uid, category=c).first()
        score = sc.score if sc else 0
        w[c] = max(1, 100 - score)
    return w

def personal_level(uid: int, cat: str):
    sc = UserCategoryScore.query.filter_by(user_id=uid, category=cat).first()
    s = sc.score if sc else 0
    base = max(1, min(5, s // 20 + 1))
    return max(1, min(5, base + random.choice([-1, 0, 0, 1])))

from sqlalchemy import text
from datetime import datetime

from sqlalchemy import text

def select_next_question(
    user,
    mode: str = "morning",
    *,
    exclude_today_same_mode: bool = False,   # für Extra-Fragen wichtig
    tz=APP_TZ,
    recent_days_same_mode: int = 7,
    exclude_last_n_same_mode: int = 3,
):
    """
    Wählt eine nächste Frage:
      - gewichtet nach Motive/Chance (AFFINITY), Need (UserCategoryScore), Mode (morning/evening)
      - schließt BEANTWORTETE Fragen immer aus
      - optional: schließt heute bereits GESTELLTE Fragen im selben Modus aus (für Extra-Fragen)
      - optional: meidet die letzten N gestellten Fragen im selben Modus (Lookback-Fenster)
    """
    now_local, start_utc, end_utc = today_bounds_utc(tz)

    # 1) beantwortete (hart ausschließen)
    answered_ids = {
        h.question_id
        for h in UserQuestionHistory.query.filter_by(user_id=user.id).all()
        if h.answered_at is not None
    }

    # 2) heute gestellte im selben Modus (weiches Exclude)
    asked_today_ids = set()
    if exclude_today_same_mode:
        asked_today_ids = {
            h.question_id
            for h in (UserQuestionHistory.query
                      .filter(UserQuestionHistory.user_id == user.id)
                      .filter(UserQuestionHistory.mode == mode)
                      .filter(UserQuestionHistory.asked_at >= start_utc)
                      .filter(UserQuestionHistory.asked_at < end_utc)
                      .all())
        }

    # 3) letzte N gestellte im selben Modus (weiches Exclude, über Lookback-Fenster)
    recent_cutoff = datetime.utcnow() - timedelta(days=max(1, recent_days_same_mode))
    recent_hist = (UserQuestionHistory.query
                   .filter(UserQuestionHistory.user_id == user.id)
                   .filter(UserQuestionHistory.mode == mode)
                   .filter(UserQuestionHistory.asked_at >= recent_cutoff)
                   .order_by(UserQuestionHistory.asked_at.desc())
                   .limit(max(0, exclude_last_n_same_mode))
                   .all())
    recent_last_ids = {h.question_id for h in recent_hist}

    hard_exclude = set(answered_ids)
    soft_exclude = asked_today_ids | recent_last_ids

    # Gewichtung
    aw = affinity_weights_for_user(user)
    nw = need_weights_for_user(user.id)
    mw = MODE_WEIGHT.get(mode, {c: 1 for c in CATS})
    score_by_cat = {c: aw[c] * nw[c] * mw.get(c, 1) for c in CATS}
    ordered_cats = sorted(CATS, key=lambda c: score_by_cat[c], reverse=True)

    def candidates_for_cat(cat: str, respect_level: bool, respect_mode: bool):
        q = Question.query.filter(Question.category == cat)
        if respect_mode:
            q = q.filter((Question.mode == mode) | (Question.mode == "any"))
        if respect_level:
            lvl = personal_level(user.id, cat)
            q = q.filter(Question.difficulty.between(max(1, lvl - 1), min(5, lvl + 1)))
        items = q.all()
        out = []
        for qq in items:
            if qq.id in hard_exclude:  # nie wieder eine beantwortete
                continue
            if qq.id in soft_exclude:  # vermeide heute/letzte N
                continue
            out.append(qq)
        return out

    # Mehrstufige Suche
    for cat in ordered_cats:
        cand = candidates_for_cat(cat, True, True)
        if cand: return random.choice(cand)
    for cat in ordered_cats:
        cand = candidates_for_cat(cat, False, True)
        if cand: return random.choice(cand)
    for cat in ordered_cats:
        cand = candidates_for_cat(cat, False, False)
        if cand: return random.choice(cand)

    # Fallbacks
    q = Question.query
    if hard_exclude:
        q = q.filter(Question.id.notin_(hard_exclude))
    if soft_exclude:
        q = q.filter(Question.id.notin_(soft_exclude))
    any_unanswered = q.order_by(text("RANDOM()")).first()
    if any_unanswered: return any_unanswered
    return Question.query.order_by(text("RANDOM()")).first()

def generate_feedback(q, answer, user):
    """
    Leichtgewichtiges, regelbasiertes Feedback für Free-User.
    - nutzt Question.suggested_tips (pipe-getrennt)
    - ergänzt 1–2 Qualitätschecks
    """
    # Tipps aus der Frage extrahieren
    tips = []
    if q and getattr(q, "suggested_tips", None):
        tips = [t.strip() for t in q.suggested_tips.split("|") if t.strip()]

    # Heuristiken
    extras = []
    ans_lower = (answer or "").lower()

    # Länge/Signal
    if len(ans_lower) < 40:
        extras.append("Schreibe 1–2 Sätze mehr, um konkreter zu werden.")

    # Zeitanker
    if not any(k in ans_lower for k in ["heute", "morgen", "uhr", "wochentag"]):
        extras.append("Lege eine konkrete Zeit fest (z. B. heute 18:00).")

    # Nutzer-Kontext (motive/chance)
    motive = (getattr(user, "motive", "") or "").lower()
    chance = (getattr(user, "chance", "") or "").lower()
    if "stress" in motive or "stress" in ans_lower or "ruhe" in chance:
        extras.append("Baue eine kleine Ruhe-Routine ein (2–3 Minuten Atem, Stretch, kurzer Walk).")

    final = tips + extras
    return " • ".join(final[:3]) if final else "Gute Richtung – formuliere einen konkreten Schritt."

def update_after_answer(uid: int, qid: int, quality: int = 3):
    hist = UserQuestionHistory.query.filter_by(user_id=uid, question_id=qid).first()
    if not hist:
        hist = UserQuestionHistory(user_id=uid, question_id=qid)
        db.session.add(hist)
    hist.answered_at = datetime.utcnow()
    hist.quality = int(quality or 3)

    q = Question.query.get(qid)
    if q:
        sc = UserCategoryScore.query.filter_by(user_id=uid, category=q.category).first()
        if not sc:
            sc = UserCategoryScore(user_id=uid, category=q.category, score=0)
            db.session.add(sc)
        sc.score = min(100, (sc.score or 0) + int(quality or 3))
        sc.last_seen = datetime.utcnow()

    db.session.commit()

# ===========================
# Prompt
# ===========================
@app.route("/prompt", methods=["GET", "POST"])
@login_required
def prompt():
    # Gemeinsame Zeitlogik
    now_local, _, _ = today_bounds_utc(APP_TZ)
    current_mode = "evening" if now_local.hour >= 18 else "morning"

    # ---------------- GET: Frage anzeigen ----------------
    if request.method == "GET":
        is_extra = request.args.get("extra") == "1"
        if is_extra:
            ok, _ = require_feature_or_charge(db, current_user, FEATURE.EXTRA_QUESTION)
            if not ok:
                return redirect(url_for("buy_tokens"))

        # Frage auswählen (mit Anti-Wiederholungs-Filtern je Modus)
        q = select_next_question(
            current_user,
            mode=current_mode,
            exclude_today_same_mode=is_extra,  # Extra-Frage: heute bereits Gestelltes im selben Modus vermeiden
            tz=APP_TZ,
            recent_days_same_mode=7,
            exclude_last_n_same_mode=3,
        )
        if not q:
            seed_questions_if_empty()
            db.session.commit()
            q = select_next_question(
                current_user,
                mode=current_mode,
                exclude_today_same_mode=is_extra,
                tz=APP_TZ,
                recent_days_same_mode=7,
                exclude_last_n_same_mode=3,
            )

        # Anzeige-Text
        display_text = q.text

        # „gestellt“-Log
        asked = UserQuestionHistory.query.filter_by(
            user_id=current_user.id,
            question_id=q.id
        ).first()
        if not asked:
            db.session.add(
                UserQuestionHistory(
                    user_id=current_user.id,
                    question_id=q.id,
                    mode=current_mode,
                    asked_at=datetime.utcnow(),
                )
            )
            db.session.commit()

        return render_template(
            "prompt.html",
            question=q,
            mode=current_mode,
            display_text=display_text
        )

    # ---------------- POST: Antwort speichern ----------------
    qid_str = (request.form.get("question_id") or "").strip()
    q = Question.query.get(int(qid_str)) if qid_str.isdigit() else None
    if not q:
        asked_row = (UserQuestionHistory.query
                     .filter_by(user_id=current_user.id)
                     .order_by(UserQuestionHistory.asked_at.desc())
                     .first())
        if asked_row:
            q = Question.query.get(asked_row.question_id)
        if not q:
            return redirect(url_for("prompt"))

    answer = (request.form.get("answer") or "").strip()
    if not answer:
        return redirect(url_for("prompt"))

    # Falls du display_text im Formular mitsendest (hidden), sonst q.text
    shown_text = (request.form.get("question_text") or q.text).strip()

    if is_pro(current_user):
        feedback_text = ai_generate_feedback(
            shown_text,
            answer,
            current_user.motive,
            current_user.chance,
            mode=current_mode,
        )
    else:
        feedback_text = generate_feedback(q, answer, current_user)

    r = Reflection(
        user_id=current_user.id,
        question=shown_text,
        answer=answer,
        feedback=feedback_text,
        category=q.category,
        subcategory=q.subcategory,
        mode=current_mode,
        timestamp=datetime.utcnow(),
    )
    db.session.add(r)
    db.session.commit()

    update_after_answer(current_user.id, q.id, quality=3)
    update_streak_and_grant_tokens(db, current_user)

    return redirect(url_for("feedback_view", rid=r.id))

# ===========================
# Feedback-Ansicht & Reflection-Liste
# ===========================
@app.route("/feedback/<int:rid>")
@login_required
def feedback_view(rid):
    r = Reflection.query.get_or_404(rid)
    if r.user_id != current_user.id:
        return "Nicht erlaubt", 403
    return render_template("feedback.html", question=r.question, answer=r.answer, feedback=r.feedback, rid=r.id)

@app.route("/reflection", endpoint="reflection")
@login_required
def reflection():
    reflections = (Reflection.query
                   .filter_by(user_id=current_user.id)
                   .order_by(Reflection.timestamp.desc())
                   .all())
    return render_template("reflection.html", reflections=reflections)

# ===========================
# Seed Questions
# ===========================
# ===========================
# Seed Questions (sauber: morning vs evening)
# ===========================
def seed_questions_if_empty():
    if Question.query.first():
        return

    seed = [
        # ---------- MORNING ----------
        ("selbstbild", None, 1, "morning",
         "Worauf möchtest du heute deinen Fokus legen?",
         "Zeitfenster|Konkreter erster Schritt"),
        ("emotion", None, 1, "morning",
         "Welche Stimmung nimmst du mit in den Tag – und was braucht sie?",
         "Emotion benennen|Bedürfnis ableiten"),
        ("gewohnheit", None, 1, "morning",
         "Welche kleine Gewohnheit machst du heute sicher (≤ 2 Minuten)?",
         "Kontext setzen|Startsignal"),
        ("beziehung", None, 1, "morning",
         "Wen möchtest du heute bewusst wertschätzen – wie konkret?",
         "Person|Form der Wertschätzung"),
        ("mindset", None, 1, "morning",
         "Welche Annahme willst du heute testen statt einfach zu glauben?",
         "Annahme formulieren|Mini-Experiment"),
        ("vision", None, 1, "morning",
         "Welcher kleinste Schritt bringt dich heute deiner Vision näher?",
         "Vision in 1 Satz|Schritt ≤15 Min"),
        ("zukunft", None, 1, "morning",
         "Welches Ergebnis möchtest du bis 18:00 erreicht haben?",
         "Konkret|Messbar|Realistisch"),
        ("koerper", None, 1, "morning",
         "Was gibt dir heute Morgen Energie – und wie sicherst du dir 10 Minuten dafür?",
         "Konkret planen|Mini-Schritt"),

        # ---------- EVENING ----------
        ("selbstbild", None, 2, "evening",
         "Worauf warst du heute an dir selbst stolz – ganz konkret?",
         "Beispiel nennen|Gefühl benennen"),
        ("emotion", None, 2, "evening",
         "Wann hast du heute auf dich gehört statt nur zu funktionieren?",
         "Moment erkennen|Körper-Signal"),
        ("gewohnheit", None, 2, "evening",
         "Welche Entscheidung war heute gut genug statt perfekt – warum?",
         "Entscheidung|Lerneffekt"),
        ("beziehung", None, 2, "evening",
         "Gab es heute einen kleinen Moment echter Nähe?",
         "Was passiert|Warum bedeutsam"),
        ("mindset", None, 2, "evening",
         "Wo hat sich heute deine Perspektive geändert?",
         "Vorher|Auslöser|Nachher"),
        ("vision", None, 2, "evening",
         "Welcher Gedanke hat heute deine Kreativität angestoßen?",
         "Auslöser|Idee skizzieren|Nächster Versuch"),
        ("zukunft", None, 2, "evening",
         "Was war heute ein Schritt, für den dir dein zukünftiges Ich dankt?",
         "Schritt|Auswirkung|Wiederholen?"),
        ("koerper", None, 2, "evening",
         "Woran hast du heute gemerkt, dass dein Körper gut versorgt war?",
         "Signal benennen|Situation"),
    ]

    for cat, sub, diff, mode, text, tips in seed:
        db.session.add(Question(
            category=cat, subcategory=sub, difficulty=diff,
            mode=mode, text=text, suggested_tips=tips
        ))
    db.session.commit()

# ===========================
#  Start
# ===========================
if __name__ == "__main__":
    with app.app_context():
        ensure_user_columns()
        ensure_groups_columns()
        db.create_all()
        seed_questions_if_empty()
        print("✅ DB ok, Seeds vorhanden. Starte Server…")
    app.run(host="127.0.0.1", port=5050, debug=True)