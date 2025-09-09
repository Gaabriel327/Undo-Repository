# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin

db = SQLAlchemy()

# ---------------------------
# User & Reflections
# ---------------------------
class User(db.Model, UserMixin):
    __tablename__ = "user"

    id = db.Column(db.Integer, primary_key=True)

    # Account
    username = db.Column(db.String(80), unique=True, nullable=False, index=True)
    email = db.Column(db.String(120), unique=True, nullable=False, index=True)
    password = db.Column(db.String(200), nullable=False)

    # Abo/Rewards
    subscription = db.Column(db.String(20))
    tokens = db.Column(db.Integer, default=0)
    streak = db.Column(db.Integer, default=0)
    streak_count = db.Column(db.Integer, default=0)

    # Aktivität
    last_active = db.Column(db.DateTime, default=datetime.utcnow)
    last_reflection_date = db.Column(db.DateTime)

    # Persönliche Daten / Onboarding
    first_name = db.Column(db.String(100))
    birth_date = db.Column(db.String(20))
    motive = db.Column(db.Text)   # Beweggrund
    chance = db.Column(db.Text)   # Aussicht/Ziel
    profile_completed = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"<User {self.id}:{self.username}>"

class Reflection(db.Model):
    __tablename__ = "reflection"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("user.id"), nullable=False, index=True)

    # Frage/Antwort/Meta
    question = db.Column(db.Text)
    answer = db.Column(db.Text, nullable=False)
    feedback = db.Column(db.Text)

    # Kategorisierung
    category = db.Column(db.String(100))
    subcategory = db.Column(db.String(100))
    mode = db.Column(db.String(20))  # 'morning' | 'evening' | optional

    # Verlauf
    parent_id = db.Column(db.Integer)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow, index=True)

    user = db.relationship("User", backref=db.backref("reflections", lazy=True))

    def __repr__(self):
        return f"<Reflection {self.id} user={self.user_id} cat={self.category}>"

# ---------------------------
# Gruppen (WeDo)
# ---------------------------
class Group(db.Model):
    __tablename__ = "groups"

    id = db.Column(db.String, primary_key=True)
    name = db.Column(db.String, nullable=False, index=True)
    created_by = db.Column(db.String, nullable=False, index=True)
    # CSV der User-IDs: "1,7,12"
    group_members = db.Column(db.String, nullable=True)

    def __repr__(self):
        return f"<Group {self.id} {self.name}>"

# ---------------------------
# Fragen-Engine (regelbasiert)
# ---------------------------
class Question(db.Model):
    __tablename__ = "question"

    id = db.Column(db.Integer, primary_key=True)

    # 7 Kategorien (Keys): 'selbstbild','emotion','gewohnheit','beziehung','mindset','vision','zukunft'
    category = db.Column(db.String(50), nullable=False, index=True)
    subcategory = db.Column(db.String(100))

    # Schwierigkeitsstufe 1..5
    difficulty = db.Column(db.Integer, default=1, index=True)

    # 'morning' | 'evening' | 'any'
    mode = db.Column(db.String(10), default="any", index=True)

    # Fragetext (einzigartig)
    text = db.Column(db.Text, nullable=False, unique=True)

    # Pipe- oder JSON-String mit 2–5 Hinweisen (regelbasiertes Feedback)
    suggested_tips = db.Column(db.Text)

    def __repr__(self):
        return f"<Question {self.id} [{self.category}/{self.mode}] d={self.difficulty}>"

class UserCategoryScore(db.Model):
    __tablename__ = "user_category_score"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    category = db.Column(db.String(50), nullable=False, index=True)  # s.o.
    score = db.Column(db.Integer, default=0)  # 0..100 Kompetenz/Fortschritt je Kategorie
    last_seen = db.Column(db.DateTime)

    __table_args__ = (
        db.UniqueConstraint("user_id", "category", name="uq_user_category_once"),
    )

    def __repr__(self):
        return f"<UserCategoryScore u={self.user_id} {self.category}:{self.score}>"

class UserQuestionHistory(db.Model):
    __tablename__ = "user_question_history"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=False, index=True)
    question_id = db.Column(db.Integer, nullable=False, index=True)
    mode = db.Column(db.String(10), default="any", index=True)

    asked_at = db.Column(db.DateTime, default=datetime.utcnow, index=True)
    answered_at = db.Column(db.DateTime)
    quality = db.Column(db.Integer)  # 1..5 (z. B. Selbsteinschätzung oder Heuristik)

    # Sicherheit: Ein User bekommt eine Frage nur einmal
    __table_args__ = (
        db.UniqueConstraint("user_id", "question_id", name="uq_user_question_once"),
    )

    def __repr__(self):
        return f"<UserQuestionHistory u={self.user_id} q={self.question_id}>"

# Nützliche Indizes
db.Index("idx_reflection_user_time", Reflection.user_id, Reflection.timestamp)
db.Index("idx_question_cat_diff_mode", Question.category, Question.difficulty, Question.mode)

from datetime import datetime, timedelta
# ... bestehende Imports/DB ...

class PromoCode(db.Model):
    id         = db.Column(db.Integer, primary_key=True)
    code       = db.Column(db.String(64), unique=True, nullable=False)
    uses_left  = db.Column(db.Integer, default=1)
    plan       = db.Column(db.String(20), default="pro")   # z.B. "pro"
    days       = db.Column(db.Integer, default=30)         # Pro-Laufzeit
    token_grant = db.Column(db.Integer, default=0)         # optional Token-Gutschrift
    expires_at = db.Column(db.DateTime, nullable=True)     # optional Ablaufdatum
    pro_until = db.Column(db.DateTime, nullable=True)  # Ende der Pro-Laufzeit
# Optional: beim User eine Pro-Laufzeit
# In ensure_user_columns() (Flask) fügst du pro_until hinzu:
# ALTER TABLE user ADD COLUMN pro_until TEXT