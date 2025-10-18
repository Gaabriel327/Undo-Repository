# models.py
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin
import uuid 

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
    birth_date = db.Column(db.Date)
    motive = db.Column(db.Text)   # Beweggrund
    chance = db.Column(db.Text)   # Aussicht/Ziel
    profile_completed = db.Column(db.Boolean, default=False)

    #Promo Code
    pro_until = db.Column(db.DateTime, nullable=True)
    promo_locked = db.Column(db.Boolean, default=False)      # verhindert Downgrade
    promo_code_id = db.Column(db.Integer, db.ForeignKey("promo_codes.id"), nullable=True)
    promo_code = db.relationship("PromoCode", back_populates="users")

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

    # String-UUID als Primary Key, Default wird clientseitig von SQLAlchemy vergeben:
    id            = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))

    name          = db.Column(db.String(120), nullable=False)
    created_by    = db.Column(db.String(64), nullable=False)
    group_members = db.Column(db.Text, default="")

    # falls du sie nutzt – harmlose optionale Felder:
    created_at    = db.Column(db.DateTime, default=datetime.utcnow)
    motive        = db.Column(db.Text)
    chance        = db.Column(db.Text)
    last_q_text   = db.Column(db.Text)
    last_q_day    = db.Column(db.String(16))
    last_q_mode   = db.Column(db.String(16))
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
    __tablename__ = "promo_codes"
    id = db.Column(db.Integer, primary_key=True)
    code = db.Column(db.String(64), unique=True, nullable=False, index=True)
    duration_days = db.Column(db.Integer, nullable=False, default=30)
    active = db.Column(db.Boolean, default=True, nullable=False)
    expires_at = db.Column(db.DateTime, nullable=True)
    max_uses = db.Column(db.Integer, nullable=True)    # None = unbegrenzt
    used_count = db.Column(db.Integer, default=0, nullable=False)
    note = db.Column(db.String(200))
    users = db.relationship("User", back_populates="promo_code")