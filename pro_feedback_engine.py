# pro_feedback_engine.py
"""
Zentrale Engine für UNDO Pro / Tokens / KI-Feedback.

Enthält:
- Feature-Gating (Pro vs. Free) + Tokenpreise
- Token-Abbuchung pro Feature
- Streak-Belohnungen (3/5/7 → 1/2/3 Tokens; am 7. Tag Reset)
- GPT-Feedback (gpt-4o-mini) + Wochen-/Monats-Report + Antwortvergleich
- Gruppen-Fragegenerator (WeDo) + Solo-Fragegenerator
- Robuste Fallbacks ohne KI

Integration in Flask (Beispiel):
    from pro_feedback_engine import (
        is_pro, FEATURE, require_feature_or_charge,
        update_streak_and_grant_tokens, ai_generate_feedback,
        ai_generate_group_question, ai_generate_question
    )
"""

from __future__ import annotations

import re
import os
import time
import random
from datetime import datetime, timedelta
from typing import Optional, Tuple, List

# OpenAI (neues SDK)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # SDK nicht installiert


# ------------------------------------------------------------
# Export-Liste (für "from pro_feedback_engine import *")
# ------------------------------------------------------------
# --- ganz oben in pro_feedback_engine.py, nach Imports/Utilities ---
__all__ = [
    "is_pro",
    "FEATURE",
    "require_feature_or_charge",
    "update_streak_and_grant_tokens",
    "ai_generate_feedback",
    "ai_weekly_report",
    "ai_monthly_report",
    "ai_answer_compare",
    "ai_generate_question",
    "ai_generate_group_question",  # <-- neu exportiert
]


# ------------------------------------------------------------
# OpenAI Helper
# ------------------------------------------------------------
def _ensure_openai_client() -> "OpenAI":
    """Erzeugt einen OpenAI-Client oder wirft RuntimeError, wenn Key/SDK fehlt."""
    if OpenAI is None:
        raise RuntimeError("OpenAI SDK nicht installiert. `pip install openai>=1.40`")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY fehlt (in .env/Umgebung setzen).")
    return OpenAI(api_key=api_key)


def _call_openai_safe(fn, *, max_retries: int = 2, timeout_s: float = 6.0, fallback_text: Optional[str] = None) -> str:
    """
    Führt eine OpenAI-Operation robust aus:
    - Soft-Timeout je Aufruf
    - wenige Retries bei flüchtigen Fehlern
    - bei Fehlern -> fallback_text (falls gesetzt), sonst Exception
    """
    last_err = None
    for attempt in range(max_retries + 1):
        start = time.time()
        try:
            return fn()
        except Exception as e:
            last_err = e
            time.sleep(0.05 + random.random() * 0.15)  # leichter Backoff
        finally:
            if (time.time() - start) > timeout_s:
                last_err = TimeoutError("OpenAI call exceeded soft timeout")
        # geht in nächsten Versuch
    if fallback_text is not None:
        return fallback_text
    raise last_err if last_err else RuntimeError("OpenAI call failed")


# ------------------------------------------------------------
# Feature-Definition & Preise
# ------------------------------------------------------------
class FEATURE:
    """Enum-ähnliche Sammlung der Feature-Keys."""
    WEDO = "wedo"                          # Nur Pro
    RADAR = "radar"                        # Pro inkl.; Free: 3 Tokens pro Nutzung
    ANSWER_COMPARE = "answer_compare"      # Nach 1 Woche: beide 1 Token
    EXTRA_QUESTION = "extra_question"      # Beide 1 Token
    WEEKLY_REPORT = "weekly_report"        # Pro frei, Free: 2 Tokens
    MONTHLY_REPORT = "monthly_report"      # Pro frei, Free: 4 Tokens
    EXTRA_WEDO = "extra_wedo"

TOKEN_PRICES = {
    FEATURE.RADAR: 3,           # Free
    FEATURE.ANSWER_COMPARE: 1,  # Pro/Free beide 1
    FEATURE.EXTRA_QUESTION: 1,  # Pro/Free beide 1
    FEATURE.WEEKLY_REPORT: 2,   # Free
    FEATURE.MONTHLY_REPORT: 4,  # Free
    FEATURE.EXTRA_WEDO: 2,
}

PRO_FREE = {
    FEATURE.WEDO: "pro_only",               # nur Pro
    FEATURE.RADAR: "included_in_pro",       # Pro 0 Token, Free: 3 Tokens
    FEATURE.ANSWER_COMPARE: "token_for_both",
    FEATURE.EXTRA_QUESTION: "token_for_both",
    FEATURE.WEEKLY_REPORT: "included_in_pro",
    FEATURE.MONTHLY_REPORT: "included_in_pro",
    FEATURE.EXTRA_WEDO: "token_for_both",
}


# ------------------------------------------------------------
# Utility
# ------------------------------------------------------------
def is_pro(user) -> bool:
    """Prüft, ob Pro aktiv ist – via user.subscription == 'pro' ODER Zeitfenster user.pro_until."""
    if (user.subscription or "").lower() == "pro":
        return True
    until = getattr(user, "pro_until", None)
    if until:
        try:
            if isinstance(until, str):
                until = datetime.fromisoformat(until)
        except Exception:
            until = None
        if until and until >= datetime.utcnow():
            return True
    return False


def feature_cost_for_user(user, feature: str) -> Tuple[bool, int, str]:
    """
    Gibt zurück: (allowed, token_cost, reason)
    - allowed = False, wenn z. B. WEDO in Free.
    - token_cost = 0..n
    """
    rule = PRO_FREE.get(feature)

    if rule == "pro_only":
        if is_pro(user):
            return True, 0, "Pro-only Feature freigeschaltet."
        return False, 0, "Dieses Feature ist nur in UNDO Pro verfügbar."

    if rule == "included_in_pro":
        if is_pro(user):
            return True, 0, "In Pro enthalten."
        return True, TOKEN_PRICES.get(feature, 0), "In Free via Tokens."

    if rule == "token_for_both":
        return True, TOKEN_PRICES.get(feature, 0), "Token erforderlich."

    return True, 0, "Kein Preis hinterlegt."


def require_feature_or_charge(db, user, feature: str) -> Tuple[bool, str]:
    """
    Prüft Freischaltung/Kosten. Zieht Tokens ab, wenn nötig und vorhanden.
    Commit’t bei Abbuchung. Gibt (ok, message) zurück.
    """
    allowed, cost, reason = feature_cost_for_user(user, feature)
    if not allowed:
        return False, reason

    if cost <= 0:
        return True, "OK (kostenlos)"

    current = int(user.tokens or 0)
    if current < cost:
        return False, f"Zu wenige Tokens. Benötigt: {cost}."

    user.tokens = current - cost
    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        return False, "Abbuchung fehlgeschlagen."
    return True, f"{cost} Token(s) abgebucht."


# ------------------------------------------------------------
# Streak-Logik (3/5/7 & Reset)
# ------------------------------------------------------------
def update_streak_and_grant_tokens(db, user, now: Optional[datetime] = None) -> None:
    """
    Aktualisiert Streak basierend auf user.last_reflection_date.
    Belohnungen:
      Tag 3 → +1 Token
      Tag 5 → +2 Tokens
      Tag 7 → +3 Tokens & Streak-Reset auf 0
    """
    now = now or datetime.utcnow()
    today = now.date()
    last = user.last_reflection_date.date() if user.last_reflection_date else None

    if last == today:
        pass
    elif last == (today - timedelta(days=1)):
        user.streak = int(user.streak or 0) + 1
    else:
        user.streak = 1

    user.last_reflection_date = now

    earned = 0
    if user.streak == 3:
        earned += 1
    elif user.streak == 5:
        earned += 2
    elif user.streak == 7:
        earned += 3
        user.streak = 0  # Reset

    if earned:
        user.tokens = int(user.tokens or 0) + earned

    try:
        db.session.commit()
    except Exception:
        db.session.rollback()
        raise


# ------------------------------------------------------------
# Fallback-Feedback (regelbasiert, UNDO-Stil)
# ------------------------------------------------------------
def _fallback_feedback(question_text: str, answer_text: str, motive: str, chance: str) -> str:
    """Kurzes Fallback-Feedback im UNDO-Fließtext-Stil (ohne Listen)."""
    ans = (answer_text or "").strip()
    tight = len(ans) < 40
    lacks_time = not any(k in ans.lower() for k in ["heute", "morgen", "uhr"])

    p1 = "Das klingt bedeutsam – und du gehst achtsam damit um. Zwischen den Zeilen zeigt sich, was dir wichtig ist."
    hint_m = " Dein Warum schimmert mit." if (motive or "").strip() else ""
    hint_c = f" {chance} bleibt als Richtung spürbar." if (chance or "").strip() else ""
    p2_parts = []
    if tight:
        p2_parts.append("Vielleicht hilft es, deinen Gedanken noch zwei Sätze Raum zu geben.")
    
    if lacks_time:
        p2_parts.append("Ein kleines Zeitfenster heute kann den Knoten lockern.")
    
    p2 = (" ".join(p2_parts) or "Vielleicht trägt ein leiser Perspektivwechsel weiter.") + hint_m + hint_c
    impulse = "UNDO-Impuls: Einmal kurz anhalten, atmen, neu ausrichten – nur so lange, bis es leise klickt."
    return f"{p1}\n\n{p2}\n\n{impulse}"


# ------------------------------------------------------------
# KI-Funktionen (GPT-4o-mini) – UNDO-Stil
# ------------------------------------------------------------
# def ai_generate_feedback(..., mode: str | None = None, ) -> str:

def ai_generate_feedback(
    question_text: str,
    answer_text: str,
    motive: str,
    chance: str,
    mode: str | None = None,   # "morning" | "evening" | None
    *,
    impulse_label: str = "UNDO-Impuls"   # <— Default: Solo weiterhin UNDO-Impuls
) -> str:
    """
    UNDO-Feedback: 2 kurze Absätze + optionaler „UNDO-Impuls“ (eine Zeile).
    Kein Listenstil, keine Emojis, kein Coaching-Jargon.
    Motive & Chance weich eingebunden. Ton passt sich dem Modus an.
    """
    def _tone_for_mode(m: str | None) -> str:
        if m == "morning":
            return ("Klinge leicht und anhebend; gib Zuversicht für den Start in den Tag. "
                    "Halte den Fokus klein und machbar.")
        if m == "evening":
            return ("Klinge entlastend und freundlich; würdige den Tag und richte sanft den Blick "
                    "auf einen ruhigen Abschluss oder ein leises Weiter.")
        return "Klinge ruhig, klar und zugewandt."

    def _soft_fallback() -> str:
        return _fallback_feedback(question_text, answer_text, motive, chance)
    
    try:
        client = _ensure_openai_client()

        system = (
            "Schreibe wie ein einfühlsamer, klarer Mensch im UNDO-Stil. "
            "Maximal ~120 Wörter. "
            "Keine Bulletpoints/Nummern, keine Emojis, kein Fachjargon. "
            "Kein Dozieren und keine Wiederholung der Eingaben. "
            "Ziel: Verstehen spiegeln + eine sanfte Richtungsänderung anstoßen. "
            "Spreche den User stets im du an, also keine Ich-Perspektive in den Fragen"
            f"{_tone_for_mode(mode)} "
            "Struktur:\n"
            "Absatz 1: Kurzes Spiegeln der Bedeutung & des Moments.\n"
            "Absatz 2: Neue, kleine Perspektive, die aus Gewohnheiten führt (keine Checkliste).\n"
            "Schluss Eine Zeile unter dem Rest: 'UNDO-Impuls: <eine sehr kleine Einladung in einem Satz>'."
        )

        mode_hint = f"Modus: {mode}" if mode else "Modus: unbekannt"
        user_msg = (
            f"{mode_hint}\n"
            f"Frage: {question_text}\n"
            f"Antwort: {answer_text}\n"
            f"Motive (Warum): {motive or '-'}\n"
            f"Chance (Ziel): {chance or '-'}\n\n"
            "Gib genau zwei sehr kurze Absätze. "
            "Hänge, wenn sinnvoll, eine letzte Ein-Satz-Zeile an, die mit 'UNDO-Impuls:' beginnt. "
            "Vermeide Listen, Bindestrichreihen und Floskeln."
        )

        def _do():
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.55,
                max_tokens=260,
            )
            text = (resp.choices[0].message.content or "").strip()
            # Listenoptik entfernen
            if any(b in text for b in ("\n- ", "\n• ", "\n1.", "\n2.", "\n3.")):
                text = text.replace("\n- ", "\n").replace("\n• ", "\n")
                text = text.replace("\n1.", "\n").replace("\n2.", "\n").replace("\n3.", "\n")
            return text

        text = _call_openai_safe(_do, fallback_text=_soft_fallback())
        if len(text.split()) < 8 or "Feedback:" in text:
            return _soft_fallback()
        return text

    except Exception:
        return _soft_fallback()


def ai_weekly_report(snippets: List[str], motive: str, chance: str) -> str:
    """Kompakter Wochenrückblick: 1–2 Absätze + optionaler Impuls (UNDO-Stil)."""
    try:
        client = _ensure_openai_client()
        content = "\n\n".join(f"- {s}" for s in snippets[:12])
        system = (
            "Schreibe wie ein einfühlsamer, klarer Mensch im UNDO-Stil. "
            "1–2 kurze Absätze, maximal ~140 Wörter, keine Listen. "
            "Kurzes Spiegeln der Woche, ein ruhiger Fokus, sanfter Ausblick. "
            "Optional: Eine Schlusszeile 'UNDO-Impuls: ...'."
        )
        user = f"Beweggrund: {motive or '-'} | Aussicht: {chance or '-'}\nBeispiele der Woche:\n{content}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=260,
        )
        text = (resp.choices[0].message.content or "").strip()
        if any(b in text for b in ("\n- ", "\n• ", "\n1.", "\n2.", "\n3.")):
            text = text.replace("\n- ", "\n").replace("\n• ", "\n")
            text = text.replace("\n1.", "\n").replace("\n2.", "\n").replace("\n3.", "\n")
        return text if len(text.split()) >= 8 else "Ein ruhiger Wochenblick: Was trug, darf leiser wachsen. UNDO-Impuls: Am Sonntag kurz ordnen, dann leicht starten."
    except Exception:
        return "Ein ruhiger Wochenblick: Was trug, darf leiser wachsen. UNDO-Impuls: Am Sonntag kurz ordnen, dann leicht starten."


def ai_monthly_report(snippets: List[str], motive: str, chance: str) -> str:
    """Kompakter Monatsrückblick: 2 Absätze + optionaler Impuls (UNDO-Stil)."""
    try:
        client = _ensure_openai_client()
        content = "\n\n".join(f"- {s}" for s in snippets[:20])
        system = (
            "Schreibe wie ein einfühlsamer, klarer Mensch im UNDO-Stil. "
            "2 Absätze, maximal ~180 Wörter, keine Listen. "
            "Würdige die Entwicklung, mache zwei stille Stärken sichtbar und zeige behutsam eine Richtung. "
            "Optional: Eine Schlusszeile 'UNDO-Impuls: ...'."
        )
        user = f"Beweggrund: {motive or '-'} | Aussicht: {chance or '-'}\nMonatsbeispiele:\n{content}"

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.5,
            max_tokens=320,
        )
        text = (resp.choices[0].message.content or "").strip()
        if any(b in text for b in ("\n- ", "\n• ", "\n1.", "\n2.", "\n3.")):
            text = text.replace("\n- ", "\n").replace("\n• ", "\n")
            text = text.replace("\n1.", "\n").replace("\n2.", "\n").replace("\n3.", "\n")
        return text if len(text.split()) >= 8 else "Ein stiller Monatsblick: Deine Linie wird klarer. UNDO-Impuls: Nimm dir eine Sache, die leicht bleibt – und zieh sie leise durch."
    except Exception:
        return "Ein stiller Monatsblick: Deine Linie wird klarer. UNDO-Impuls: Nimm dir eine Sache, die leicht bleibt – und zieh sie leise durch."


def ai_answer_compare(question_text: str, previous_answer: str, current_answer: str) -> str:
    """Vergleich zweier Antworten – 2 Sätze + optionaler Impuls (UNDO-Stil)."""
    try:
        client = _ensure_openai_client()
        system = (
            "Schreibe wie ein einfühlsamer, klarer Mensch im UNDO-Stil. "
            "Zwei Sätze, keine Liste. "
            "Erstes: kurz spiegeln, was neu/gewachsen ist. "
            "Zweites: sanft die Richtung halten. "
            "Optional: Schlusszeile 'UNDO-Impuls: ...' (eine Zeile)."
        )
        user = (
            f"Frage: {question_text}\n"
            f"Vorherige Antwort: {previous_answer}\n"
            f"Aktuelle Antwort: {current_answer}\n"
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": user}],
            temperature=0.55,
            max_tokens=180,
        )
        text = (resp.choices[0].message.content or "").strip()
        if any(b in text for b in ("\n- ", "\n• ", "\n1.", "\n2.", "\n3.")):
            text = text.replace("\n- ", "\n").replace("\n• ", "\n")
            text = text.replace("\n1.", "\n").replace("\n2.", "\n").replace("\n3.", "\n")
        return text if len(text.split()) >= 6 else "Du bist klarer geworden – und das trägt. UNDO-Impuls: Bleib klein, aber täglich sichtbar."
    except Exception:
        return "Du bist klarer geworden – und das trägt. UNDO-Impuls: Bleib klein, aber täglich sichtbar."


# ------------------------------------------------------------
# Fragegeneratoren
# ------------------------------------------------------------
def ai_generate_group_question(
    motive: str | None,
    chance: str | None,
    mode: str | None = None,          # "morning" | "evening" optional
    seed_texts: List[str] | None = None
) -> str:
    """
    Liefert GENAU EINE Gruppenfrage im UNDO/WeDo-Stil.
    - 1 Satz (8–18 Wörter), „wir“-Perspektive, warm und klar.
    - Motive/Chance nur subtil als Hintergrund.
    - Modus kann den Ton (Start/Abschluss) andeuten.
    - Fallback nutzt Seeds.
    """
    default_seeds_morning = [
         "Womit beginnt ihr heute, damit es sich leicht und stimmig anfühlt?",
         "Welche kleine Entscheidung gibt eurem Tag heute einen guten Ton?",
         "Worauf wollt ihr euch gemeinsam einlassen, damit Bewegung entsteht?",
         "Was braucht euer Miteinander heute, um ruhig in Gang zu kommen?"
    ]
    default_seeds_evening = [
         "Welcher Moment hat euch heute ein Stück näher zusammengebracht?",
         "Wofür wollt ihr euch heute gemeinsam Anerkennung geben?",
         "Was dürft ihr heute abschließen, damit es ruhiger in euch wird?",
         "Welche kleine Sache hat heute einen echten Unterschied gemacht?"
    ]

    seeds = (seed_texts or [])[:]
    if not seeds:
        seeds = default_seeds_evening if (mode == "evening") else default_seeds_morning

    motive_s = (motive or "").strip()
    chance_s = (chance or "").strip()

    # Versuche KI
    try:
        client = _ensure_openai_client()
        system = (
            "Du bist ein empathischer Coach im UNDO/WeDo-Stil. "
            "Gib genau EINE Frage, ein einzelner Satz (8–18 Wörter), "
            "in ihr-Perspektive, warm und klar. "
            "Kein Listenstil, keine Befehle, kein Fachjargon, keine Wiederholung der Nutzerdaten."
        )
        user_msg = (
            f"Modus: {mode or 'unbekannt'}\n"
            f"Gruppen-Motive: {motive_s or '-'}\n"
            f"Gruppen-Chance: {chance_s or '-'}\n"
            "Formatiere GENAU: nur 1 Satz, sonst nichts."
        )

        def _do():
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.5,
                max_tokens=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = text.split("\n")[0].strip()
            # minimale Normalisierung
            if not text.endswith("?"):
                text = text.rstrip(". ") + "?"
            return text

        return _call_openai_safe(_do, fallback_text=None)  # wenn KI patzt -> unten Seeds

    except Exception:
        pass

    # Fallback Seeds, minimal personalisiert
    base = random.choice(seeds)
    if motive_s or chance_s:
        # nur eine leichte Andeutung
        if "heute" in base.lower():
            base = base.replace("heute", "heute – passend zu unserem Warum")
    if not base.endswith("?"):
        base += "?"
    return base


def ai_generate_question(motive: str, chance: str, mode: str, seed_texts: List[str] | None = None) -> str:
    """"
   " Du formulierst EINE kurze Gruppenfrage im UNDO-Stil – warm, klar, ohne Listen, keine Emojis. "
   " Sprich die Gruppe als 'ihr' an (keine Ich- oder Wir-Perspektive).  max. ~22 Wörter. "
   " Subtiler Bezug auf Motiv/Chance. Fallback nutzt seed_texts.
    """
    motive_s = (motive or "").strip()
    chance_s = (chance or "").strip()
    fallback_seeds = seed_texts or [
        "Worauf richtest du heute deinen Blick – ganz bewusst?",
        "Welche kleine Entscheidung macht deinen Tag heute leichter?",
        "Was braucht dich heute für 10 ruhige Minuten wirklich?",
        "Womit beginnst du, damit es sich stimmig anfühlt?"
    ]
    fallback = random.choice(fallback_seeds)
    if motive_s or chance_s:
        fallback = f"{fallback} (mit Blick auf: {motive_s or 'dein Warum'} / {chance_s or 'dein Ziel'})"

    try:
        client = _ensure_openai_client()
        system = (
            "Formuliere genau EINE Frage im UNDO-Stil. Warm, konkret, natürlich. "
            "Max. 22 Wörter. Kein Listenstil, kein Jargon, keine Emojis. "
            "Gib NUR die Frage zurück."
            "Spreche den User im Du an"
        )
        user_msg = (
            f"Modus: {mode or 'unbekannt'}\n"
            f"Motiv: {motive_s or '-'}\n"
            f"Chance: {chance_s or '-'}\n"
            "Kontext: Tägliche Selbstreflexion, die zu kleinen bewussten Veränderungen einlädt."
        )

        def _do():
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_msg},
                ],
                temperature=0.4,
                max_tokens=50,
            )
            text = (resp.choices[0].message.content or "").strip()
            text = text.split("\n")[0].strip()
            if not text.endswith("?"):
                text += "?"
            if len(text) > 180:
                text = text[:180].rstrip() + "?"
            return text

        return _call_openai_safe(_do, fallback_text=fallback)

    except Exception:
        return fallback