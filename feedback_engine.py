# feedback_engine.py
import random

# Bewertung der Antwortqualität
def analyze_answer(answer):
    tokens = answer.lower().split()
    length = len(tokens)

    if length < 10:
        return "kurz"
    elif any(phrase in answer.lower() for phrase in ["weiß nicht", "keine ahnung", "bin mir nicht sicher"]):
        return "unsicher"
    elif any(phrase in answer.lower() for phrase in ["ich denke", "mir ist aufgefallen", "ich habe erkannt"]):
        return "reflektiert"
    else:
        return "mittel"

# Hauptfunktion zur Feedback-Generierung
def generate_feedback(answer):
    tone = analyze_answer(answer)

    if tone == "kurz":
        return (
            "Du hast dich auf den ersten Schritt eingelassen, und das zählt. "
            "Auch wenn die Antwort noch knapp ausfällt, zeigt sie Bereitschaft zur Auseinandersetzung. "
            "Vielleicht ist es hilfreich, dir noch einen Moment Zeit zu nehmen und tiefer in dich hineinzuhören. "
            "Was genau beschäftigt dich, wenn du die Frage erneut liest? "
            "Gerade unter der Oberfläche liegt oft der spannendste Gedanke verborgen. "
            "Vertrau dir – du darfst mutiger werden."
        )

    elif tone == "unsicher":
        return (
            "Unsicherheit gehört zum Prozess der Selbstreflexion dazu – und sie ist ein Zeichen von Mut. "
            "Deine Antwort wirkt vorsichtig tastend, als würdest du dich langsam herantasten. "
            "Vielleicht spürst du innerlich, dass da mehr ist, aber du bist noch nicht bereit, es ganz auszusprechen. "
            "Diese Spannung ist wertvoll: Sie zeigt, dass du dich ernsthaft mit dir selbst beschäftigst. "
            "Nimm diese Unsicherheit als Einladung – nicht als Schwäche. "
            "Versuch es morgen nochmal, mit einem kleinen Schritt mehr Offenheit."
        )

    elif tone == "reflektiert":
        return (
            "Deine Antwort wirkt durchdacht und zeigt, dass du bereits aktiv mit dir arbeitest. "
            "Du formulierst ehrlich, differenziert und mutig – das ist eine echte Qualität. "
            "Besonders wertvoll ist, dass du dir selbst Fragen stellst, anstatt dich mit einfachen Antworten zufriedenzugeben. "
            "Vielleicht gibt es dennoch einen Bereich, dem du bisher ausweichst? "
            "Wirkliche Tiefe beginnt dort, wo man sich selbst überrascht. "
            "Bleib dran – deine Gedanken haben Substanz."
        )

    else:  # mittel
        return (
            "Deine Antwort enthält Ansätze von Tiefe – ein guter Startpunkt. "
            "An manchen Stellen wirkt sie noch zurückhaltend oder etwas allgemein. "
            "Das kann ein Zeichen dafür sein, dass du dir Zeit lässt, dich zu öffnen – was völlig in Ordnung ist. "
            "Stell dir vor, du würdest diesen Gedanken einem guten Freund erzählen: Was würdest du noch ergänzen? "
            "Manchmal helfen kleine Details, um die eigene Haltung besser zu verstehen. "
            "Bleib neugierig auf dich selbst – genau dort beginnt Veränderung."
        )
    
    # 🔢 Bewertung der Antwortqualität für Tokens
def evaluate_tokens(answer):
    words = len(answer.split())
    score = 0

    if words > 20:
        score += 1  # Länge
    if words > 50:
        score += 1  # Ausführlichkeit
    if any(p in answer.lower() for p in ["ich glaube", "was ich spüre", "wenn ich ehrlich bin", "tief in mir"]):
        score += 2  # Tiefe (einfache Trigger-Phrasen)

    return min(score, 5)