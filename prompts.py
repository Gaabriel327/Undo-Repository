import random

PROMPT_CATEGORIES = {
    "morning": {
        "Selbstwert": {
            "parent": "Selbstbild",
            "questions": [
                "Was macht dich heute wertvoll – unabhängig von Leistung?",
                "Wie würdest du dich heute selbst anerkennen?",
                "Was darfst du dir heute erlauben?"
            ]
        },
        "Grenzen setzen": {
            "parent": "Beziehungen",
            "questions": [
                "Wo darfst du heute klarer Nein sagen?",
                "Wie schützt du heute deine Energie?",
                "Was willst du heute nicht mehr tolerieren?"
            ]
        },
        "Mut zeigen": {
            "parent": "Kreativität & Vision",
            "questions": [
                "Was würdest du tun, wenn du mutiger wärst?",
                "Was wäre heute ein mutiger Schritt für dich?",
                "Wovor hast du Respekt – und willst es trotzdem tun?"
            ]
        },
        "Fokus & Entscheidung": {
            "parent": "Entscheidungsmuster",
            "questions": [
                "Was hat heute Priorität für dich?",
                "Welche Entscheidung wirst du heute bewusst treffen?",
                "Wo darfst du dich heute nicht ablenken lassen?"
            ]
        }
    },
    "evening": {
        "Selbstreflexion": {
            "parent": "Selbstbild",
            "questions": [
                "Was hast du heute über dich selbst erkannt?",
                "Wann warst du heute ehrlich zu dir?",
                "Wo hast du dich heute selbst überrascht?"
            ]
        },
        "Emotionale Achtsamkeit": {
            "parent": "Emotionale Intelligenz",
            "questions": [
                "Wie bist du heute mit deinen Gefühlen umgegangen?",
                "Gab es heute einen Moment echter Verbindung?",
                "Wie gut hast du dich selbst verstanden gefühlt?"
            ]
        },
        "Perspektivwechsel": {
            "parent": "Perspektivwechsel",
            "questions": [
                "Was würdest du aus heutiger Sicht anders machen?",
                "Wie könnte jemand anderes deinen Tag deuten?",
                "Was hat dich heute deinen Blickwinkel verändern lassen?"
            ]
        },
        "Zukunft & Mut": {
            "parent": "Kreativität & Vision",
            "questions": [
                "Was traust du dir morgen zu?",
                "Was hast du heute nicht gesagt, obwohl du es wolltest?",
                "Was wäre ein mutiger Schritt, den du morgen tun könntest?"
            ]
        }
    }
}

# Liefert: Hauptkategorie, Subkategorie, Frage
def get_question(mode="morning"):
    if mode not in PROMPT_CATEGORIES:
        raise ValueError("Mode must be 'morning' or 'evening'")
    subcategory = random.choice(list(PROMPT_CATEGORIES[mode].keys()))
    data = PROMPT_CATEGORIES[mode][subcategory]
    parent_category = data["parent"]
    question = random.choice(data["questions"])
    
    return parent_category, subcategory, question