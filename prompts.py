import random

morning_questions = [
    "Was ist dein Ziel für heute?",
    "Worauf willst du dich heute konzentrieren?"
]

evening_questions = [
    "Was hast du heute gelernt?",
    "Worauf bist du heute stolz?"
]

def get_question(time_of_day):
    if time_of_day == "morning":
        return random.choice(morning_questions)
    elif time_of_day == "evening":
        return random.choice(evening_questions)
    return "Wie geht es dir gerade?"
