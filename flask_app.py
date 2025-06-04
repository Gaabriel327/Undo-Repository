from flask import Flask, render_template
from prompts import get_question
from datetime import datetime

app = Flask(__name__)

@app.route("/")
def index():
    now = datetime.now()
    time_of_day = "morning" if now.hour < 18 else "evening"
    question = get_question(time_of_day)
    return render_template("index.html", question=question)

if __name__ == "__main__":
    app.run(debug=True)
    