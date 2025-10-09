# group_engine.py
import uuid
import sqlite3
from flask import g

DATABASE = "undo.db"

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db

def init_groups_tables():
    db = get_db()
    db.execute('''
        CREATE TABLE IF NOT EXISTS groups (
            id TEXT PRIMARY KEY,
            name TEXT,
            created_by TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS group_members (
            group_id TEXT,
            user_id TEXT,
            PRIMARY KEY (group_id, user_id)
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            group_id TEXT,
            text TEXT
        )
    ''')
    db.execute('''
        CREATE TABLE IF NOT EXISTS answers (
            question_id TEXT,
            user_id TEXT,
            answer TEXT,
            PRIMARY KEY (question_id, user_id)
        )
    ''')
    db.commit()

def create_groups(group_name, user_id):
    db = get_db()
    group_id = str(uuid.uuid4())
    db.execute("INSERT INTO groups (id, name, created_by) VALUES (?, ?, ?)",
               (group_id, group_name, user_id))
    db.execute("INSERT INTO group_members (group_id, user_id) VALUES (?, ?)",
               (group_id, user_id))
    db.commit()
    return group_id

def add_user_to_groups(group_id, user_id):
    db = get_db()
    db.execute("INSERT OR IGNORE INTO group_members (group_id, user_id) VALUES (?, ?)",
               (group_id, user_id))
    db.commit()

def get_groups(group_id):
    db = get_db()
    groups = db.execute("SELECT * FROM groups WHERE id = ?", (group_id,)).fetchone()
    members = db.execute("SELECT user_id FROM group_members WHERE group_id = ?", (group_id,)).fetchall()
    questions = db.execute("SELECT * FROM questions WHERE group_id = ?", (group_id,)).fetchall()
    return {
        "id": groups["id"],
        "name": groups["name"],
        "members": [row["user_id"] for row in members],
        "questions": questions
    }

def add_question(group_id, text):
    db = get_db()
    question_id = str(uuid.uuid4())
    db.execute("INSERT INTO questions (id, group_id, text) VALUES (?, ?, ?)",
               (question_id, group_id, text))
    db.commit()
    return question_id

def submit_answer(question_id, user_id, answer_text):
    db = get_db()
    db.execute("INSERT OR REPLACE INTO answers (question_id, user_id, answer) VALUES (?, ?, ?)",
               (question_id, user_id, answer_text))
    db.commit()

def get_answers(question_id):
    db = get_db()
    rows = db.execute("SELECT user_id, answer FROM answers WHERE question_id = ?", (question_id,)).fetchall()
    return [{"user_id": r["user_id"], "answer": r["answer"]} for r in rows]

def get_groups_for_user(user_id):
    db = get_db()
    rows = db.execute("""
        SELECT g.id, g.name FROM groups g
        JOIN group_members gm ON g.id = gm.group_id
        WHERE gm.user_id = ?
    """, (user_id,)).fetchall()
    return [{"id": r["id"], "name": r["name"]} for r in rows]