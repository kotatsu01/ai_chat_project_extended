from flask import Flask, render_template, request, jsonify, session
import json
import os
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.secret_key = 'your_secret_key'

KNOWLEDGE_FILE = 'knowledge.json'
FEEDBACK_FILE = 'feedback.json'

if not os.path.exists(KNOWLEDGE_FILE):
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump([], f)

if not os.path.exists(FEEDBACK_FILE):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump([], f)

def load_knowledge():
    with open(KNOWLEDGE_FILE, 'r') as f:
        return json.load(f)

def save_knowledge(data):
    with open(KNOWLEDGE_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_feedback():
    with open(FEEDBACK_FILE, 'r') as f:
        return json.load(f)

def save_feedback(data):
    with open(FEEDBACK_FILE, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def find_best_answer(question, knowledge):
    if not knowledge:
        return "情報が見つかりませんでした。"
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([question] + knowledge)
    cosine_sim = cosine_similarity(tfidf[0:1], tfidf[1:])
    best_index = cosine_sim.argmax()
    return knowledge[best_index]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST'])
def submit():
    new_answer = request.form['new_answer']
    knowledge = load_knowledge()
    knowledge.append(new_answer)
    save_knowledge(knowledge)
    return "回答を追加しました！"

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    knowledge = load_knowledge()
    best_answer = find_best_answer(question, knowledge)
    templates = [
        "了解しました。{}。",
        "かしこまりました。{}。",
        "承知しました！{}。",
        "はい、{}。また何かあれば聞いてください。"
    ]
    response = random.choice(templates).format(best_answer)
    session['last_answer'] = best_answer
    return jsonify({'response': response})

@app.route('/feedback', methods=['POST'])
def feedback():
    data = load_feedback()
    user_feedback = {
        'answer': session.get('last_answer', ''),
        'rating': request.form['rating']
    }
    data.append(user_feedback)
    save_feedback(data)
    return "評価を受け付けました！"

if __name__ == '__main__':
    app.run(debug=True)
