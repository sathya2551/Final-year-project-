from flask import Flask, render_template, request, jsonify
import pickle
import json
import random
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import datetime

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

lem = WordNetLemmatizer()

def normalize(text):
    tokens = [t for t in word_tokenize(text.lower()) if t.isalnum()]
    return " ".join(lem.lemmatize(t) for t in tokens)

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

with open("intents.json", "r", encoding="utf-8-sig") as f:
    intents = json.load(f)

CONFIDENCE_THRESHOLD = 0.50
LOGFILE = "chat_logs.txt"

def log_interaction(user_msg, pred_tag, confidence, reply):
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.datetime.now()}\t{user_msg}\t{pred_tag}\t{confidence:.3f}\t{reply}\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def chatbot():
    data = request.get_json(force=True)
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"reply": "Please type a message."})

    vec = vectorizer.transform([normalize(msg)])
    try:
        probs = model.predict_proba(vec)[0]
        best_idx = probs.argmax()
        best_prob = probs[best_idx]
        tag = model.classes_[best_idx]
    except Exception:
        reply = "Sorry, I couldn't process that."
        log_interaction(msg, "error", 0.0, reply)
        return jsonify({"reply": reply})

    if best_prob < CONFIDENCE_THRESHOLD:
        reply = "I didn't understand that. Please rephrase or ask about courses, admissions, fees, contact, etc."
        log_interaction(msg, tag, best_prob, reply)
        return jsonify({"reply": reply})

    for intent in intents.get("intents", []):
        if intent.get("tag") == tag:
            reply = random.choice(intent.get("responses", ["Sorry."]))
            log_interaction(msg, tag, best_prob, reply)
            return jsonify({"reply": reply})

    reply = "I don't know the answer to that."
    log_interaction(msg, "unknown", best_prob, reply)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    app.run(debug=True)
