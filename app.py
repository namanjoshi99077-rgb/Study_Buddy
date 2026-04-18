import os
import json
import re
import requests
import pdfplumber
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename


app = Flask(__name__, static_folder="static")
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024  

UPLOAD_FOLDER = "C:/tmp/studyai"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

GROQ_API_KEY = "Groq_API"

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
HEADERS = {
    "Authorization": f"Bearer {GROQ_API_KEY}",
    "Content-Type": "application/json",
}

# Free model — very fast (300 tokens/sec), generous free limits
MODEL = "llama-3.1-8b-instant"



@app.route("/")
def index():
    return send_from_directory("static", "index.html")


#  HELPER — extract text from PDF
def extract_pdf_text(file_storage) -> str:
    filename = secure_filename(file_storage.filename)
    path = os.path.join(UPLOAD_FOLDER, filename)
    file_storage.save(path)
    try:
        pages = []
        with pdfplumber.open(path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    pages.append(t)
        return "\n".join(pages).strip()
    finally:
        if os.path.exists(path):
            os.remove(path)


#  HELPER — call Groq API
def ask_groq(system: str, user: str) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": 1024,
    }

    resp = requests.post(GROQ_API_URL, headers=HEADERS, json=payload, timeout=30)

    if resp.status_code == 401:
        raise Exception("Invalid Groq API key. Open app.py and fix GROQ_API_KEY.")

    if resp.status_code == 429:
        raise Exception("Rate limit hit. Wait a few seconds and try again.")

    if resp.status_code != 200:
        raise Exception("Groq API error " + str(resp.status_code) + ": " + resp.text[:300])

    return resp.json()["choices"][0]["message"]["content"].strip()


#  HELPER — safely extract JSON from raw text
def parse_json(raw: str):
    cleaned = re.sub(r'```(?:json)?|```', '', raw).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    m = re.search(r'\[.*\]', cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())
    m = re.search(r'\{.*\}', cleaned, re.DOTALL)
    if m:
        return json.loads(m.group())
    raise json.JSONDecodeError("No JSON found in response", raw, 0)


#  NOTE LEVEL SPECS
LEVEL_PROMPTS = {
    0: (
        "You are an expert tutor. "
        "Write a 25-50 word beginner-friendly overview of the given topic. "
        "Include exactly 1 simple real-world example. "
        "Return ONLY this raw JSON, no extra text, no markdown:\n"
        '{"body":"your overview here","examples":["one real example"],"practiceQs":[]}'
    ),
    1: (
        "You are an expert tutor. "
        "Write a 75-100 word intermediate explanation of the given topic. "
        "Include 2-3 real-world examples. "
        "Return ONLY this raw JSON, no extra text, no markdown:\n"
        '{"body":"your explanation here","examples":["example 1","example 2","example 3"],"practiceQs":[]}'
    ),
    2: (
        "You are an expert tutor. "
        "Write a 100-200 word advanced explanation of the given topic with depth and nuance. "
        "Include 2-3 examples and exactly 3 exam-style practice questions. "
        "Return ONLY this raw JSON, no extra text, no markdown:\n"
        '{"body":"your explanation here","examples":["example 1","example 2"],"practiceQs":["question 1","question 2","question 3"]}'
    ),
}


#  /api/summary  — PDF Summarizer
@app.route("/api/summary", methods=["POST"])
def api_summary():
    text = request.form.get("text", "").strip()

    if not text and "pdf_file" in request.files:
        f = request.files["pdf_file"]
        if f and f.filename:
            try:
                text = extract_pdf_text(f)
            except Exception as e:
                return jsonify({"error": "PDF read failed: " + str(e)}), 400

    if not text:
        return jsonify({"error": "No content provided."}), 400

    try:
        raw = ask_groq(
            "You are an expert academic summarizer. "
            "Return ONLY a raw JSON array of 6-10 bullet point strings. "
            "No markdown, no explanation, just the array.\n"
            'Example: ["Point one.","Point two.","Point three."]',
            "Summarize this into 6-10 bullet points:\n\n" + text[:4000]
        )
        bullets = parse_json(raw)
        if not isinstance(bullets, list):
            raise ValueError("Expected a list")
        return jsonify({"bullets": bullets})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#  /api/flashcards  — Flashcard Creator
@app.route("/api/flashcards", methods=["POST"])
def api_flashcards():
    text = request.form.get("text", "").strip()

    if not text:
        return jsonify({"error": "No content provided."}), 400

    try:
        raw = ask_groq(
            "You are a study flashcard expert. "
            "Return ONLY a raw JSON array of objects with keys 'q' and 'a'. "
            "No markdown, no explanation, just the array.\n"
            'Example: [{"q":"What is X?","a":"X is Y."},{"q":"Define Z.","a":"Z means W."}]',
            "Create 6-8 flashcard Q&A pairs from this content:\n\n" + text[:4000]
        )
        cards = parse_json(raw)
        if not isinstance(cards, list):
            raise ValueError("Expected a list")
        return jsonify({"cards": cards})

    except Exception as e:
        return jsonify({"error": str(e)}), 500



#  /api/notes  — Note Generator

@app.route("/api/notes", methods=["POST"])
def api_notes():
    topic = request.form.get("topic", "").strip()
    try:
        level = int(request.form.get("level", 0))
        if level not in (0, 1, 2):
            level = 0
    except (ValueError, TypeError):
        level = 0

    if not topic:
        return jsonify({"error": "Topic is required."}), 400

    try:
        raw = ask_groq(LEVEL_PROMPTS[level], "Topic: " + topic)
        data = parse_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Expected a dict")
        return jsonify({
            "body":       data.get("body", ""),
            "examples":   data.get("examples", []),
            "practiceQs": data.get("practiceQs", []),
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


#  /api/questions  — Question Generator
@app.route("/api/questions", methods=["POST"])
def api_questions():
    text = request.form.get("text", "").strip()

    if not text:
        return jsonify({"error": "No content provided."}), 400

    try:
        raw = ask_groq(
            "You are an expert exam paper setter. "
            "Return ONLY a raw JSON array of exactly 5 exam-style question strings. "
            "No markdown, no explanation, just the array.\n"
            'Example: ["Question 1?","Question 2?","Question 3?","Question 4?","Question 5?"]',
            "Generate the 5 most important exam questions from this content:\n\n" + text[:4000]
        )
        questions = parse_json(raw)
        if not isinstance(questions, list):
            raise ValueError("Expected a list")
        return jsonify({"questions": questions})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    print("StudyAI (Groq - Free & Fast) running -> http://localhost:5000")
    app.run(debug=True, port=5000)
