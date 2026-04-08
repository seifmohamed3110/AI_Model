"""
app.py

Flask API — routing only.
All logic lives in modules/.
No ML, no text processing, no business rules here.

Endpoints:
    GET  /               — serve the HTML frontend
    POST /analyze        — analyze a resume file
    GET  /health         — health check

Run:
    flask --app app run --port 5000 --no-reload
"""

import os
import tempfile
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from modules.extractor   import extract_text
from modules.features    import extract_features
from modules.career      import detect_field
from modules.scorer      import score_resume
from modules.ats         import check_ats
from modules.writing     import check_writing
from modules.suggestions import get_suggestions, get_jd_match

load_dotenv()

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf", "docx"}


def allowed_file(filename: str) -> bool:
    return (
        "." in filename and
        filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only PDF and DOCX files are supported"}), 400

    user_field = request.form.get("field", None)
    jd_text    = request.form.get("job_description", None)

    suffix = "." + file.filename.rsplit(".", 1)[1].lower()
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    try:
        file.save(tmp.name)
        tmp.close()

        try:
            text = extract_text(tmp.name)
        except Exception as e:
            return jsonify({"error": f"Could not read file: {str(e)}"}), 422

        if len(text.strip()) < 50:
            return jsonify({"error": "Could not extract enough text from file"}), 422

        features       = extract_features(text)
        detected_field = detect_field(text, user_override=user_field)
        score_result   = score_resume(features)
        ats_issues     = check_ats(text)
        writing_issues = check_writing(text)
        improvements   = get_suggestions(features, detected_field)

        jd_result = None
        if jd_text and jd_text.strip():
            jd_result = get_jd_match(text, jd_text)

        response = {
            "score":            score_result["score"],
            "grade":            score_result["grade_label"],
            "word_count":       features["word_count"],
            "detected_field":   detected_field,
            "summary":          score_result["summary"],
            "strong_points":    score_result["strong_points"],
            "quick_wins":       score_result["quick_wins"],
            "missing_sections": score_result["missing_sections"],
            "ats_issues":       ats_issues,
            "writing_issues":   writing_issues,
            "improvements":     improvements,
        }

        if jd_result:
            response["jd_match_score"]   = jd_result["match_score"]
            response["missing_keywords"] = jd_result["missing_keywords"]

        return jsonify(response), 200

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port  = int(os.getenv("PORT", 5000))
    app.run(debug=debug, port=port, use_reloader=False)