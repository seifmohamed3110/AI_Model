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
    flask --app app run --port 5001 --no-reload
"""

import os
import tempfile
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

from modules.extractor import extract_text
from modules.results import build_resume_result

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
    return jsonify({"status": "ok", "models_loaded": True}), 200


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
    jd_text = request.form.get("job_description", None)

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

        result = build_resume_result(
            text=text,
            user_field=user_field,
            jd_text=jd_text,
        )

        return jsonify(result), 200

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5001))
    app.run(debug=debug, port=port, use_reloader=False)