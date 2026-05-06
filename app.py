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
import sys
import subprocess
import tempfile
import importlib.metadata
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv

APP_VERSION = "1.0.0"

FEATURE_LIST = {
    "career_classifier": {
        "classes":  ["business", "creative", "marketing", "other", "tech"],
        "pipeline": "TF-IDF + Logistic Regression",
        "meta":     "models/Career model.meta.json",
    },
    "resume_scorer": {
        "feature_count":  27,
        "feature_groups": ["text_volume", "contact_info", "structure",
                           "writing_quality", "field_keywords", "impact"],
        "grades":         ["weak", "average", "strong"],
        "meta":           "models/Scorer model.meta.json",
    },
}


def _get_commit_hash() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            timeout=3,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


_COMMIT_HASH = _get_commit_hash()

load_dotenv()

_MODELS_OK = True
_MODELS_ERROR: str | None = None

try:
    from modules.extractor import extract_text
    from modules.results import build_resume_result
except Exception as _boot_exc:
    _MODELS_OK = False
    _MODELS_ERROR = str(_boot_exc)

    def extract_text(path):  # type: ignore[misc]
        raise RuntimeError("Models not loaded at startup")

    def build_resume_result(**kwargs):  # type: ignore[misc]
        raise RuntimeError("Models not loaded at startup")

app = Flask(__name__)

app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024# hot try exception

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
    try:
        pkg_versions = {
            "python":       sys.version.split()[0],
            "flask":        importlib.metadata.version("flask"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost":      importlib.metadata.version("xgboost"),
        }
    except Exception:
        pkg_versions = {"python": sys.version.split()[0]}

    payload = {
        "status":           "ok" if _MODELS_OK else "degraded",
        "version":          APP_VERSION,
        "commit":           _COMMIT_HASH,
        "timestamp":        datetime.now(timezone.utc).isoformat(),
        "models_loaded":    _MODELS_OK,
        "package_versions": pkg_versions,
        "features":         FEATURE_LIST,
    }
    if not _MODELS_OK:
        payload["model_error"] = _MODELS_ERROR

    return jsonify(payload), 200 if _MODELS_OK else 503


@app.route("/analyze", methods=["POST"])
def analyze():
    if not _MODELS_OK:
        return jsonify({"error": "Service unavailable — models failed to load", "detail": _MODELS_ERROR}), 503

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

        try:
            result = build_resume_result(
                text=text,
                user_field=user_field,
                jd_text=jd_text,
            )
            return jsonify(result), 200
        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({
                "error": "Analysis logic failed",
                "details": str(e),
                "trace": traceback.format_exc()
            }), 500

    finally:
        try:
            os.unlink(tmp.name)
        except Exception:
            pass


if __name__ == "__main__":
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    port = int(os.getenv("PORT", 5001))
    app.run(debug=debug, port=5001, use_reloader=False)# hot try exception