"""
modules/scorer.py

Responsibilities:
- Load scorer_model.pkl once at import time
- Accept a feature dictionary from features.py
- Return a numeric score 0-100, grade label, and summary

No Flask. No training. No file I/O beyond loading the model.
"""

import os
import pickle
import numpy as np

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "scorer_model.pkl")

# ── Load model once at import time ────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    _PAYLOAD = pickle.load(f)

_MODEL         = _PAYLOAD["model"]
_FEATURE_NAMES = _PAYLOAD["feature_names"]

# ── Grade definitions ─────────────────────────────────────────────────────────
GRADE_LABELS = {
    0: "Needs Work",
    1: "Average",
    2: "Strong",
}

GRADE_SUMMARIES = {
    0: "This resume needs significant improvement. Focus on adding measurable achievements, fixing structure, and removing weak language.",
    1: "This resume is functional but has room to improve. Strengthen your bullet points, add more quantified results, and ensure all key sections are present.",
    2: "This is a strong resume. Minor polish and field-specific tweaks can push it further.",
}


def _features_to_array(features: dict) -> np.ndarray:
    """
    Convert a feature dictionary to a numpy array
    in the exact column order the model was trained on.
    """
    return np.array([[features.get(name, 0) for name in _FEATURE_NAMES]])


def _grade_to_score(grade: int, features: dict) -> float:
    """
    Convert a 3-class grade to a 0-100 score using feature signals
    to place the score within the grade band.

    Grade bands:
        0 (weak)    →  0 – 39
        1 (average) → 40 – 69
        2 (strong)  → 70 – 100
    """
    bands = {
        0: (5,  39),
        1: (40, 69),
        2: (70, 98),
    }
    low, high = bands[grade]

    # Use a few key features to position within the band
    signals = [
        min(features.get("word_count", 0) / 600, 1.0),
        features.get("has_email", 0),
        features.get("has_phone", 0),
        features.get("has_linkedin", 0),
        min(features.get("section_count", 0) / 6, 1.0),
        min(features.get("bullet_count", 0) / 10, 1.0),
        min(features.get("action_verb_count", 0) / 10, 1.0),
        min(features.get("quantification_count", 0) / 5, 1.0),
        max(0, 1 - features.get("filler_count", 0) / 5),
        max(0, 1 - features.get("first_person_count", 0) / 5),
    ]

    # Average signal positions the score within the band
    signal_avg = sum(signals) / len(signals)
    score = low + signal_avg * (high - low)
    return round(float(score), 1)


def score_resume(features: dict) -> dict:
    """
    Takes a feature dictionary from extract_features().
    Returns a dict with score, grade, grade_label, and summary.
    """
    X = _features_to_array(features)
    grade = int(_MODEL.predict(X)[0])
    proba = _MODEL.predict_proba(X)[0]

    score      = _grade_to_score(grade, features)
    grade_label = GRADE_LABELS[grade]
    summary    = GRADE_SUMMARIES[grade]
    confidence = round(float(proba[grade]), 3)

    # Identify strong points
    strong_points = []
    if features.get("has_email") and features.get("has_phone"):
        strong_points.append("Contact information is complete")
    if features.get("has_linkedin"):
        strong_points.append("LinkedIn profile is present")
    if features.get("quantification_count", 0) >= 2:
        strong_points.append("Good use of quantified achievements")
    if features.get("action_verb_count", 0) >= 5:
        strong_points.append("Strong action verbs used throughout")
    if features.get("bullet_count", 0) >= 5:
        strong_points.append("Good use of bullet points")
    if features.get("section_count", 0) >= 4:
        strong_points.append("Resume has a clear section structure")
    if features.get("word_count", 0) >= 300:
        strong_points.append("Resume has good length and detail")

    # Identify quick wins
    quick_wins = []
    if not features.get("has_email"):
        quick_wins.append("Add your email address")
    if not features.get("has_phone"):
        quick_wins.append("Add your phone number")
    if not features.get("has_linkedin"):
        quick_wins.append("Add your LinkedIn URL")
    if features.get("filler_count", 0) > 0:
        quick_wins.append("Remove filler phrases like 'responsible for'")
    if features.get("first_person_count", 0) > 2:
        quick_wins.append("Remove first-person pronouns (I, me, my)")
    if features.get("quantification_count", 0) == 0:
        quick_wins.append("Add at least one quantified achievement (numbers, percentages)")
    if features.get("bullet_count", 0) < 3:
        quick_wins.append("Use bullet points to list your experience")
    if features.get("word_count", 0) < 150:
        quick_wins.append("Expand your resume — it is too short")

    # Identify missing sections
    missing_sections = []
    if not features.get("has_summary"):
        missing_sections.append("Summary or Objective section")
    if not features.get("has_experience"):
        missing_sections.append("Experience section")
    if not features.get("has_education"):
        missing_sections.append("Education section")
    if not features.get("has_skills"):
        missing_sections.append("Skills section")

    return {
        "score":            score,
        "grade":            grade,
        "grade_label":      grade_label,
        "summary":          summary,
        "confidence":       confidence,
        "strong_points":    strong_points,
        "quick_wins":       quick_wins,
        "missing_sections": missing_sections,
    }