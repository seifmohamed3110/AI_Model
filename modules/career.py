"""
modules/career.py
"""

import os
import re
import pickle
import sys
import sklearn

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(BASE_DIR, "models", "career_model.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "models", "career_classes.pkl")

# ── Fix broken pickle import (VERY IMPORTANT) ────────────────────────────────
sys.modules["sklear"] = sklearn

# ── Load model ───────────────────────────────────────────────────────────────
with open(MODEL_PATH, "rb") as f:
    _MODEL = pickle.load(f)

with open(CLASSES_PATH, "rb") as f:
    _CLASSES = pickle.load(f)

# ── Keyword lists ────────────────────────────────────────────────────────────
TECH_KEYWORDS = [
    "python", "java", "javascript", "typescript", "c++", "c#",
    "react", "angular", "vue", "node", "django", "flask", "fastapi",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux",
    "machine learning", "deep learning", "tensorflow", "pytorch",
    "scikit-learn", "data science", "sql", "nosql", "mongodb",
    "postgresql", "mysql", "rest api", "graphql", "devops",
    "ci/cd", "terraform", "ansible", "spark", "hadoop",
    "software engineer", "backend", "frontend", "full stack",
    "software developer", "data engineer", "ml engineer",
]

MARKETING_KEYWORDS = [
    "seo", "sem", "google analytics", "google ads", "facebook ads",
    "content marketing", "email marketing", "social media marketing",
    "hubspot", "mailchimp", "hootsuite", "campaign", "conversion rate",
    "a/b testing", "marketing automation", "brand strategy",
    "market research", "lead generation", "ppc", "digital marketing",
    "marketing manager", "marketing specialist", "marketing director",
]

CREATIVE_KEYWORDS = [
    "photoshop", "illustrator", "indesign", "figma", "sketch",
    "adobe xd", "after effects", "premiere pro", "lightroom",
    "ux design", "ui design", "user experience", "wireframe",
    "prototype", "branding", "typography", "graphic design",
    "art director", "creative director", "motion graphics",
    "video editing", "illustration", "photography",
]

BUSINESS_KEYWORDS = [
    "project management", "stakeholder", "business development",
    "financial analysis", "forecasting", "supply chain", "procurement",
    "risk management", "compliance", "p&l", "revenue growth",
    "team leadership", "cross-functional", "change management",
    "six sigma", "process improvement", "okr", "kpi",
    "business analyst", "operations manager", "product manager",
]

VALID_FIELDS = {"tech", "marketing", "creative", "business", "other"}


def _keyword_score(text_lower: str, keywords: list) -> int:
    return sum(
        1 for kw in keywords
        if re.search(r"\b" + re.escape(kw) + r"\b", text_lower)
    )


def detect_field(text: str, user_override: str = None) -> str:
    if user_override:
        override = user_override.strip().lower()
        if override in VALID_FIELDS:
            return override

    text_lower = text.lower()

    scores = {
        "tech":      _keyword_score(text_lower, TECH_KEYWORDS),
        "marketing": _keyword_score(text_lower, MARKETING_KEYWORDS),
        "creative":  _keyword_score(text_lower, CREATIVE_KEYWORDS),
        "business":  _keyword_score(text_lower, BUSINESS_KEYWORDS),
    }

    best_field = max(scores, key=scores.get)
    best_score = scores[best_field]

    if best_score >= 4:
        return best_field

    text_truncated = text[:1000]
    prediction = _MODEL.predict([text_truncated])[0]
    return prediction


def get_confidence(text: str) -> dict:
    text_lower = text.lower()
    text_truncated = text[:1000]

    proba = _MODEL.predict_proba([text_truncated])[0]
    ml_scores = dict(zip(_CLASSES, proba))

    keyword_scores = {
        "tech":      _keyword_score(text_lower, TECH_KEYWORDS),
        "marketing": _keyword_score(text_lower, MARKETING_KEYWORDS),
        "creative":  _keyword_score(text_lower, CREATIVE_KEYWORDS),
        "business":  _keyword_score(text_lower, BUSINESS_KEYWORDS),
    }

    result = {}
    for field in VALID_FIELDS:
        ml_prob  = float(ml_scores.get(field, 0.0))
        kw_score = min(keyword_scores.get(field, 0) / 10.0, 1.0)
        result[field] = round((ml_prob + kw_score) / 2, 3)

    return result