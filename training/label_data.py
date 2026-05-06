"""
training/label_data.py

Responsibilities:
- Read data/raw_resumes.csv (Kaggle snehaanbhawal/resume-dataset)
- Score each resume using a rule-based function that mirrors
  the manual gold dataset's 4-dimension rubric:
      ats_score          (0-2)
      writing_score      (0-2)
      achievement_score  (0-2)
      relevance_score    (0-2)
  and derives a final_score (0-100) matching the gold-label scale.
- Assign quality_label via the same grade thresholds used by reviewers:
      0 = weak    (final_score < 40)
      1 = average (final_score 40-69)
      2 = strong  (final_score >= 70)
- Align column names to the gold dataset schema:
      cv_text        — resume plain text
      career_label   — field bucket (tech / business / creative / other)
      quality_label  — 0 / 1 / 2

Output: data/labeled_resumes.csv

Run once from the project root:
    python training/label_data.py
"""

import os
import re
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT    = os.path.join(DATA_DIR, "raw_resumes.csv")
OUTPUT   = os.path.join(DATA_DIR, "labeled_resumes.csv")


# ── Category → career_label mapping ─────────────────────────────────────────
# Matches the 4 labels present in the manual gold dataset:
#   tech | business | creative | other
CATEGORY_MAP = {
    "Information-Technology": "tech",
    "Engineering":            "tech",

    "Designer":        "creative",
    "Arts":            "creative",
    "Digital-Media":   "creative",
    "Apparel":         "creative",

    "Business-Development": "business",
    "Sales":                "business",
    "Consultant":           "business",
    "BPO":                  "business",
    "Bpo":                  "business",
    "Finance":              "business",
    "Accountant":           "business",
    "Banking":              "business",
    "HR":                   "business",
    "Hr":                   "business",
    "Public-Relations":     "business",

    "Healthcare":    "other",
    "Fitness":       "other",
    "Teacher":       "other",
    "Agriculture":   "other",
    "Automobile":    "other",
    "Chef":          "other",
    "Aviation":      "other",
    "Construction":  "other",
    "Advocate":      "other",
}


# ── Keyword lists (mirrors gold-label rubric dimensions) ─────────────────────
SECTION_HEADINGS = [
    "experience", "work experience", "employment", "education", "skills",
    "summary", "objective", "profile", "certifications", "projects",
    "achievements", "awards", "publications", "languages", "interests",
    "references", "volunteer", "internship",
]

ACTION_VERBS = [
    "achieved", "built", "created", "delivered", "designed", "developed",
    "drove", "established", "executed", "generated", "implemented",
    "improved", "increased", "launched", "led", "managed", "optimized",
    "reduced", "scaled", "spearheaded", "streamlined", "transformed",
    "architected", "automated", "deployed", "engineered",
]

FILLER_PHRASES = [
    "responsible for", "duties included", "worked on",
    "helped with", "assisted in", "involved in",
]

TECH_HINTS = [
    "python", "java", "javascript", "typescript", "react", "node", "django",
    "flask", "fastapi", "aws", "azure", "gcp", "docker", "kubernetes",
    "sql", "nosql", "tensorflow", "pytorch", "scikit-learn", "api", "graphql",
]

MARKETING_HINTS = [
    "seo", "sem", "google analytics", "google ads", "facebook ads", "hubspot",
    "salesforce", "crm", "campaign", "copywriting", "ppc", "roi", "kpi",
]

CREATIVE_HINTS = [
    "figma", "photoshop", "illustrator", "indesign", "after effects",
    "premiere pro", "canva", "ux", "ui", "wireframe", "prototype", "portfolio",
]

BUSINESS_HINTS = [
    "project management", "stakeholder", "budget", "forecasting", "strategy",
    "operations", "procurement", "compliance", "risk management",
    "financial analysis", "revenue", "cost reduction", "team leadership",
]


# ── Helpers ──────────────────────────────────────────────────────────────────
def _keyword_present(text: str, keyword: str) -> bool:
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def _count_keywords(text: str, keywords: list) -> int:
    return sum(1 for kw in keywords if _keyword_present(text, kw))


def _detect_sections(lines: list) -> list:
    found = []
    for line in lines:
        clean = line.strip().lower().rstrip(":")
        if clean in SECTION_HEADINGS and len(clean) < 40:
            found.append(clean)
    return found


# ── Dimension scorers (each returns 0 | 1 | 2) ───────────────────────────────

def _score_ats(text: str, text_lower: str, lines: list) -> int:
    """
    ats_score: measures machine-readability and structural completeness.
      0 – missing core contact/sections, very short or cluttered
      1 – has basic contact + most core sections
      2 – full contact (email + phone + LinkedIn/URL) AND all 3 core sections
    """
    has_email   = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text))
    has_phone   = bool(re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text))
    has_link    = bool(re.search(r"(linkedin\.com|github\.com|behance\.net|dribbble\.com)", text_lower))

    found_sections = _detect_sections(lines)
    has_experience = any(s in found_sections for s in ["experience", "work experience", "employment", "internship"])
    has_education  = "education" in found_sections
    has_skills     = "skills" in found_sections

    contact_score  = sum([has_email, has_phone, has_link])   # 0-3
    core_sections  = sum([has_experience, has_education, has_skills])  # 0-3

    if contact_score >= 3 and core_sections == 3:
        return 2
    elif contact_score >= 1 and core_sections >= 2:
        return 1
    else:
        return 0


def _score_writing(text: str, text_lower: str, lines: list, words: list) -> int:
    """
    writing_score: measures language quality and professional tone.
      0 – heavy filler language, lots of first-person, very short
      1 – moderate quality; some action verbs but also filler
      2 – strong verb-driven, minimal filler, appropriate length
    """
    wc = len(words)
    verb_count   = _count_keywords(text_lower, ACTION_VERBS)
    filler_count = sum(
        len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        for p in FILLER_PHRASES
    )
    first_person = len(re.findall(r"\b(i|me|my|myself)\b", text_lower))

    # Length penalty / bonus
    length_ok = 200 <= wc <= 1000

    if verb_count >= 6 and filler_count <= 1 and first_person <= 2 and length_ok:
        return 2
    elif verb_count >= 2 and filler_count <= 4 and wc >= 120:
        return 1
    else:
        return 0


def _score_achievement(text: str, text_lower: str, lines: list) -> int:
    """
    achievement_score: measures quantification and impact evidence.
      0 – no numbers or measurable results
      1 – 1-2 quantified results
      2 – 3+ quantified results with metrics
    """
    quant = len(re.findall(
        r"\b\d+\s*(%|percent|x|times|million|billion|k\b|users|customers|"
        r"revenue|hours|days|months|years|projects|clients|team members|branches|"
        r"egp|usd|sar|aed)\b",
        text_lower
    ))
    bullets = [l for l in lines if re.match(r"^[\•\-\*\–\—\►\▪\○\●]", l)]
    bullet_count = len(bullets)

    if quant >= 3 and bullet_count >= 4:
        return 2
    elif quant >= 1 or bullet_count >= 2:
        return 1
    else:
        return 0


def _score_relevance(text_lower: str, career_label: str) -> int:
    """
    relevance_score: measures field-specific keyword richness.
      0 – fewer than 1 field keyword hit
      1 – 1-4 field keyword hits
      2 – 5+ field keyword hits
    """
    if career_label == "tech":
        field_hits = _count_keywords(text_lower, TECH_HINTS)
    elif career_label == "creative":
        field_hits = _count_keywords(text_lower, CREATIVE_HINTS)
    elif career_label in {"marketing"}:
        field_hits = _count_keywords(text_lower, MARKETING_HINTS)
    elif career_label == "business":
        field_hits = _count_keywords(text_lower, BUSINESS_HINTS)
    else:
        field_hits = 0

    if field_hits >= 5:
        return 2
    elif field_hits >= 1:
        return 1
    else:
        return 0


# ── Master scorer ─────────────────────────────────────────────────────────────

def rule_based_score(text: str, career_label: str = "other") -> dict:
    """
    Score a resume using the same 4-dimension rubric as the gold dataset.

    Returns a dict with:
        ats_score          int  0-2
        writing_score      int  0-2
        achievement_score  int  0-2
        relevance_score    int  0-2
        final_score        float 0-100
        quality_label      int  0/1/2
    """
    if not isinstance(text, str) or len(text.strip()) < 50:
        return {
            "ats_score": 0, "writing_score": 0,
            "achievement_score": 0, "relevance_score": 0,
            "final_score": 0.0, "quality_label": 0,
        }

    text_lower = text.lower()
    lines      = [l.strip() for l in text.split("\n") if l.strip()]
    words      = text.split()

    ats   = _score_ats(text, text_lower, lines)
    writ  = _score_writing(text, text_lower, lines, words)
    ach   = _score_achievement(text, text_lower, lines)
    rel   = _score_relevance(text_lower, career_label)

    # Weighted combination → 0-100
    # Weights calibrated so the distribution matches the gold dataset:
    #   ats 20%, writing 30%, achievement 30%, relevance 20%
    raw = (ats * 20 / 2) + (writ * 30 / 2) + (ach * 30 / 2) + (rel * 20 / 2)
    final_score = round(min(100.0, max(0.0, raw)), 2)

    quality_label = score_to_grade(final_score)

    return {
        "ats_score":         ats,
        "writing_score":     writ,
        "achievement_score": ach,
        "relevance_score":   rel,
        "final_score":       final_score,
        "quality_label":     quality_label,
    }


def score_to_grade(final_score: float) -> int:
    """Convert numeric score to 3-class quality_label (matches gold dataset)."""
    if final_score >= 70:
        return 2   # strong
    elif final_score >= 40:
        return 1   # average
    else:
        return 0   # weak


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw resumes...")

    chunks = []
    for chunk in pd.read_csv(
        INPUT,
        usecols=["Category", "Resume_str"],
        chunksize=100,
        dtype=str,
        encoding="utf-8",
        on_bad_lines="skip",
    ):
        chunks.append(chunk)

    df = pd.concat(chunks, ignore_index=True)
    df.columns = df.columns.str.strip()

    print(f"  Rows loaded: {len(df)}")

    # Map to career_label (gold dataset column name)
    df["career_label"] = df["Category"].str.strip().str.title().map(CATEGORY_MAP)

    unmapped = df[df["career_label"].isna()]["Category"].unique()
    if len(unmapped):
        print(f"  Unmapped categories: {unmapped}")

    # Rename Resume_str → cv_text (gold dataset column name)
    df = df.rename(columns={"Resume_str": "cv_text"})
    df = df.dropna(subset=["career_label", "cv_text"])
    df["cv_text"] = df["cv_text"].astype(str).str.strip()
    df = df[df["cv_text"].str.len() > 50]

    print(f"  Rows after cleaning: {len(df)}")
    print("Scoring resumes...")

    score_rows = df.apply(
        lambda row: rule_based_score(row["cv_text"], row["career_label"]),
        axis=1,
        result_type="expand",
    )
    df = pd.concat([df, score_rows], axis=1)

    # Final column order matches gold dataset schema
    out_cols = [
        "cv_text", "career_label", "quality_label",
        "ats_score", "writing_score", "achievement_score",
        "relevance_score", "final_score",
    ]
    df = df[out_cols]

    print("\nQuality label distribution:")
    print(df["quality_label"].value_counts().sort_index().rename(
        {0: "weak (0)", 1: "average (1)", 2: "strong (2)"}
    ).to_string())

    print("\nFinal score statistics:")
    print(df["final_score"].describe().round(2).to_string())

    print("\nCareer label distribution:")
    print(df["career_label"].value_counts().to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\n✓ Saved: {OUTPUT}")