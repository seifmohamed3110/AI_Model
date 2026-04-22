"""
training/label_data.py

Responsibilities:
- Read data/raw_resumes.csv (Kaggle snehaanbhawal/resume-dataset)
- Score each resume using a rule-based function
- Save data/labeled_resumes.csv with score and grade columns

Grade mapping:
    0 = weak    (score < 40)
    1 = average (score 40-69)
    2 = strong  (score >= 70)

Run once from the project root:
    python training/label_data.py
"""

import os
import re
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
INPUT = os.path.join(DATA_DIR, "raw_resumes.csv")
OUTPUT = os.path.join(DATA_DIR, "labeled_resumes.csv")


# ── Category → field mapping ─────────────────────────────────────────────────
CATEGORY_MAP = {
    "Information-Technology": "tech",
    "Engineering": "tech",

    "Designer": "creative",
    "Arts": "creative",
    "Digital-Media": "creative",
    "Apparel": "creative",

    "Business-Development": "business",
    "Sales": "business",
    "Consultant": "business",
    "BPO": "business",
    "Finance": "business",
    "Accountant": "business",
    "Banking": "business",
    "HR": "business",
    "Hr": "business",
    "Bpo": "business",
    "Public-Relations": "business",

    "Healthcare": "other",
    "Fitness": "other",
    "Teacher": "other",
    "Agriculture": "other",
    "Automobile": "other",
    "Chef": "other",
    "Aviation": "other",
    "Construction": "other",
    "Advocate": "other",
}

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


def _keyword_present(text: str, keyword: str) -> bool:
    return bool(re.search(r"\b" + re.escape(keyword) + r"\b", text))


def _count_keywords(text: str, keywords: list) -> int:
    return sum(1 for kw in keywords if _keyword_present(text, kw))


def _detect_sections(lines: list) -> list:
    found_sections = []
    for line in lines:
        clean = line.strip().lower().rstrip(":")
        if clean in SECTION_HEADINGS and len(clean) < 40:
            found_sections.append(clean)
    return found_sections


def rule_based_score(text: str, field: str = "other") -> float:
    """
    Score a resume 0-100 using deterministic rules.
    Used only to generate training labels — not the final scorer.
    """
    if not isinstance(text, str) or len(text.strip()) < 50:
        return 10.0

    score = 0.0
    text_lower = text.lower()
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    words = text.split()

    # ── Word count (max 12 pts) ───────────────────────────────────────────────
    wc = len(words)
    if 250 <= wc <= 650:
        score += 12
    elif 180 <= wc < 250 or 650 < wc <= 900:
        score += 8
    elif 120 <= wc < 180:
        score += 4

    # ── Contact info (max 15 pts) ─────────────────────────────────────────────
    if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
        score += 5
    if re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text):
        score += 5
    if re.search(r"linkedin\.com", text_lower):
        score += 5

    # ── Field-specific links (max 6 pts) ──────────────────────────────────────
    if field == "tech" and re.search(r"github\.com", text_lower):
        score += 6
    elif field in {"creative", "marketing"} and re.search(
        r"(portfolio|behance\.net|dribbble\.com|personal site|website)", text_lower
    ):
        score += 6

    # ── Sections (max 20 pts) ─────────────────────────────────────────────────
    found_sections = _detect_sections(lines)
    section_count = len(found_sections)
    score += min(section_count * 4, 20)

    has_experience = any(s in found_sections for s in ["experience", "work experience", "employment", "internship"])
    has_education = "education" in found_sections
    has_skills = "skills" in found_sections

    if has_experience and has_education and has_skills:
        score += 4

    # ── Bullet points (max 10 pts) ────────────────────────────────────────────
    bullets = [l for l in lines if re.match(r"^[\•\-\*\–\—\►\▪\○\●]", l)]
    bullet_count = len(bullets)
    if bullet_count >= 6:
        score += 10
    elif bullet_count >= 3:
        score += 6
    elif bullet_count >= 1:
        score += 2

    # ── Quantification (max 12 pts) ───────────────────────────────────────────
    quant = len(re.findall(
        r"\b\d+\s*(%|percent|x|times|million|billion|k\b|users|customers|revenue|hours|days|months|years|projects|clients|team members)\b",
        text_lower
    ))
    if quant >= 4:
        score += 12
    elif quant >= 2:
        score += 8
    elif quant >= 1:
        score += 4

    # ── Action verbs (max 10 pts) ─────────────────────────────────────────────
    verb_count = _count_keywords(text_lower, ACTION_VERBS)
    if verb_count >= 6:
        score += 10
    elif verb_count >= 3:
        score += 6
    elif verb_count >= 1:
        score += 3

    # ── Field keyword richness (max 10 pts) ───────────────────────────────────
    if field == "tech":
        field_hits = _count_keywords(text_lower, TECH_HINTS)
    elif field == "marketing":
        field_hits = _count_keywords(text_lower, MARKETING_HINTS)
    elif field == "creative":
        field_hits = _count_keywords(text_lower, CREATIVE_HINTS)
    elif field == "business":
        field_hits = _count_keywords(text_lower, BUSINESS_HINTS)
    else:
        field_hits = 0

    if field_hits >= 6:
        score += 10
    elif field_hits >= 3:
        score += 6
    elif field_hits >= 1:
        score += 3

    # ── Penalties ──────────────────────────────────────────────────────────────
    filler_count = sum(
        len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        for p in FILLER_PHRASES
    )
    score -= min(filler_count * 2, 10)

    first_person = len(re.findall(r"\b(i|me|my|myself|we|our)\b", text_lower))
    score -= min(first_person * 1, 5)

    if wc < 120:
        score -= 8

    missing_core = 0
    if not has_experience:
        missing_core += 1
    if not has_education:
        missing_core += 1
    if not has_skills:
        missing_core += 1
    score -= missing_core * 3

    return round(max(0.0, min(100.0, score)), 2)


def score_to_grade(score: float) -> int:
    """Convert numeric score to 3-class label."""
    if score >= 70:
        return 2   # strong
    elif score >= 40:
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

    df["field"] = df["Category"].str.strip().str.title().map(CATEGORY_MAP)
    unmapped = df[df["field"].isna()]["Category"].unique()
    if len(unmapped):
        print(f"  Unmapped categories: {unmapped}")

    df = df.dropna(subset=["field", "Resume_str"])
    df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
    df = df[df["Resume_str"].str.len() > 50]

    print(f"  Rows after cleaning: {len(df)}")

    print("Scoring resumes...")
    df["score"] = df.apply(lambda row: rule_based_score(row["Resume_str"], row["field"]), axis=1)
    df["grade"] = df["score"].apply(score_to_grade)

    print("\nGrade distribution:")
    print(df["grade"].value_counts().sort_index().rename(
        {0: "weak", 1: "average", 2: "strong"}
    ).to_string())

    print("\nScore statistics:")
    print(df["score"].describe().round(2).to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\n✓ Saved: {OUTPUT}")