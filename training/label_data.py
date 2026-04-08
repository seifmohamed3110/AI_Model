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
INPUT    = os.path.join(DATA_DIR, "raw_resumes.csv")
OUTPUT   = os.path.join(DATA_DIR, "labeled_resumes.csv")


# ── Category → field mapping ─────────────────────────────────────────────────
CATEGORY_MAP = {
    "Information-Technology": "tech",
    "Engineering":            "tech",

    "Designer":               "creative",
    "Arts":                   "creative",
    "Digital-Media":          "creative",
    "Apparel":                "creative",

    "Business-Development":   "business",
    "Sales":                  "business",
    "Consultant":             "business",
    "BPO":                    "business",
    "Finance":                "business",
    "Accountant":             "business",
    "Banking":                "business",
    "HR":                     "business",
    "Hr":                     "business",
    "BPO":                    "business",
    "Bpo":                    "business",
    "Public-Relations":       "business",

    "Healthcare":             "other",
    "Fitness":                "other",
    "Teacher":                "other",
    "Agriculture":            "other",
    "Automobile":             "other",
    "Chef":                   "other",
    "Aviation":               "other",
    "Construction":           "other",
    "Advocate":               "other",
}


# ── Rule-based scorer ─────────────────────────────────────────────────────────

def rule_based_score(text: str) -> float:
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

    # ── Word count (max 15 pts) ───────────────────────────────────────────────
    wc = len(words)
    if wc >= 400:
        score += 15
    elif wc >= 250:
        score += 10
    elif wc >= 150:
        score += 5

    # ── Contact info (max 15 pts) ─────────────────────────────────────────────
    if re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text):
        score += 5
    if re.search(r"(\+?\d[\d\s\-\(\)]{7,}\d)", text):
        score += 5
    if re.search(r"linkedin\.com", text_lower):
        score += 5

    # ── Sections (max 20 pts) ─────────────────────────────────────────────────
    section_keywords = [
        "experience", "education", "skills", "summary",
        "objective", "certifications", "projects", "achievements"
    ]
    found = sum(1 for kw in section_keywords if kw in text_lower)
    score += min(found * 4, 20)

    # ── Bullet points (max 10 pts) ────────────────────────────────────────────
    bullets = [l for l in lines if re.match(r"^[\•\-\*\–\—\►\▪\○\●]", l)]
    if len(bullets) >= 6:
        score += 10
    elif len(bullets) >= 3:
        score += 5

    # ── Quantification (max 10 pts) ───────────────────────────────────────────
    quant = len(re.findall(
        r"\b\d+\s*(%|percent|x|times|million|billion|k\b|users|customers|revenue)", text_lower
    ))
    if quant >= 3:
        score += 10
    elif quant >= 1:
        score += 5

    # ── Action verbs (max 10 pts) ─────────────────────────────────────────────
    action_verbs = [
        "achieved", "built", "created", "delivered", "designed", "developed",
        "drove", "established", "executed", "generated", "implemented",
        "improved", "increased", "launched", "led", "managed", "optimized",
        "reduced", "scaled", "spearheaded", "streamlined", "transformed",
    ]
    verb_count = sum(
        1 for v in action_verbs
        if re.search(r"\b" + v + r"\b", text_lower)
    )
    if verb_count >= 5:
        score += 10
    elif verb_count >= 2:
        score += 5

    # ── Penalties (max -20 pts) ───────────────────────────────────────────────
    filler_phrases = [
        "responsible for", "duties included", "worked on",
        "helped with", "assisted in", "involved in",
    ]
    filler_count = sum(
        len(re.findall(r"\b" + re.escape(p) + r"\b", text_lower))
        for p in filler_phrases
    )
    score -= min(filler_count * 3, 15)

    first_person = len(re.findall(r"\b(i|me|my|myself)\b", text_lower))
    score -= min(first_person * 1, 5)

    return round(max(0.0, min(100.0, score)), 2)


def score_to_grade(score: float) -> int:
    """Convert numeric score to 3-class label."""
    if score >= 50:
        return 2   # strong
    elif score >= 30:
        return 1   # average
    else:
        return 0   # weak


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading raw resumes...")

    # Only load the columns we need - skip Resume_html entirely
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

    # Map categories to fields
    df["field"] = df["Category"].str.strip().str.title().map(CATEGORY_MAP)
    unmapped = df[df["field"].isna()]["Category"].unique()
    if len(unmapped):
        print(f"  Unmapped categories: {unmapped}")

    df = df.dropna(subset=["field", "Resume_str"])
    df["Resume_str"] = df["Resume_str"].astype(str).str.strip()
    df = df[df["Resume_str"].str.len() > 50]

    print(f"  Rows after cleaning: {len(df)}")

    print("Scoring resumes...")
    df["score"] = df["Resume_str"].apply(rule_based_score)
    df["grade"] = df["score"].apply(score_to_grade)

    print(f"\nGrade distribution:")
    print(df["grade"].value_counts().sort_index().rename(
        {0: "weak", 1: "average", 2: "strong"}
    ).to_string())

    print(f"\nScore statistics:")
    print(df["score"].describe().round(2).to_string())

    df.to_csv(OUTPUT, index=False)
    print(f"\n✓ Saved: {OUTPUT}")