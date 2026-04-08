"""
modules/writing.py

Responsibilities:
- Check resume text for writing quality issues
- Return a list of issue strings

Checks performed:
- First person pronouns
- Filler and weak phrases
- Wrong brand capitalizations (Github vs GitHub)
- Period inconsistency across bullet points
- Double spaces
- Passive voice indicators
- Buzzword overuse

No Flask. No ML. No file I/O.
"""

import re
from typing import List


# ── Brand capitalization map ──────────────────────────────────────────────────
# wrong → correct
BRAND_CAPS = {
    "github":     "GitHub",
    "linkedin":   "LinkedIn",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "postgresql": "PostgreSQL",
    "mongodb":    "MongoDB",
    "mysql":      "MySQL",
    "devops":     "DevOps",
    "tensorflow": "TensorFlow",
    "pytorch":    "PyTorch",
    "scikit-learn": "scikit-learn",
    "numpy":      "NumPy",
    "pandas":     "pandas",
    "fastapi":    "FastAPI",
    "graphql":    "GraphQL",
    "nosql":      "NoSQL",
    "chatgpt":    "ChatGPT",
    "openai":     "OpenAI",
    "powerpoint": "PowerPoint",
    "excel":      "Excel",
    "wordpress":  "WordPress",
    "shopify":    "Shopify",
    "hubspot":    "HubSpot",
    "salesforce": "Salesforce",
}

# ── Filler and weak phrases ───────────────────────────────────────────────────
FILLER_PHRASES = [
    "responsible for",
    "duties included",
    "worked on",
    "helped with",
    "assisted in",
    "involved in",
    "participated in",
    "tasked with",
    "was part of",
    "team player",
    "hard worker",
    "detail oriented",
    "go getter",
    "think outside the box",
    "synergy",
    "leverage",
    "utilize",
    "utilized",
    "dynamic",
    "passionate about",
    "results driven",
    "proven track record",
    "various tasks",
]

# ── Passive voice indicators ──────────────────────────────────────────────────
PASSIVE_PATTERNS = [
    r"\bwas (responsible|involved|tasked|assigned)\b",
    r"\bwere (responsible|involved|tasked|assigned)\b",
    r"\bhas been\b",
    r"\bhave been\b",
]


def _get_bullet_lines(lines: List[str]) -> List[str]:
    """Return lines that start with a bullet character or dash."""
    return [
        l for l in lines
        if re.match(r"^\s*[\•\-\*\–\—\►\▪\○\●]", l)
    ]


def check_writing(text: str) -> List[str]:
    """
    Check resume text for writing quality issues.
    Returns a list of issue description strings.
    """
    issues = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text_lower = text.lower()

    # ── 1. First person pronouns ──────────────────────────────────────────────
    first_person = re.findall(r"\b(I|me|my|myself|we|our)\b", text)
    if len(first_person) > 0:
        examples = list(set(first_person))[:3]
        issues.append(
            f"First-person pronouns found ({', '.join(examples)}) — resumes should use implied subject, e.g. 'Led a team' not 'I led a team'"
        )

    # ── 2. Filler and weak phrases ────────────────────────────────────────────
    found_fillers = []
    for phrase in FILLER_PHRASES:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text_lower):
            found_fillers.append(phrase)

    if found_fillers:
        examples = found_fillers[:3]
        issues.append(
            f"Weak or filler phrases found: '{', '.join(examples)}' — replace with strong action verbs and specific achievements"
        )

    # ── 3. Wrong brand capitalizations ───────────────────────────────────────
    wrong_brands = []
    for wrong, correct in BRAND_CAPS.items():
        # Look for the wrong version appearing in the text
        # Match whole word, case insensitive, but not already correctly cased
        pattern = r"\b" + re.escape(wrong) + r"\b"
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if match != correct:
                wrong_brands.append(f"'{match}' → '{correct}'")

    if wrong_brands:
        examples = wrong_brands[:4]
        issues.append(
            f"Incorrect brand capitalization: {', '.join(examples)}"
        )

    # ── 4. Period inconsistency across bullet points ──────────────────────────
    bullet_lines = _get_bullet_lines(lines)
    if len(bullet_lines) >= 3:
        ends_with_period = [l.rstrip().endswith(".") for l in bullet_lines]
        has_periods    = sum(ends_with_period)
        missing_periods = len(ends_with_period) - has_periods

        if has_periods > 0 and missing_periods > 0:
            issues.append(
                f"Inconsistent punctuation in bullet points — {has_periods} bullets end with a period, {missing_periods} do not. Pick one style and apply it consistently"
            )

    # ── 5. Double spaces ──────────────────────────────────────────────────────
    double_spaces = len(re.findall(r"  +", text))
    if double_spaces > 2:
        issues.append(
            f"Double spaces found ({double_spaces} instances) — these are invisible but can cause ATS parsing issues"
        )

    # ── 6. Passive voice ──────────────────────────────────────────────────────
    passive_hits = []
    for pattern in PASSIVE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        passive_hits.extend(matches)

    if passive_hits:
        issues.append(
            "Passive voice detected — use active voice and strong action verbs instead, e.g. 'Led the team' not 'Was responsible for leading the team'"
        )

    # ── 7. All caps words (excluding abbreviations) ───────────────────────────
    all_caps_words = re.findall(r"\b[A-Z]{4,}\b", text)
    # Filter out known acceptable abbreviations
    acceptable = {"HTML", "CSS", "JSON", "REST", "API", "SQL", "AWS", "GCP",
                  "CRM", "ERP", "KPI", "ROI", "SEM", "SEO", "PPC", "MVP",
                  "CEO", "CTO", "CFO", "COO", "HR", "IT", "UI", "UX", "CI",
                  "CD", "QA", "NLP", "ML", "AI", "DL", "ETL", "SLA", "OKR"}
    unexpected_caps = [w for w in all_caps_words if w not in acceptable]

    if len(unexpected_caps) > 3:
        examples = list(set(unexpected_caps))[:3]
        issues.append(
            f"Excessive use of ALL CAPS words ({', '.join(examples)}...) — use normal capitalization for readability"
        )

    # ── 8. Very long bullets ──────────────────────────────────────────────────
    long_bullets = [
        l for l in bullet_lines
        if len(l.split()) > 30
    ]
    if long_bullets:
        issues.append(
            f"{len(long_bullets)} bullet point(s) are too long (over 30 words) — keep bullets concise, ideally 15-20 words"
        )

    # ── 9. Repetitive words ───────────────────────────────────────────────────
    words = re.findall(r"\b[a-z]{5,}\b", text_lower)
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

    # Exclude common resume words
    exclude = {
        "experience", "skills", "education", "management", "project",
        "development", "business", "company", "worked", "working",
        "responsible", "including", "position", "within", "through",
        "between", "various", "across", "during", "while", "about",
        "their", "these", "those", "other", "where", "which", "there",
    }
    overused = [
        w for w, count in word_freq.items()
        if count >= 5 and w not in exclude
    ]

    if overused:
        examples = sorted(overused, key=lambda w: -word_freq[w])[:3]
        issues.append(
            f"Repetitive words detected: '{', '.join(examples)}' — vary your language to avoid sounding repetitive"
        )

    return issues