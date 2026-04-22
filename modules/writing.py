"""
modules/writing.py

Responsibilities:
- Check resume text for writing quality issues
- Return issue strings for backward compatibility
- Provide structured writing analysis with score, strengths, and issues

Checks performed:
- First person pronouns
- Filler and weak phrases
- Wrong brand capitalizations
- Period inconsistency across bullet points
- Double spaces
- Passive voice indicators
- Excessive ALL CAPS words (with abbreviation-safe handling)
- Very long bullets
- Repetitive words

No Flask. No ML. No file I/O.
"""

import re
from typing import List, Dict, Any


# ── Brand capitalization map ──────────────────────────────────────────────────
# wrong → correct
BRAND_CAPS = {
    "github": "GitHub",
    "linkedin": "LinkedIn",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "postgresql": "PostgreSQL",
    "mongodb": "MongoDB",
    "mysql": "MySQL",
    "devops": "DevOps",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "numpy": "NumPy",
    "pandas": "pandas",
    "fastapi": "FastAPI",
    "graphql": "GraphQL",
    "nosql": "NoSQL",
    "chatgpt": "ChatGPT",
    "openai": "OpenAI",
    "powerpoint": "PowerPoint",
    "excel": "Excel",
    "wordpress": "WordPress",
    "shopify": "Shopify",
    "hubspot": "HubSpot",
    "salesforce": "Salesforce",
}

# ── Acceptable abbreviations / certifications / acronyms ─────────────────────
ACCEPTABLE_CAPS = {
    "HTML", "CSS", "JSON", "REST", "API", "SQL", "NOSQL", "XML",
    "AWS", "GCP", "SDK", "CLI", "SRE", "ETL", "ELT", "DBMS",
    "CRM", "ERP", "KPI", "ROI", "OKR", "PNL", "B2B", "B2C",
    "SEM", "SEO", "PPC", "CTR", "CPC", "CPA", "CPM",
    "CEO", "CTO", "CFO", "COO", "HR", "IT", "PM", "QA",
    "UI", "UX",
    "NLP", "ML", "AI", "DL", "LLM", "RAG",
    "CI", "CD", "SLA", "SLO", "SLI",
    "PMP", "CFA", "CPA", "ACCA", "FRM", "MBA", "PHR", "SHRM",
    "CSM", "PSM", "CSPO", "ITIL", "TOEFL", "IELTS", "SAT", "GRE", "GMAT",
    "MVP", "SKU", "GPA",
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


def analyze_writing(text: str) -> Dict[str, Any]:
    """
    Structured writing analysis.
    Returns:
        {
            "score": float,      # 0.0 to 5.0
            "band": str,
            "strengths": List[str],
            "issues": List[str]
        }
    """
    issues: List[str] = []
    strengths: List[str] = []

    lines = [l.strip() for l in text.split("\n") if l.strip()]
    text_lower = text.lower()

    # Start from a full score and subtract penalties
    score = 5.0

    # ── 1. First person pronouns ──────────────────────────────────────────────
    first_person = re.findall(r"\b(I|me|my|myself|we|our)\b", text)
    if first_person:
        examples = list(dict.fromkeys(first_person))[:3]
        issues.append(
            f"First-person pronouns found ({', '.join(examples)}) — resumes should use implied subject, e.g. 'Led a team' not 'I led a team'"
        )
        score -= 0.5
    else:
        strengths.append("Uses professional resume style without first-person pronouns")

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
        score -= min(0.8, 0.2 * len(found_fillers))
    else:
        strengths.append("Uses stronger, more direct language without obvious filler phrases")

    # ── 3. Wrong brand capitalizations ────────────────────────────────────────
    wrong_brands = []
    for wrong, correct in BRAND_CAPS.items():
        pattern = r"\b" + re.escape(wrong) + r"\b"

        for match in re.finditer(pattern, text, re.IGNORECASE):
            found_text = match.group(0)
            start, end = match.span()

            left_context = text[max(0, start - 20):start].lower()
            right_context = text[end:min(len(text), end + 20)].lower()

            # Ignore URLs, domains, and email/domain contexts
            if (
                "http://" in left_context
                or "https://" in left_context
                or "www." in left_context
                or ".com" in right_context
                or ".net" in right_context
                or ".org" in right_context
                or ".io" in right_context
                or ".dev" in right_context
                or "@" in left_context
            ):
                continue

            if found_text != correct:
                wrong_brands.append(f"'{found_text}' → '{correct}'")

    wrong_brands = list(dict.fromkeys(wrong_brands))
    if wrong_brands:
        examples = wrong_brands[:4]
        issues.append(
            f"Incorrect brand capitalization: {', '.join(examples)}"
        )
        score -= 0.4
    else:
        strengths.append("Brand and technology names are capitalized consistently")

    # ── 4. Period inconsistency across bullet points ──────────────────────────
    bullet_lines = _get_bullet_lines(lines)
    if len(bullet_lines) >= 3:
        ends_with_period = [l.rstrip().endswith(".") for l in bullet_lines]
        has_periods = sum(ends_with_period)
        missing_periods = len(ends_with_period) - has_periods

        if has_periods > 0 and missing_periods > 0:
            issues.append(
                f"Inconsistent punctuation in bullet points — {has_periods} bullets end with a period, {missing_periods} do not. Pick one style and apply it consistently"
            )
            score -= 0.3
        else:
            strengths.append("Bullet punctuation is consistent")

    # ── 5. Double spaces ──────────────────────────────────────────────────────
    double_spaces = len(re.findall(r"  +", text))
    if double_spaces > 2:
        issues.append(
            f"Double spaces found ({double_spaces} instances) — these are invisible but can cause ATS parsing issues"
        )
        score -= 0.2

    # ── 6. Passive voice ──────────────────────────────────────────────────────
    passive_hits = []
    for pattern in PASSIVE_PATTERNS:
        matches = re.findall(pattern, text_lower)
        passive_hits.extend(matches)

    if passive_hits:
        issues.append(
            "Passive voice detected — use active voice and strong action verbs instead, e.g. 'Led the team' not 'Was responsible for leading the team'"
        )
        score -= 0.4
    else:
        strengths.append("Uses mostly active voice")

    # ── 7. All caps words (excluding abbreviations / certifications) ─────────
    all_caps_words = re.findall(r"\b[A-Z]{2,}\b", text)

    unexpected_caps = []
    for w in all_caps_words:
        if w in ACCEPTABLE_CAPS:
            continue
        if len(w) <= 5:
            continue
        unexpected_caps.append(w)

    unique_unexpected_caps = list(dict.fromkeys(unexpected_caps))
    if len(unique_unexpected_caps) > 3:
        examples = unique_unexpected_caps[:3]
        issues.append(
            f"Excessive use of ALL CAPS words ({', '.join(examples)}...) — use normal capitalization for readability"
        )
        score -= 0.5
    else:
        if len(all_caps_words) > 0:
            strengths.append("Uses abbreviations and acronyms without obvious capitalization misuse")

    # ── 8. Very long bullets ──────────────────────────────────────────────────
    long_bullets = [
        l for l in bullet_lines
        if len(l.split()) > 30
    ]
    if long_bullets:
        issues.append(
            f"{len(long_bullets)} bullet point(s) are too long (over 30 words) — keep bullets concise, ideally 15-20 words"
        )
        score -= min(0.5, 0.15 * len(long_bullets))
    else:
        if bullet_lines:
            strengths.append("Bullet points are concise and readable")

    # ── 9. Repetitive words ───────────────────────────────────────────────────
    words = re.findall(r"\b[a-z]{5,}\b", text_lower)
    word_freq = {}
    for w in words:
        word_freq[w] = word_freq.get(w, 0) + 1

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
        score -= 0.4
    else:
        strengths.append("Avoids obvious repetitive wording")

    # Clamp score
    score = max(0.0, min(5.0, round(score, 1)))

    # Score band
    if score >= 4.5:
        band = "Excellent"
    elif score >= 3.8:
        band = "Good"
    elif score >= 2.8:
        band = "Average"
    else:
        band = "Needs Improvement"

    strengths = strengths[:4]

    return {
        "score": score,
        "band": band,
        "strengths": strengths,
        "issues": issues,
    }


def check_writing(text: str) -> List[str]:
    """
    Backward-compatible wrapper.
    Returns only the issue list so older callers still work.
    """
    return analyze_writing(text)["issues"]