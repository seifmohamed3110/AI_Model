"""
modules/ats.py

Responsibilities:
- Check resume text for ATS compatibility issues
- Return a list of issue strings

Checks performed:
- Bad or missing section headings
- Special characters that confuse parsers
- Mixed date formats
- Contact info placement
- Two-column layout detection
- Tables and text boxes (inferred from formatting)
- File name is not checked here — that is app.py's job

No Flask. No ML. No file I/O.
"""

import re
from typing import List


# ── Known good section headings ───────────────────────────────────────────────
STANDARD_HEADINGS = {
    "summary", "objective", "profile", "professional summary",
    "work experience", "experience", "employment", "employment history",
    "education", "academic background",
    "skills", "technical skills", "core competencies",
    "certifications", "licenses", "awards", "achievements",
    "projects", "portfolio", "publications",
    "languages", "interests", "hobbies",
    "references", "volunteer", "volunteering",
}

# Headings that are non-standard and confuse ATS
BAD_HEADINGS = [
    "what i bring", "my experience", "my skills", "about me",
    "who i am", "my story", "career highlights", "stuff i know",
    "things i've done", "my journey",
]

# Special characters that commonly break ATS parsers
PROBLEMATIC_CHARS = [
    "\u2022",  # bullet •
    "\u2013",  # en dash –
    "\u2014",  # em dash —
    "\u2018",  # left single quote '
    "\u2019",  # right single quote '
    "\u201c",  # left double quote "
    "\u201d",  # right double quote "
    "\u00ae",  # registered trademark ®
    "\u2122",  # trademark ™
    "\u00a9",  # copyright ©
]

# Date format patterns
DATE_FORMATS = {
    "mm/yyyy":   re.compile(r"\b\d{1,2}/\d{4}\b"),
    "mon_yyyy":  re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b", re.IGNORECASE),
    "yyyy":      re.compile(r"\b(20\d{2}|19\d{2})\b"),
    "mm-yyyy":   re.compile(r"\b\d{1,2}-\d{4}\b"),
}


def check_ats(text: str) -> List[str]:
    """
    Check resume text for ATS compatibility issues.
    Returns a list of issue description strings.
    """
    issues = []
    lines  = [l.strip() for l in text.split("\n") if l.strip()]
    text_lower = text.lower()

    # ── 1. Contact info placement ─────────────────────────────────────────────
    first_200 = text[:200].lower()
    has_email_top = bool(re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text[:200]
    ))
    has_phone_top = bool(re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)", text[:200]
    ))
    if not has_email_top:
        issues.append(
            "Email address not found in the top section — ATS may miss it"
        )
    if not has_phone_top:
        issues.append(
            "Phone number not found in the top section — ATS may miss it"
        )

    # ── 2. Missing standard sections ─────────────────────────────────────────
    found_headings = set()
    for line in lines:
        clean = line.lower().rstrip(":").strip()
        if clean in STANDARD_HEADINGS and len(line) < 40:
            found_headings.add(clean)

    critical = {"experience", "work experience", "employment"}
    if not critical.intersection(found_headings):
        issues.append(
            "No 'Experience' section heading found — ATS expects a standard heading"
        )

    if "education" not in found_headings and "academic background" not in found_headings:
        issues.append(
            "No 'Education' section heading found — ATS expects a standard heading"
        )

    if not {"skills", "technical skills", "core competencies"}.intersection(found_headings):
        issues.append(
            "No 'Skills' section heading found — ATS expects a standard heading"
        )

    # ── 3. Non-standard headings ──────────────────────────────────────────────
    for line in lines:
        clean = line.lower().rstrip(":").strip()
        if clean in BAD_HEADINGS:
            issues.append(
                f"Non-standard section heading '{line.strip()}' may confuse ATS — use standard names like Experience, Skills"
            )

    # ── 4. Special characters ─────────────────────────────────────────────────
    found_special = [c for c in PROBLEMATIC_CHARS if c in text]
    if found_special:
        issues.append(
            f"Special characters found ({len(found_special)} types) — these can corrupt ATS parsing. Use plain hyphens and ASCII characters"
        )

    # ── 5. Mixed date formats ─────────────────────────────────────────────────
    found_date_formats = []
    for fmt_name, pattern in DATE_FORMATS.items():
        if pattern.search(text):
            found_date_formats.append(fmt_name)

    if len(found_date_formats) > 1:
        issues.append(
            f"Mixed date formats detected ({', '.join(found_date_formats)}) — use one consistent format throughout, e.g. 'Jan 2022'"
        )

    # ── 6. Two-column layout detection ───────────────────────────────────────
    # Heuristic: many lines are very short (< 20 chars) next to long lines
    # suggests side-by-side columns which ATS reads out of order
    short_lines = sum(1 for l in lines if len(l) < 20)
    if len(lines) > 10 and short_lines / len(lines) > 0.4:
        issues.append(
            "Possible two-column layout detected — ATS reads left-to-right and may scramble column content. Use a single-column layout"
        )

    # ── 7. Images or graphics (inferred) ─────────────────────────────────────
    # If text is very short for a resume it may be image-heavy
    word_count = len(text.split())
    if word_count < 100:
        issues.append(
            "Very little text extracted — resume may contain images or graphics that ATS cannot read"
        )

    # ── 8. Tables (inferred from pipe characters) ─────────────────────────────
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines >= 3:
        issues.append(
            "Possible table formatting detected — ATS often cannot parse tables correctly. Use plain text lists instead"
        )

    # ── 9. Headers and footers ────────────────────────────────────────────────
    # Repeated content at start and end suggests header/footer
    if len(lines) > 10:
        first_line = lines[0].lower().strip()
        last_line  = lines[-1].lower().strip()
        if first_line == last_line and len(first_line) > 5:
            issues.append(
                "Repeated content at top and bottom suggests a header/footer — ATS may duplicate or misplace this content"
            )

    return issues