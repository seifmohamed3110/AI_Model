"""
modules/ats.py

Responsibilities:
- Check resume text for ATS compatibility issues
- Return a list of user-friendly issue strings

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
    "\u2018",  # left single quote
    "\u2019",  # right single quote
    "\u201c",  # left double quote
    "\u201d",  # right double quote
    "\u00ae",  # registered trademark ®
    "\u2122",  # trademark ™
    "\u00a9",  # copyright ©
]

# Date format patterns
DATE_FORMATS = {
    "MM/YYYY": re.compile(r"\b\d{1,2}/\d{4}\b"),
    "Mon YYYY": re.compile(r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4}\b", re.IGNORECASE),
    "YYYY": re.compile(r"\b(20\d{2}|19\d{2})\b"),
    "MM-YYYY": re.compile(r"\b\d{1,2}-\d{4}\b"),
}


def check_ats(text: str) -> List[str]:
    """
    Check resume text for ATS compatibility issues.
    Returns a list of user-friendly issue description strings.
    """
    issues = []
    lines = [l.strip() for l in text.split("\n") if l.strip()]

    # ── 1. Contact info placement ─────────────────────────────────────────────
    top_section = text[:200]

    has_email_top = bool(re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", top_section
    ))
    has_phone_top = bool(re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)", top_section
    ))

    if not has_email_top:
        issues.append(
            "Place your email address near the top of the resume so it is easy for ATS systems and recruiters to find."
        )

    if not has_phone_top:
        issues.append(
            "Place your phone number near the top of the resume so it is easy for ATS systems and recruiters to find."
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
            "Use a clear 'Experience' section heading so ATS systems can recognize your work history more easily."
        )

    if "education" not in found_headings and "academic background" not in found_headings:
        issues.append(
            "Use a clear 'Education' section heading so ATS systems can recognize your academic background more easily."
        )

    if not {"skills", "technical skills", "core competencies"}.intersection(found_headings):
        issues.append(
            "Add a clear 'Skills' section heading so ATS systems can identify important keywords more easily."
        )

    # ── 3. Non-standard headings ──────────────────────────────────────────────
    for line in lines:
        clean = line.lower().rstrip(":").strip()
        if clean in BAD_HEADINGS:
            issues.append(
                f"Replace the heading '{line.strip()}' with a more standard heading such as Experience, Skills, or Education."
            )

    # ── 4. Special characters ─────────────────────────────────────────────────
    found_special = [c for c in PROBLEMATIC_CHARS if c in text]
    if found_special:
        issues.append(
            "Some special symbols may make ATS reading less reliable. Use simple standard characters where possible, especially for bullets and dashes."
        )

    # ── 5. Mixed date formats ─────────────────────────────────────────────────
    found_date_formats = []
    for fmt_name, pattern in DATE_FORMATS.items():
        if pattern.search(text):
            found_date_formats.append(fmt_name)

    if len(found_date_formats) > 1:
        issues.append(
            f"Use one date format consistently throughout the resume. Right now multiple formats appear ({', '.join(found_date_formats)})."
        )

    # ── 6. Two-column layout detection ───────────────────────────────────────
    short_lines = sum(1 for l in lines if len(l) < 20)
    if len(lines) > 10 and short_lines / len(lines) > 0.4:
        issues.append(
            "The layout may be too complex for some ATS systems. A single-column layout is usually safer and easier to read."
        )

    # ── 7. Images or graphics (inferred) ─────────────────────────────────────
    word_count = len(text.split())
    if word_count < 100:
        issues.append(
            "Very little text was detected. If your resume uses images, icons, or graphics, some ATS systems may not read them well."
        )

    # ── 8. Tables (inferred from pipe characters) ─────────────────────────────
    pipe_lines = sum(1 for l in lines if "|" in l)
    if pipe_lines >= 3:
        issues.append(
            "Table-style formatting may reduce ATS readability. Plain text lists are usually safer."
        )

    # ── 9. Headers and footers ────────────────────────────────────────────────
    if len(lines) > 10:
        first_line = lines[0].lower().strip()
        last_line = lines[-1].lower().strip()
        if first_line == last_line and len(first_line) > 5:
            issues.append(
                "Some repeated content may be coming from a header or footer. Keep important resume details in the main body where ATS systems read them more reliably."
            )

    return issues