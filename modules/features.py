"""
modules/features.py

Responsibilities:
- Take clean resume text
- Return a flat dictionary of numeric features
- Used by scorer.py (ML input) and suggestions.py (logic input)

No Flask. No ML. No file I/O.
"""

import re


# ── Keyword lists ────────────────────────────────────────────────────────────

TECH_SKILLS = [
    "python", "java", "javascript", "typescript", "c++", "c#", "sql", "nosql",
    "react", "angular", "vue", "node", "django", "flask", "fastapi", "spring",
    "aws", "azure", "gcp", "docker", "kubernetes", "git", "linux", "bash",
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn",
    "pandas", "numpy", "data science", "api", "rest", "graphql", "mongodb",
    "postgresql", "mysql", "redis", "elasticsearch", "spark", "hadoop",
    "ci/cd", "devops", "agile", "scrum", "terraform", "ansible",
]

MARKETING_SKILLS = [
    "seo", "sem", "google analytics", "google ads", "facebook ads",
    "content marketing", "email marketing", "social media", "hubspot",
    "salesforce", "crm", "brand", "campaign", "copywriting", "conversion",
    "a/b testing", "marketing automation", "mailchimp", "hootsuite",
    "market research", "lead generation", "ppc", "roi", "kpi",
]

CREATIVE_SKILLS = [
    "photoshop", "illustrator", "indesign", "figma", "sketch", "adobe xd",
    "after effects", "premiere pro", "lightroom", "canva", "typography",
    "branding", "ux", "ui", "user experience", "wireframe", "prototype",
    "illustration", "photography", "video editing", "motion graphics",
    "creative direction", "art direction", "portfolio",
]

BUSINESS_SKILLS = [
    "project management", "stakeholder", "budget", "forecasting", "strategy",
    "operations", "supply chain", "procurement", "vendor", "compliance",
    "risk management", "financial analysis", "excel", "powerpoint",
    "business development", "p&l", "revenue", "cost reduction",
    "team leadership", "cross-functional", "kpi", "okr", "six sigma",
    "process improvement", "change management",
]

FILLER_PHRASES = [
    "responsible for", "duties included", "worked on", "helped with",
    "assisted in", "involved in", "participated in", "tasked with",
    "was part of", "contributed to",
]

ACTION_VERBS = [
    "achieved", "built", "created", "delivered", "designed", "developed",
    "drove", "established", "executed", "generated", "implemented", "improved",
    "increased", "launched", "led", "managed", "optimized", "reduced",
    "scaled", "spearheaded", "streamlined", "transformed", "architected",
    "automated", "coordinated", "deployed", "engineered", "founded",
    "grew", "hired", "negotiated", "produced", "raised", "recruited",
]

SECTION_HEADINGS = [
    "experience", "work experience", "employment", "education", "skills",
    "summary", "objective", "profile", "certifications", "projects",
    "achievements", "awards", "publications", "languages", "interests",
    "references", "volunteer", "internship",
]


# ── Helper functions ─────────────────────────────────────────────────────────

def _count_keywords(text: str, keywords: list) -> int:
    """Count how many keywords from the list appear in the text."""
    count = 0
    for kw in keywords:
        if re.search(r"\b" + re.escape(kw) + r"\b", text):
            count += 1
    return count


def _is_bullet_line(line: str) -> bool:
    """Return True if the line starts with a bullet-like character or dash."""
    stripped = line.strip()
    return bool(re.match(r"^[\•\-\*\–\—\►\▪\○\●]", stripped))


# ── Main function ─────────────────────────────────────────────────────────────

def extract_features(text: str) -> dict:
    """
    Takes clean resume text.
    Returns a flat dictionary of numeric features.
    """
    text_lower = text.lower()
    lines = text.split("\n")
    non_empty_lines = [l for l in lines if l.strip()]

    # ── Basic counts ─────────────────────────────────────────────────────────
    words = text.split()
    word_count = len(words)
    char_count = len(text)
    line_count = len(non_empty_lines)
    avg_line_length = (
        sum(len(l) for l in non_empty_lines) / line_count
        if line_count > 0 else 0
    )

    # ── Contact info ─────────────────────────────────────────────────────────
    has_email = int(bool(re.search(
        r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", text
    )))
    has_phone = int(bool(re.search(
        r"(\+?\d[\d\s\-\(\)]{7,}\d)", text
    )))
    has_linkedin = int(bool(re.search(
        r"linkedin\.com", text_lower
    )))
    has_github = int(bool(re.search(
        r"github\.com", text_lower
    )))
    has_portfolio = int(bool(re.search(
        r"(portfolio|behance\.net|dribbble\.com|personal site|mysite|website)", text_lower
    )))

    # ── Section detection ────────────────────────────────────────────────────
    found_sections = []
    for line in non_empty_lines:
        line_clean = line.strip().lower().rstrip(":")
        if line_clean in SECTION_HEADINGS and len(line.strip()) < 40:
            found_sections.append(line_clean)

    section_count    = len(found_sections)
    has_summary      = int(any(s in found_sections for s in ["summary", "objective", "profile"]))
    has_experience   = int(any(s in found_sections for s in ["experience", "work experience", "employment", "internship"]))
    has_education    = int("education" in found_sections)
    has_skills       = int("skills" in found_sections)
    has_certifications = int("certifications" in found_sections)
    has_projects     = int("projects" in found_sections)
    has_achievements = int(any(s in found_sections for s in ["achievements", "awards"]))

    # ── Bullet points ────────────────────────────────────────────────────────
    bullet_lines = [l for l in non_empty_lines if _is_bullet_line(l)]
    bullet_count = len(bullet_lines)
    avg_bullet_length = (
        sum(len(l.split()) for l in bullet_lines) / bullet_count
        if bullet_count > 0 else 0
    )

    # ── Writing quality signals ───────────────────────────────────────────────
    first_person_count = len(re.findall(
        r"\b(i|me|my|myself)\b", text_lower
    ))
    filler_count = sum(
        len(re.findall(r"\b" + re.escape(phrase) + r"\b", text_lower))
        for phrase in FILLER_PHRASES
    )

    # ── Field skill counts ────────────────────────────────────────────────────
    tech_skill_count       = _count_keywords(text_lower, TECH_SKILLS)
    marketing_skill_count  = _count_keywords(text_lower, MARKETING_SKILLS)
    creative_skill_count   = _count_keywords(text_lower, CREATIVE_SKILLS)
    business_skill_count   = _count_keywords(text_lower, BUSINESS_SKILLS)

    # ── Impact signals ────────────────────────────────────────────────────────
    quantification_count = len(re.findall(
        r"\b\d+\s*(%|percent|x|times|million|billion|k\b|users|customers|revenue|hours|days|months)", text_lower
    ))
    action_verb_count = _count_keywords(text_lower, ACTION_VERBS)

    # ── Assemble and return ───────────────────────────────────────────────────
    return {
        "word_count":            word_count,
        "char_count":            char_count,
        "line_count":            line_count,
        "avg_line_length":       round(avg_line_length, 2),
        "has_email":             has_email,
        "has_phone":             has_phone,
        "has_linkedin":          has_linkedin,
        "has_github":            has_github,
        "has_portfolio":         has_portfolio,
        "section_count":         section_count,
        "has_summary":           has_summary,
        "has_experience":        has_experience,
        "has_education":         has_education,
        "has_skills":            has_skills,
        "has_certifications":    has_certifications,
        "has_projects":          has_projects,
        "has_achievements":      has_achievements,
        "bullet_count":          bullet_count,
        "avg_bullet_length":     round(avg_bullet_length, 2),
        "first_person_count":    first_person_count,
        "filler_count":          filler_count,
        "tech_skill_count":      tech_skill_count,
        "marketing_skill_count": marketing_skill_count,
        "creative_skill_count":  creative_skill_count,
        "business_skill_count":  business_skill_count,
        "quantification_count":  quantification_count,
        "action_verb_count":     action_verb_count,
    }


def detect_content_gaps(features: dict, field: str) -> dict:
    """
    Detect personalization gaps used by suggestions ranking logic.
    """
    normalized_field = (field or "other").strip().lower()
    if normalized_field not in {"tech", "marketing", "creative", "business", "other"}:
        normalized_field = "other"

    gaps = {
        "no_github": False,
        "no_portfolio": False,
        "no_metrics": features.get("quantification_count", 0) < 2,
        "no_projects": not bool(features.get("has_projects", 0)),
        "no_achievements": not bool(features.get("has_achievements", 0)),
    }

    if normalized_field == "tech":
        gaps["no_github"] = not bool(features.get("has_github", 0))
    if normalized_field in {"creative", "marketing"}:
        gaps["no_portfolio"] = not bool(features.get("has_portfolio", 0))

    return gaps
def extract_keywords(text: str, top_n: int = 10) -> list:
    """Extract top N keywords from text based on frequency."""
    from collections import Counter
    
    # Convert to lowercase and find words (minimum 3 characters)
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    
    # Common stop words to filter out
    stop_words = {
        "the", "and", "for", "with", "from", "are", "was", "were",
        "have", "has", "had", "but", "not", "you", "your", "our",
        "this", "that", "they", "them", "their", "will", "would",
        "could", "should", "may", "might", "also", "very", "just",
        "what", "which", "when", "where", "who", "whom", "about",
        "than", "then", "there", "their", "can", "did", "does"
    }
    
    # Filter out stop words and short words
    keywords = [word for word in words if word not in stop_words]
    
    # Count frequencies
    keyword_counts = Counter(keywords)
    
    # Return top N keywords
    return [word for word, count in keyword_counts.most_common(top_n)]
