"""
modules/suggestions.py

Responsibilities:
- Accept feature dictionary and detected career field
- Return a list of field-specific improvement suggestions
- Never penalize creative/media resumes for missing GitHub
- Never penalize non-tech resumes for missing technical skills

No Flask. No ML. No file I/O.
"""

from typing import List


def get_suggestions(features: dict, field: str) -> List[str]:
    """
    Generate field-specific improvement suggestions.

    Args:
        features: dictionary from extract_features()
        field: one of tech, marketing, creative, business, other

    Returns:
        list of suggestion strings ordered by importance
    """
    suggestions = []
    field = field.lower().strip()

    # ── Universal suggestions (apply to all fields) ───────────────────────────

    if not features.get("has_email"):
        suggestions.append(
            "Add your email address — it is required contact information"
        )

    if not features.get("has_phone"):
        suggestions.append(
            "Add your phone number — recruiters need a way to contact you"
        )

    if not features.get("has_linkedin"):
        suggestions.append(
            "Add your LinkedIn profile URL — most recruiters check LinkedIn before calling"
        )

    if not features.get("has_summary"):
        suggestions.append(
            "Add a Summary or Profile section at the top — give recruiters a 3-sentence overview of who you are"
        )

    if not features.get("has_experience"):
        suggestions.append(
            "Add an Experience section with your work history"
        )

    if not features.get("has_education"):
        suggestions.append(
            "Add an Education section listing your degrees and institutions"
        )

    if not features.get("has_skills"):
        suggestions.append(
            "Add a Skills section — ATS systems scan for keywords here"
        )

    if features.get("word_count", 0) < 150:
        suggestions.append(
            "Your resume is too short — aim for at least 400 words to give enough context"
        )

    if features.get("bullet_count", 0) < 3:
        suggestions.append(
            "Use bullet points to list your achievements — they are easier to scan than paragraphs"
        )

    if features.get("quantification_count", 0) == 0:
        suggestions.append(
            "Add quantified achievements — numbers make your impact concrete, e.g. 'Increased sales by 30%'"
        )
    elif features.get("quantification_count", 0) < 3:
        suggestions.append(
            "Add more quantified achievements — aim for at least 3 metrics across your experience"
        )

    if features.get("action_verb_count", 0) < 3:
        suggestions.append(
            "Start bullet points with strong action verbs like Led, Built, Delivered, Optimized"
        )

    if features.get("filler_count", 0) > 0:
        suggestions.append(
            "Remove filler phrases like 'responsible for' and 'worked on' — replace with direct action verbs"
        )

    if features.get("first_person_count", 0) > 2:
        suggestions.append(
            "Remove first-person pronouns — write 'Led a team of 5' not 'I led a team of 5'"
        )

    if features.get("avg_bullet_length", 0) > 30:
        suggestions.append(
            "Shorten your bullet points — aim for 15-20 words each, focus on the outcome not the process"
        )

    # ── Tech-specific suggestions ─────────────────────────────────────────────
    if field == "tech":
        if not features.get("has_github"):
            suggestions.append(
                "Add your GitHub profile URL — essential for tech roles, recruiters will check your code"
            )
        if features.get("tech_skill_count", 0) < 5:
            suggestions.append(
                "Expand your Skills section with more specific technologies — languages, frameworks, cloud platforms, and tools"
            )
        if not features.get("has_certifications"):
            suggestions.append(
                "Consider adding relevant certifications — AWS, GCP, Azure, or language-specific certs strengthen a tech resume"
            )
        if features.get("quantification_count", 0) < 2:
            suggestions.append(
                "Add technical impact metrics — e.g. 'Reduced API response time by 40%' or 'Scaled system to 1M users'"
            )

    # ── Marketing-specific suggestions ───────────────────────────────────────
    elif field == "marketing":
        if features.get("marketing_skill_count", 0) < 4:
            suggestions.append(
                "Add more marketing tools and platforms to your Skills section — Google Analytics, HubSpot, SEO, SEM, etc."
            )
        if not features.get("has_certifications"):
            suggestions.append(
                "Add marketing certifications — Google Analytics, Google Ads, HubSpot, or Meta Blueprint are highly valued"
            )
        if features.get("quantification_count", 0) < 3:
            suggestions.append(
                "Marketing resumes need strong metrics — add conversion rates, ROI figures, campaign reach, and revenue impact"
            )
        if not features.get("has_portfolio"):
            suggestions.append(
                "Add a link to your portfolio or campaign samples — marketing roles expect to see your work"
            )

    # ── Creative-specific suggestions ────────────────────────────────────────
    elif field == "creative":
        if not features.get("has_portfolio"):
            suggestions.append(
                "Add a portfolio link — essential for creative roles. Use Behance, Dribbble, or your own site"
            )
        if features.get("creative_skill_count", 0) < 4:
            suggestions.append(
                "List the specific tools you use — Figma, Photoshop, Illustrator, After Effects, Premiere Pro, etc."
            )
        if features.get("quantification_count", 0) < 2:
            suggestions.append(
                "Add impact metrics even for creative work — e.g. 'Redesign increased user engagement by 25%' or 'Delivered 50+ brand assets'"
            )
        # Do NOT suggest GitHub for creative resumes

    # ── Business-specific suggestions ────────────────────────────────────────
    elif field == "business":
        if features.get("business_skill_count", 0) < 4:
            suggestions.append(
                "Add business-specific keywords — project management, stakeholder management, P&L, forecasting, etc."
            )
        if features.get("quantification_count", 0) < 3:
            suggestions.append(
                "Business resumes need financial and operational metrics — revenue figures, cost savings, team sizes, budget managed"
            )
        if not features.get("has_certifications"):
            suggestions.append(
                "Consider adding business certifications — PMP, Six Sigma, CPA, CFA, or MBA strengthen a business resume"
            )

    # ── Other-specific suggestions ────────────────────────────────────────────
    elif field == "other":
        if features.get("quantification_count", 0) < 2:
            suggestions.append(
                "Add measurable achievements specific to your field — patient counts, student outcomes, cases handled, etc."
            )
        if not features.get("has_certifications"):
            suggestions.append(
                "Add any relevant licenses or certifications for your field"
            )

    return suggestions


def get_jd_match(resume_text: str, jd_text: str) -> dict:
    """
    Compare resume text against a job description.
    Returns match score and list of missing keywords.

    Args:
        resume_text: clean resume text
        jd_text: job description text pasted by user

    Returns:
        dict with match_score (0-100) and missing_keywords list
    """
    import re

    def extract_keywords(text: str) -> set:
        """Extract meaningful keywords from text."""
        text_lower = text.lower()
        # Remove common stop words
        stop_words = {
            "the", "and", "for", "are", "with", "this", "that", "have",
            "will", "you", "your", "our", "their", "from", "been", "was",
            "were", "has", "had", "not", "but", "they", "all", "can",
            "also", "its", "into", "than", "then", "when", "what", "who",
            "how", "which", "about", "more", "other", "some", "such",
            "work", "using", "must", "able", "well", "good", "etc",
        }
        words = re.findall(r"\b[a-z][a-z0-9\+\#\-\.]{2,}\b", text_lower)
        return {w for w in words if w not in stop_words}

    resume_keywords = extract_keywords(resume_text)
    jd_keywords     = extract_keywords(jd_text)

    if not jd_keywords:
        return {"match_score": 0, "missing_keywords": []}

    matched  = resume_keywords.intersection(jd_keywords)
    missing  = jd_keywords - resume_keywords

    match_score = round(len(matched) / len(jd_keywords) * 100, 1)

    # Return only the most meaningful missing keywords
    # Filter out very common words and sort by length (longer = more specific)
    meaningful_missing = sorted(
        [w for w in missing if len(w) > 4],
        key=len,
        reverse=True
    )[:20]

    return {
        "match_score":      match_score,
        "missing_keywords": meaningful_missing,
    }