"""
modules/suggestions.py

Responsibilities:
- Accept feature dictionary and detected career field
- Return ranked, field-specific improvement suggestions
- Never penalize creative/media resumes for missing GitHub
- Never penalize non-tech resumes for missing technical skills
- Limit generic suggestions to top 3-4, boost field-specific ones

No Flask. No ML. No file I/O.
"""

from typing import List, Tuple
from modules.features import detect_content_gaps


# ── Suggestion weights (importance ranking) ───────────────────────────────────
WEIGHTS = {
    # Critical (weight 10) - Missing core information
    "missing_email": 10,
    "missing_phone": 10,
    "missing_summary": 10,
    "missing_experience": 10,
    "missing_education": 10,
    "missing_skills": 10,

    # High (weight 7) - Important but not blocking
    "missing_linkedin": 7,
    "missing_github_tech": 7,
    "missing_portfolio_creative": 7,
    "missing_portfolio_marketing": 7,
    "no_quantification": 7,
    "too_short": 7,

    # Medium (weight 4) - Quality improvements
    "few_bullets": 4,
    "weak_verbs": 4,
    "filler_phrases": 4,
    "first_person": 4,
    "long_bullets": 4,
    "few_tech_skills": 4,
    "few_marketing_skills": 4,
    "few_creative_skills": 4,
    "few_business_skills": 4,
    "missing_certifications": 4,
    "low_quantification": 4,

    # Low (weight 2) - Minor polish
    "needs_more_quantification": 2,
}

# ── Field-specific suggestion templates ──────────────────────────────────────
FIELD_SUGGESTIONS = {
    "tech": [
        ("missing_github_tech", "Add your GitHub profile URL — essential for tech roles, recruiters will check your code"),
        ("few_tech_skills", "Expand your Skills section with more specific technologies — languages, frameworks, cloud platforms, and tools"),
        ("missing_certifications_tech", "Consider adding relevant certifications — AWS, GCP, Azure, or language-specific certs strengthen a tech resume"),
        ("low_quantification_tech", "Add technical impact metrics — e.g. 'Reduced API response time by 40%' or 'Scaled system to 1M users'"),
    ],
    "marketing": [
        ("few_marketing_skills", "Add more marketing tools and platforms to your Skills section — Google Analytics, HubSpot, SEO, SEM, etc."),
        ("missing_certifications_marketing", "Add marketing certifications — Google Analytics, Google Ads, HubSpot, or Meta Blueprint are highly valued"),
        ("low_quantification_marketing", "Marketing resumes need strong metrics — add conversion rates, ROI figures, campaign reach, and revenue impact"),
        ("missing_portfolio_marketing", "Add a link to your portfolio or campaign samples — marketing roles expect to see your work"),
    ],
    "creative": [
        ("missing_portfolio_creative", "Add a portfolio link — essential for creative roles. Use Behance, Dribbble, or your own site"),
        ("few_creative_skills", "List the specific tools you use — Figma, Photoshop, Illustrator, After Effects, Premiere Pro, etc."),
        ("low_quantification_creative", "Add impact metrics even for creative work — e.g. 'Redesign increased user engagement by 25%' or 'Delivered 50+ brand assets'"),
    ],
    "business": [
        ("few_business_skills", "Add business-specific keywords — project management, stakeholder management, P&L, forecasting, etc."),
        ("low_quantification_business", "Business resumes need financial and operational metrics — revenue figures, cost savings, team sizes, budget managed"),
        ("missing_certifications_business", "Consider adding business certifications — PMP, Six Sigma, CPA, CFA, or MBA strengthen a business resume"),
    ],
    "other": [
        ("low_quantification_other", "Add measurable achievements specific to your field — patient counts, student outcomes, cases handled, etc."),
        ("missing_certifications_other", "Add any relevant licenses or certifications for your field"),
    ],
}

# ── Universal suggestion templates ───────────────────────────────────────────
UNIVERSAL_SUGGESTIONS = [
    ("missing_email", "Add your email address — it is required contact information"),
    ("missing_phone", "Add your phone number — recruiters need a way to contact you"),
    ("missing_linkedin", "Add your LinkedIn profile URL — most recruiters check LinkedIn before calling"),
    ("missing_summary", "Add a Summary or Profile section at the top — give recruiters a 3-sentence overview of who you are"),
    ("missing_experience", "Add an Experience section with your work history"),
    ("missing_education", "Add an Education section listing your degrees and institutions"),
    ("missing_skills", "Add a Skills section — ATS systems scan for keywords here"),
    ("too_short", "Your resume is too short — aim for at least 400 words to give enough context"),
    ("few_bullets", "Use bullet points to list your achievements — they are easier to scan than paragraphs"),
    ("no_quantification", "Add quantified achievements — numbers make your impact concrete, e.g. 'Increased sales by 30%'"),
    ("low_quantification", "Add more quantified achievements — aim for at least 3 metrics across your experience"),
    ("weak_verbs", "Start bullet points with strong action verbs like Led, Built, Delivered, Optimized"),
    ("filler_phrases", "Remove filler phrases like 'responsible for' and 'worked on' — replace with direct action verbs"),
    ("first_person", "Remove first-person pronouns — write 'Led a team of 5' not 'I led a team of 5'"),
    ("long_bullets", "Shorten your bullet points — aim for 15-20 words each, focus on the outcome not the process"),
]


def _get_suggestions_with_scores(features: dict, field: str) -> List[Tuple[str, str, int]]:
    """
    Generate suggestions with importance scores.

    Returns: List of (suggestion_id, suggestion_text, weight) tuples
    """
    suggestions = []
    field = field.lower().strip()

    # ── Universal suggestions ────────────────────────────────────────────────
    if not features.get("has_email"):
        suggestions.append(("missing_email", UNIVERSAL_SUGGESTIONS[0][1], WEIGHTS["missing_email"]))

    if not features.get("has_phone"):
        suggestions.append(("missing_phone", UNIVERSAL_SUGGESTIONS[1][1], WEIGHTS["missing_phone"]))

    if not features.get("has_linkedin"):
        suggestions.append(("missing_linkedin", UNIVERSAL_SUGGESTIONS[2][1], WEIGHTS["missing_linkedin"]))

    if not features.get("has_summary"):
        suggestions.append(("missing_summary", UNIVERSAL_SUGGESTIONS[3][1], WEIGHTS["missing_summary"]))

    if not features.get("has_experience"):
        suggestions.append(("missing_experience", UNIVERSAL_SUGGESTIONS[4][1], WEIGHTS["missing_experience"]))

    if not features.get("has_education"):
        suggestions.append(("missing_education", UNIVERSAL_SUGGESTIONS[5][1], WEIGHTS["missing_education"]))

    if not features.get("has_skills"):
        suggestions.append(("missing_skills", UNIVERSAL_SUGGESTIONS[6][1], WEIGHTS["missing_skills"]))

    if features.get("word_count", 0) < 150:
        suggestions.append(("too_short", UNIVERSAL_SUGGESTIONS[7][1], WEIGHTS["too_short"]))

    if features.get("bullet_count", 0) < 3:
        suggestions.append(("few_bullets", UNIVERSAL_SUGGESTIONS[8][1], WEIGHTS["few_bullets"]))

    if features.get("quantification_count", 0) == 0:
        suggestions.append(("no_quantification", UNIVERSAL_SUGGESTIONS[9][1], WEIGHTS["no_quantification"]))
    elif features.get("quantification_count", 0) < 3:
        suggestions.append(("low_quantification", UNIVERSAL_SUGGESTIONS[10][1], WEIGHTS["low_quantification"]))

    if features.get("action_verb_count", 0) < 3:
        suggestions.append(("weak_verbs", UNIVERSAL_SUGGESTIONS[11][1], WEIGHTS["weak_verbs"]))

    if features.get("filler_count", 0) > 0:
        suggestions.append(("filler_phrases", UNIVERSAL_SUGGESTIONS[12][1], WEIGHTS["filler_phrases"]))

    if features.get("first_person_count", 0) > 2:
        suggestions.append(("first_person", UNIVERSAL_SUGGESTIONS[13][1], WEIGHTS["first_person"]))

    if features.get("avg_bullet_length", 0) > 30:
        suggestions.append(("long_bullets", UNIVERSAL_SUGGESTIONS[14][1], WEIGHTS["long_bullets"]))

    # ── Field-specific suggestions ───────────────────────────────────────────
    if field == "tech":
        if not features.get("has_github"):
            suggestions.append(("missing_github_tech", FIELD_SUGGESTIONS["tech"][0][1], WEIGHTS["missing_github_tech"]))
        if features.get("tech_skill_count", 0) < 5:
            suggestions.append(("few_tech_skills", FIELD_SUGGESTIONS["tech"][1][1], WEIGHTS["few_tech_skills"]))
        if not features.get("has_certifications"):
            suggestions.append(("missing_certifications_tech", FIELD_SUGGESTIONS["tech"][2][1], WEIGHTS["missing_certifications"]))
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_tech", FIELD_SUGGESTIONS["tech"][3][1], WEIGHTS["low_quantification"]))

    elif field == "marketing":
        if features.get("marketing_skill_count", 0) < 4:
            suggestions.append(("few_marketing_skills", FIELD_SUGGESTIONS["marketing"][0][1], WEIGHTS["few_marketing_skills"]))
        if not features.get("has_certifications"):
            suggestions.append(("missing_certifications_marketing", FIELD_SUGGESTIONS["marketing"][1][1], WEIGHTS["missing_certifications"]))
        if features.get("quantification_count", 0) < 3:
            suggestions.append(("low_quantification_marketing", FIELD_SUGGESTIONS["marketing"][2][1], WEIGHTS["low_quantification"]))
        if not features.get("has_portfolio"):
            suggestions.append(("missing_portfolio_marketing", FIELD_SUGGESTIONS["marketing"][3][1], WEIGHTS["missing_portfolio_marketing"]))

    elif field == "creative":
        if not features.get("has_portfolio"):
            suggestions.append(("missing_portfolio_creative", FIELD_SUGGESTIONS["creative"][0][1], WEIGHTS["missing_portfolio_creative"]))
        if features.get("creative_skill_count", 0) < 4:
            suggestions.append(("few_creative_skills", FIELD_SUGGESTIONS["creative"][1][1], WEIGHTS["few_creative_skills"]))
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_creative", FIELD_SUGGESTIONS["creative"][2][1], WEIGHTS["low_quantification"]))

    elif field == "business":
        if features.get("business_skill_count", 0) < 4:
            suggestions.append(("few_business_skills", FIELD_SUGGESTIONS["business"][0][1], WEIGHTS["few_business_skills"]))
        if features.get("quantification_count", 0) < 3:
            suggestions.append(("low_quantification_business", FIELD_SUGGESTIONS["business"][1][1], WEIGHTS["low_quantification"]))
        if not features.get("has_certifications"):
            suggestions.append(("missing_certifications_business", FIELD_SUGGESTIONS["business"][2][1], WEIGHTS["missing_certifications"]))

    elif field == "other":
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_other", FIELD_SUGGESTIONS["other"][0][1], WEIGHTS["low_quantification"]))
        if not features.get("has_certifications"):
            suggestions.append(("missing_certifications_other", FIELD_SUGGESTIONS["other"][1][1], WEIGHTS["missing_certifications"]))

    return suggestions


def get_suggestions(features: dict, field: str, limit: int = 7) -> List[str]:
    """
    Generate ranked, field-specific improvement suggestions.

    Args:
        features: dictionary from extract_features()
        field: one of tech, marketing, creative, business, other
        limit: maximum number of suggestions to return (default 7)

    Returns:
        list of suggestion strings ordered by importance, limited to top N
    """
    # Get all suggestions with scores
    scored_suggestions = _get_suggestions_with_scores(features, field)

    if not scored_suggestions:
        return []

    gaps = detect_content_gaps(features, field)
    field_keywords = ["github", "portfolio", "tech", "marketing", "creative", "business", "certifications", "_other"]
    generic_limit = 4

    boosted = []
    for sug_id, text, weight in scored_suggestions:
        is_field_specific = any(kw in sug_id for kw in field_keywords)
        if is_field_specific:
            weight += 3

        # Gap-aware personalization boost
        if gaps["no_metrics"] and "quantification" in sug_id:
            weight += 2
        if gaps["no_projects"] and "projects" in text.lower():
            weight += 2
        if gaps["no_achievements"] and "achievement" in text.lower():
            weight += 2
        if gaps["no_github"] and "github" in sug_id:
            weight += 2
        if gaps["no_portfolio"] and "portfolio" in sug_id:
            weight += 2

        boosted.append((sug_id, text, weight, is_field_specific))

    boosted.sort(key=lambda x: x[2], reverse=True)

    # Remove duplicates by normalized text first, then by id.
    # This deduplicates overlap across quick_wins/improvements/ATS/writing buckets.
    seen_ids = set()
    seen_texts = set()
    seen = set()
    unique_suggestions = []
    for sug_id, text, weight, is_field_specific in boosted:
        norm_text = " ".join(text.lower().split())
        if sug_id in seen_ids or norm_text in seen_texts:
            continue
        seen_ids.add(sug_id)
        seen_texts.add(norm_text)
        seen.add(sug_id)
        unique_suggestions.append((sug_id, text, weight, is_field_specific))

    # Limit generic suggestions to top 3-4 while preserving critical gaps
    field_first = [s for s in unique_suggestions if s[3]]
    generic = [s for s in unique_suggestions if not s[3]]
    top_suggestions = (field_first + generic[:generic_limit])[:limit]

    # Return just the text
    return [text for _, text, _, _ in top_suggestions]


def get_suggestions_detailed(features: dict, field: str) -> dict:
    """
    Generate ranked suggestions with metadata.

    Args:
        features: dictionary from extract_features()
        field: one of tech, marketing, creative, business, other

    Returns:
        dict with critical_issues, quick_wins, field_specific_improvements, all_suggestions
    """
    scored_suggestions = _get_suggestions_with_scores(features, field)

    if not scored_suggestions:
        return {
            "critical_issues": [],
            "quick_wins": [],
            "field_specific_improvements": [],
            "all_suggestions": []
        }

    field_keywords = ["github", "portfolio", "tech", "marketing", "creative", "business", "certifications", "_other"]
    gaps = detect_content_gaps(features, field)
    generic_limit = 4
    boosted = []
    for sug_id, text, weight in scored_suggestions:
        is_field_specific = any(kw in sug_id for kw in field_keywords)
        if is_field_specific:
            weight += 3
        if gaps["no_metrics"] and "quantification" in sug_id:
            weight += 2
        if gaps["no_projects"] and "projects" in text.lower():
            weight += 2
        if gaps["no_achievements"] and "achievement" in text.lower():
            weight += 2
        if gaps["no_github"] and "github" in sug_id:
            weight += 2
        if gaps["no_portfolio"] and "portfolio" in sug_id:
            weight += 2
        boosted.append({
            "id": sug_id,
            "text": text,
            "weight": weight,
            "is_field_specific": is_field_specific
        })

    # Re-sort and deduplicate
    boosted.sort(key=lambda x: x["weight"], reverse=True)
    seen = set()
    seen_texts = set()
    unique = []
    for s in boosted:
        norm_text = " ".join(s["text"].lower().split())
        if s["id"] in seen or norm_text in seen_texts:
            continue
        seen.add(s["id"])
        seen_texts.add(norm_text)
        unique.append(s)

    field_specific = [s for s in unique if s["is_field_specific"]][:4]
    generic = [s for s in unique if not s["is_field_specific"]][:generic_limit]
    merged_ranked = sorted(field_specific + generic, key=lambda x: x["weight"], reverse=True)

    critical = [s for s in unique if s["weight"] >= 10][:3]
    quick_wins = [s for s in unique if 5 <= s["weight"] < 10][:3]

    return {
        "critical_issues": [s["text"] for s in critical],
        "quick_wins": [s["text"] for s in quick_wins],
        "field_specific_improvements": [s["text"] for s in field_specific],
        "all_suggestions": [s["text"] for s in merged_ranked[:10]]
    }


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
