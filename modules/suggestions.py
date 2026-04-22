"""
modules/suggestions.py

Responsibilities:
- Accept feature dictionary and detected career field
- Return ranked, field-specific improvement suggestions
- Generate personalized strengths
- Never penalize creative/media resumes for missing GitHub
- Never penalize non-tech resumes for missing technical skills
- Limit generic suggestions to top 3-4, boost field-specific ones

No Flask. No ML. No file I/O.
"""

from typing import List, Tuple
from modules.features import detect_content_gaps


# ── Suggestion weights (importance ranking) ───────────────────────────────────
WEIGHTS = {
    "missing_email": 10,
    "missing_phone": 10,
    "missing_summary": 10,
    "missing_experience": 10,
    "missing_education": 10,
    "missing_skills": 10,

    "missing_linkedin": 7,
    "missing_github_tech": 7,
    "missing_portfolio_creative": 7,
    "missing_portfolio_marketing": 7,
    "no_quantification": 7,
    "too_short": 7,

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

    "needs_more_quantification": 2,
}


# ── Universal suggestion templates ───────────────────────────────────────────
UNIVERSAL_SUGGESTIONS = [
    ("missing_email", "Add your email address — it is required contact information."),
    ("missing_phone", "Add your phone number — recruiters need a direct way to contact you."),
    ("missing_linkedin", "Add your LinkedIn profile URL — many recruiters check LinkedIn before reaching out."),
    ("missing_summary", "Add a short Summary or Profile section at the top to explain your background and value."),
    ("missing_experience", "Add an Experience section with your work history."),
    ("missing_education", "Add an Education section listing your degrees and institutions."),
    ("missing_skills", "Add a Skills section — ATS systems often scan it for important keywords."),
    ("too_short", "Your resume is too short — add more relevant detail so your experience and impact is clear."),
    ("few_bullets", "Use more bullet points to present achievements clearly and improve readability."),
    ("no_quantification", "Add measurable achievements — numbers make your impact easier to understand."),
    ("low_quantification", "Add more measurable results across your experience so your impact is clearer."),
    ("weak_verbs", "Start bullet points with stronger action verbs such as Led, Built, Improved, Delivered, or Optimized."),
    ("filler_phrases", "Replace filler phrases like 'responsible for' with direct action and outcome-focused language."),
    ("first_person", "Remove first-person pronouns — write 'Led a team of 5' instead of 'I led a team of 5'."),
    ("long_bullets", "Shorten long bullet points so each one focuses on one action and one result."),
]


def _field_skills_text(field: str) -> str:
    field = (field or "other").strip().lower()

    if field == "tech":
        return "Expand your Skills section with more specific technologies — programming languages, frameworks, databases, cloud tools, and platforms."
    if field == "marketing":
        return "Add more marketing tools and channel-specific skills — analytics, campaign platforms, CRM tools, content tools, and reporting tools."
    if field == "creative":
        return "List the specific tools and creative skills you use — design software, editing tools, prototyping tools, and creative workflows."
    if field == "business":
        return "Add more business-specific keywords — planning, reporting, stakeholder coordination, budgeting, operations, and process improvement."
    return "Add more role-specific skills and keywords so the resume reflects your target field more clearly."


def _field_quantification_text(field: str) -> str:
    field = (field or "other").strip().lower()

    if field == "tech":
        return "Add technical impact metrics — for example, speed improvements, scale handled, uptime, delivery time, or cost savings."
    if field == "marketing":
        return "Add marketing performance metrics — for example, conversion rate, engagement, growth, campaign reach, lead volume, or return on spend."
    if field == "creative":
        return "Add creative impact where possible — for example, engagement gains, faster delivery, improved user response, or number of assets delivered."
    if field == "business":
        return "Add business impact metrics — for example, revenue impact, cost reduction, budget handled, team size, or process efficiency gains."
    return "Add measurable achievements specific to your field so your impact is easier to understand."


def _field_portfolio_text(field: str) -> str:
    field = (field or "other").strip().lower()

    if field == "creative":
        return "Add a portfolio link — essential for creative roles. Use a portfolio site or a professional platform that shows your work clearly."
    if field == "marketing":
        return "Add a link to campaign samples, case studies, or a portfolio — marketing roles often benefit from visible work examples."
    return "Add a portfolio or work-sample link if your target role benefits from visible examples of your work."


def _field_certification_text(field: str) -> str:
    field = (field or "other").strip().lower()

    if field == "tech":
        return "If relevant to your target role, add one or two recognized technical certifications to strengthen credibility."
    if field == "marketing":
        return "If relevant to your target role, add one or two recognized marketing certifications to strengthen credibility."
    if field == "business":
        return "If relevant to your target role, add one or two recognized professional certifications to strengthen credibility."
    return "If your field values licenses or certifications, add the most relevant ones to strengthen credibility."


def _should_suggest_certifications(features: dict, field: str) -> bool:
    """
    Decide whether a certification suggestion is relevant enough to show.
    """
    if features.get("has_certifications"):
        return False

    field = (field or "other").strip().lower()

    if field == "tech":
        return (
            features.get("tech_skill_count", 0) >= 4
            or features.get("has_projects", 0)
            or features.get("has_github", 0)
        )

    if field == "marketing":
        return features.get("marketing_skill_count", 0) >= 3

    if field == "business":
        return features.get("business_skill_count", 0) >= 3

    if field == "other":
        return features.get("word_count", 0) >= 250

    return False


def get_strengths(features: dict, field: str) -> List[str]:
    """
    Generate personalized strengths based on resume features and detected field.
    """
    field = (field or "other").strip().lower()
    strengths = []

    if features.get("has_email") and features.get("has_phone"):
        strengths.append("Contact information is complete and easy to find.")

    if features.get("has_linkedin"):
        strengths.append("LinkedIn profile is included.")

    if features.get("section_count", 0) >= 4:
        strengths.append("Resume has a clear and organized section structure.")

    if features.get("bullet_count", 0) >= 5:
        strengths.append("Bullet points are used effectively for readability.")

    if features.get("quantification_count", 0) >= 2:
        strengths.append("Includes quantified achievements that make results more concrete.")

    if features.get("action_verb_count", 0) >= 5:
        strengths.append("Uses strong action-oriented language.")

    if features.get("has_projects"):
        strengths.append("Projects section adds practical evidence of skills.")

    if features.get("has_achievements"):
        strengths.append("Achievements section helps highlight accomplishments.")

    if features.get("word_count", 0) >= 300:
        strengths.append("Resume provides a solid amount of detail.")

    if field == "tech":
        if features.get("has_github"):
            strengths.append("GitHub profile is included, which is valuable for tech roles.")
        if features.get("tech_skill_count", 0) >= 5:
            strengths.append("Shows strong technical keyword coverage for tech roles.")

    elif field == "marketing":
        if features.get("has_portfolio"):
            strengths.append("Portfolio or campaign samples are included, which strengthens marketing credibility.")
        if features.get("marketing_skill_count", 0) >= 4:
            strengths.append("Shows relevant marketing tool and strategy coverage.")

    elif field == "creative":
        if features.get("has_portfolio"):
            strengths.append("Portfolio link is included, which is especially important for creative roles.")
        if features.get("creative_skill_count", 0) >= 4:
            strengths.append("Highlights relevant creative tools and design-related skills.")

    elif field == "business":
        if features.get("business_skill_count", 0) >= 4:
            strengths.append("Demonstrates strong business-oriented keyword coverage.")

    unique = []
    seen = set()
    for s in strengths:
        if s not in seen:
            unique.append(s)
            seen.add(s)

    return unique[:4]


def _get_suggestions_with_scores(features: dict, field: str) -> List[Tuple[str, str, int]]:
    """
    Generate suggestions with importance scores.
    Returns: List of (suggestion_id, suggestion_text, weight) tuples
    """
    suggestions = []
    field = (field or "other").strip().lower()

    # Universal suggestions
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

    # Field-specific suggestions
    if field == "tech":
        if not features.get("has_github"):
            suggestions.append(("missing_github_tech", "Add your GitHub profile URL — essential for tech roles, recruiters often review code samples or project repositories.", WEIGHTS["missing_github_tech"]))
        if features.get("tech_skill_count", 0) < 5:
            suggestions.append(("few_tech_skills", _field_skills_text(field), WEIGHTS["few_tech_skills"]))
        if _should_suggest_certifications(features, field):
            suggestions.append(("missing_certifications_tech", _field_certification_text(field), WEIGHTS["missing_certifications"]))
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_tech", _field_quantification_text(field), WEIGHTS["low_quantification"]))

    elif field == "marketing":
        if features.get("marketing_skill_count", 0) < 4:
            suggestions.append(("few_marketing_skills", _field_skills_text(field), WEIGHTS["few_marketing_skills"]))
        if _should_suggest_certifications(features, field):
            suggestions.append(("missing_certifications_marketing", _field_certification_text(field), WEIGHTS["missing_certifications"]))
        if features.get("quantification_count", 0) < 3:
            suggestions.append(("low_quantification_marketing", _field_quantification_text(field), WEIGHTS["low_quantification"]))
        if not features.get("has_portfolio"):
            suggestions.append(("missing_portfolio_marketing", _field_portfolio_text(field), WEIGHTS["missing_portfolio_marketing"]))

    elif field == "creative":
        if not features.get("has_portfolio"):
            suggestions.append(("missing_portfolio_creative", _field_portfolio_text(field), WEIGHTS["missing_portfolio_creative"]))
        if features.get("creative_skill_count", 0) < 4:
            suggestions.append(("few_creative_skills", _field_skills_text(field), WEIGHTS["few_creative_skills"]))
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_creative", _field_quantification_text(field), WEIGHTS["low_quantification"]))

    elif field == "business":
        if features.get("business_skill_count", 0) < 4:
            suggestions.append(("few_business_skills", _field_skills_text(field), WEIGHTS["few_business_skills"]))
        if features.get("quantification_count", 0) < 3:
            suggestions.append(("low_quantification_business", _field_quantification_text(field), WEIGHTS["low_quantification"]))
        if _should_suggest_certifications(features, field):
            suggestions.append(("missing_certifications_business", _field_certification_text(field), WEIGHTS["missing_certifications"]))

    elif field == "other":
        if features.get("quantification_count", 0) < 2:
            suggestions.append(("low_quantification_other", _field_quantification_text(field), WEIGHTS["low_quantification"]))
        if _should_suggest_certifications(features, field):
            suggestions.append(("missing_certifications_other", _field_certification_text(field), WEIGHTS["missing_certifications"]))

    return suggestions


def get_suggestions(features: dict, field: str, limit: int = 7) -> List[str]:
    """
    Generate ranked, field-specific improvement suggestions.
    """
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

    seen_ids = set()
    seen_texts = set()
    unique_suggestions = []
    for sug_id, text, weight, is_field_specific in boosted:
        norm_text = " ".join(text.lower().split())
        if sug_id in seen_ids or norm_text in seen_texts:
            continue
        seen_ids.add(sug_id)
        seen_texts.add(norm_text)
        unique_suggestions.append((sug_id, text, weight, is_field_specific))

    field_first = [s for s in unique_suggestions if s[3]]
    generic = [s for s in unique_suggestions if not s[3]]
    top_suggestions = (field_first + generic[:generic_limit])[:limit]

    return [text for _, text, _, _ in top_suggestions]


def get_suggestions_detailed(features: dict, field: str) -> dict:
    """
    Generate ranked suggestions with metadata.
    """
    scored_suggestions = _get_suggestions_with_scores(features, field)

    if not scored_suggestions:
        return {
            "strengths": get_strengths(features, field),
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
        "strengths": get_strengths(features, field),
        "critical_issues": [s["text"] for s in critical],
        "quick_wins": [s["text"] for s in quick_wins],
        "field_specific_improvements": [s["text"] for s in field_specific],
        "all_suggestions": [s["text"] for s in merged_ranked[:10]]
    }


def get_jd_match(resume_text: str, jd_text: str) -> dict:
    """
    Compare resume text against a job description.
    Returns match score and list of missing keywords.
    """
    import re

    def extract_keywords(text: str) -> set:
        text_lower = text.lower()
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
    jd_keywords = extract_keywords(jd_text)

    if not jd_keywords:
        return {"match_score": 0, "missing_keywords": []}

    matched = resume_keywords.intersection(jd_keywords)
    missing = jd_keywords - resume_keywords

    match_score = round(len(matched) / len(jd_keywords) * 100, 1)

    meaningful_missing = sorted(
        [w for w in missing if len(w) > 4],
        key=len,
        reverse=True
    )[:20]

    return {
        "match_score": match_score,
        "missing_keywords": meaningful_missing,
    }