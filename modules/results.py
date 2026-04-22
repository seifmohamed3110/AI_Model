"""
modules/results.py

Responsibilities:
- Build the final resume analysis result in one place
- Combine extraction, scoring, ATS, writing, keywords, strengths, and improvements
- Keep app.py thin and focused on API routing only

No Flask. No training. No file I/O.
"""

from typing import Optional, Dict, Any, List

from modules.features import extract_features, extract_keywords
from modules.career import detect_field
from modules.scorer import score_resume
from modules.ats import check_ats
from modules.writing import analyze_writing
from modules.suggestions import get_suggestions_detailed, get_jd_match


def _dedupe_keep_order(items: List[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        clean = " ".join(str(item).split()).strip()
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _format_ats_issues(issues: List[str]) -> List[str]:
    """
    Make ATS issues more user-friendly and less noisy for frontend display.
    """
    if not issues:
        return []

    polished = []
    for issue in issues:
        text = " ".join(str(issue).split()).strip()

        lowered = text.lower()

        if "special symbols" in lowered or "special characters" in lowered:
            polished.append(
                "Use simple standard characters where possible, especially for bullets and dashes, to keep ATS reading more reliable."
            )
        elif "email address" in lowered and "top" in lowered:
            polished.append(
                "Move your email address near the top of the resume so recruiters and ATS systems can find it quickly."
            )
        elif "phone number" in lowered and "top" in lowered:
            polished.append(
                "Move your phone number near the top of the resume so recruiters and ATS systems can find it quickly."
            )
        elif "experience" in lowered and "section heading" in lowered:
            polished.append(
                "Use a clear 'Experience' heading so your work history is easier for ATS systems to recognize."
            )
        elif "education" in lowered and "section heading" in lowered:
            polished.append(
                "Use a clear 'Education' heading so your academic background is easier for ATS systems to recognize."
            )
        elif "skills" in lowered and "section heading" in lowered:
            polished.append(
                "Use a clear 'Skills' heading so important keywords are easier for ATS systems to detect."
            )
        elif "date format" in lowered:
            polished.append(
                "Use one date format consistently throughout the resume to keep the layout clearer and easier to parse."
            )
        elif "single-column layout" in lowered or "two-column layout" in lowered or "layout may be too complex" in lowered:
            polished.append(
                "A simple single-column layout is usually easier for ATS systems to read accurately."
            )
        elif "table" in lowered:
            polished.append(
                "Avoid table-style formatting where possible and use simple text lists instead."
            )
        elif "header or footer" in lowered:
            polished.append(
                "Keep important information in the main body of the resume rather than in headers or footers."
            )
        elif "very little text" in lowered:
            polished.append(
                "Make sure important content is written as selectable text, not mainly as images, icons, or graphics."
            )
        else:
            polished.append(text)

    polished = _dedupe_keep_order(polished)

    # Keep ATS section focused and not overwhelming
    return polished[:4]


def _format_writing_strengths(items: List[str]) -> List[str]:
    return _dedupe_keep_order(items)[:4]


def _format_writing_issues(items: List[str]) -> List[str]:
    cleaned = _dedupe_keep_order(items)
    return cleaned[:5]


def _format_improvements(bucket: List[str], limit: int = 8) -> List[str]:
    return _dedupe_keep_order(bucket)[:limit]


def build_resume_result(
    text: str,
    user_field: Optional[str] = None,
    jd_text: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build the full final result object for one resume.

    Args:
        text: extracted clean resume text
        user_field: optional user-selected field override
        jd_text: optional job description text

    Returns:
        dict containing the final analysis result
    """
    # ── Core analysis ────────────────────────────────────────────────────────
    features = extract_features(text)
    detected_field = detect_field(text, user_override=user_field)
    score_result = score_resume(features)
    ats_issues_raw = check_ats(text)
    writing_result = analyze_writing(text)
    keywords = extract_keywords(text, field=detected_field, limit=12)
    suggestion_buckets = get_suggestions_detailed(features, detected_field)

    # ── Clean output for frontend ────────────────────────────────────────────
    strengths = _dedupe_keep_order(suggestion_buckets.get("strengths", []))[:4]
    improvements = _format_improvements(suggestion_buckets.get("all_suggestions", []), limit=8)
    critical_issues = _format_improvements(suggestion_buckets.get("critical_issues", []), limit=3)
    quick_wins = _format_improvements(suggestion_buckets.get("quick_wins", []), limit=3)
    field_specific_improvements = _format_improvements(
        suggestion_buckets.get("field_specific_improvements", []), limit=4
    )

    writing_strengths = _format_writing_strengths(writing_result.get("strengths", []))
    writing_issues = _format_writing_issues(writing_result.get("issues", []))
    ats_issues = _format_ats_issues(ats_issues_raw)

    # ── Build final result ───────────────────────────────────────────────────
    result = {
        # Core
        "score": score_result["score"],
        "grade": score_result["grade_label"].lower(),
        "detected_field": detected_field,
        "summary": score_result["summary"],
        "word_count": features["word_count"],

        # Final scope
        "keywords": keywords,
        "strengths": strengths,
        "improvements": improvements,

        # Structured improvement buckets
        "critical_issues": critical_issues,
        "quick_wins": quick_wins,
        "field_specific_improvements": field_specific_improvements,

        # Writing
        "writing_score": writing_result["score"],
        "writing_band": writing_result["band"],
        "writing_strengths": writing_strengths,
        "writing_issues": writing_issues,

        # ATS
        "ats_issues": ats_issues,

        # Extra metadata
        "missing_sections": score_result.get("missing_sections", []),
        "confidence": score_result.get("confidence"),
    }

    # ── Optional JD match ────────────────────────────────────────────────────
    if jd_text and jd_text.strip():
        jd_result = get_jd_match(text, jd_text)
        result["jd_match_score"] = jd_result["match_score"]
        result["missing_keywords"] = jd_result["missing_keywords"]

    return result