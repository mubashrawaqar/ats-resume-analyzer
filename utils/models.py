"""Shared data structures used across parsers and matching modules.

Keeping these in one place avoids circular imports between the
parsers/ and matching/ packages, and gives app.py a single,
predictable shape to build the UI around.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class ParsedDocument:
    """Result of extracting text from a single uploaded file.

    text: the best-effort extracted plain text (empty string if extraction
        failed completely).
    success: False if we could not get any usable text at all.
    used_ocr: True if we had to fall back to Tesseract OCR (image-based PDF).
    warnings: user-facing, non-fatal issues encountered during parsing
        (e.g. "page 2 appears to be a scanned image").
    ats_issues: formatting problems that would likely confuse a real ATS
        (tables used for layout, missing section headers, etc.) — this is
        the "ATS parseability check" bonus feature, populated later by the
        scorer/UI layer for resumes.
    """

    text: str
    success: bool = True
    used_ocr: bool = False
    warnings: List[str] = field(default_factory=list)
    ats_issues: List[str] = field(default_factory=list)


@dataclass
class ContactInfo:
    """Best-effort extracted contact details for a candidate."""

    name: str = "Not found"
    email: str = "Not found"
    phone: str = "Not found"


@dataclass
class ScoreBreakdown:
    """Explainable components that make up the overall match_score.

    Each field is a 0-100 percentage. The overall match_score is a
    weighted combination of these (see matching/scorer.py for the
    exact weights and rationale).
    """

    skills_match_pct: float
    keyword_density_pct: float
    experience_relevance_pct: float


@dataclass
class MatchResult:
    """Full result of matching one resume against the job description."""

    filename: str
    contact: ContactInfo
    matching_skills: List[str]
    missing_skills: List[str]
    match_score: float
    breakdown: ScoreBreakdown
    ats_issues: List[str]
    parse_warnings: List[str]
