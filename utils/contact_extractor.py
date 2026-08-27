"""Extract candidate name, email, and phone number from resume text.

Email and phone use regex — they have a rigid enough format that regex is
more reliable than NER for them. Name extraction is trickier: spaCy's
PERSON NER is a good default but can grab the wrong span (a section
heading, a job title, a former employer's name). So name extraction uses
two signals and cross-checks them:

1. spaCy PERSON NER over the top of the document.
2. A heuristic that looks at the first few non-empty lines for a short,
   name-shaped line (2-4 words, no digits, no section-header/job-title
   vocabulary) — resumes conventionally put the candidate's name as the
   very first line.

If the NER result fails a sanity check (it contains job-title or
section-header words), it's treated as a failed extraction and we fall
back to the heuristic instead.
"""

from __future__ import annotations

import re

from utils.models import ContactInfo
from utils.nlp_models import load_spacy_model

EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

# Covers common US/international formats: (123) 456-7890, 123-456-7890,
# +1 123 456 7890, 123.456.7890, etc. Deliberately permissive since resume
# phone formatting is inconsistent.
PHONE_PATTERN = re.compile(
    r"(\+?\d{1,3}[\s.\-]?)?(\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}"
)

# Only look at the first N characters for the name — resumes put the
# candidate's name at the top, and searching the whole document risks
# picking up a former employer's name or a reference instead.
NAME_SEARCH_WINDOW = 400

# How many non-empty lines from the top to consider for the heuristic.
HEURISTIC_LINE_WINDOW = 5

# Words that indicate a candidate string is a section header or job title,
# not a person's name. If any of these show up as a whole word inside a
# candidate name, we reject it — whether it came from NER or the heuristic.
_REJECT_WORDS = {
    "summary", "experience", "objective", "education", "skills", "profile",
    "resume", "curriculum", "vitae", "contact", "references", "projects",
    "certifications", "portfolio", "about",
    "engineer", "developer", "manager", "analyst", "specialist",
    "consultant", "architect", "designer", "director", "intern",
    "internship", "lead", "senior", "junior", "founder", "president",
    "officer", "administrator", "coordinator", "supervisor", "executive",
    "ai", "ml", "data", "software", "fullstack", "full-stack", "backend",
    "frontend", "devops", "product", "marketing", "sales", "hr",
    "scientist", "researcher", "consultancy",
}


def extract_email(text: str) -> str:
    match = EMAIL_PATTERN.search(text)
    return match.group(0) if match else "Not found"


def extract_phone(text: str) -> str:
    match = PHONE_PATTERN.search(text)
    if not match:
        return "Not found"
    # Strip stray whitespace but keep the original formatting otherwise.
    return re.sub(r"\s+", " ", match.group(0)).strip()


def _is_valid_name(candidate: str) -> bool:
    """Sanity check: does this look like a plausible human name?

    Rejects candidates that are really section headers or job titles
    (e.g. "AI Engineer", "Professional Summary") rather than a name.
    """
    candidate = candidate.strip()
    if not candidate:
        return False

    words = candidate.split()
    if not (2 <= len(words) <= 4):
        return False

    if any(ch.isdigit() for ch in candidate):
        return False

    if "@" in candidate or "http" in candidate.lower():
        return False

    if len(candidate) > 40:
        return False

    lower_words = {w.strip(".,").lower() for w in words}
    if lower_words & _REJECT_WORDS:
        return False

    # Each word should look name-shaped: letters, hyphens, or apostrophes only.
    for w in words:
        if not re.match(r"^[A-Za-z][A-Za-z'\-.]*$", w):
            return False

    return True


def _heuristic_name_from_top_lines(text: str) -> str | None:
    """Scan the first few non-empty lines for a short, name-shaped line."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in lines[:HEURISTIC_LINE_WINDOW]:
        if _is_valid_name(line):
            return line
    return None


def _ner_name(text: str) -> str | None:
    """Best-effort candidate name via spaCy PERSON entities near the top."""
    window = text[:NAME_SEARCH_WINDOW]
    try:
        nlp = load_spacy_model()
    except RuntimeError:
        return None

    doc = nlp(window)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            name = ent.text.strip()
            if _is_valid_name(name):
                return name
    return None


def extract_name(text: str) -> str:
    """Extract the candidate's name, cross-checking NER against a heuristic.

    Trusts spaCy NER when its result passes the sanity check. If NER comes
    back empty or with something that looks like a job title/section
    header, falls back to scanning the first few lines for a name-shaped
    line instead.
    """
    ner_candidate = _ner_name(text)
    if ner_candidate:
        return ner_candidate

    heuristic_candidate = _heuristic_name_from_top_lines(text)
    if heuristic_candidate:
        return heuristic_candidate

    return "Not found"


def extract_contact_info(text: str) -> ContactInfo:
    """Convenience wrapper returning all three fields at once."""
    return ContactInfo(
        name=extract_name(text),
        email=extract_email(text),
        phone=extract_phone(text),
    )
