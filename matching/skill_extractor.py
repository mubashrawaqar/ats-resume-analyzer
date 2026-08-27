"""ATS Resume Analyzer — Skill Extraction.

Uses Groq LLM extraction as the primary path and a conservative
dictionary fallback when the LLM is unavailable.

The extractor is intentionally strict:
- Only genuine skills, technologies, tools, frameworks, platforms,
  methodologies, and professional competencies are accepted.
- Generic phrases such as "experience", "strong background", "the role",
  "years", "stakeholders", etc. are rejected.
- LLM output is normalized and deduplicated.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List, Optional, Set

from dotenv import load_dotenv

from utils.nlp_models import load_spacy_model

load_dotenv()


# ---------------------------------------------------------------------------
# CANONICAL SKILL DICTIONARY
# ---------------------------------------------------------------------------

SKILL_SYNONYMS: Dict[str, List[str]] = {
    "python": ["python", "python3"],
    "java": ["java"],
    "javascript": ["javascript", "js", "es6"],
    "typescript": ["typescript", "ts"],
    "c++": ["c++", "cpp"],
    "c#": ["c#", "csharp"],
    "sql": ["sql", "mysql", "postgresql", "postgres", "t-sql"],
    "r": ["r programming"],
    "go": ["golang"],
    "machine learning": ["machine learning", "ml"],
    "deep learning": ["deep learning", "dl"],
    "natural language processing": [
        "natural language processing",
        "nlp",
    ],
    "computer vision": ["computer vision", "cv"],
    "data analysis": ["data analysis", "data analytics"],
    "data science": ["data science"],
    "artificial intelligence": [
        "artificial intelligence",
        "ai",
    ],
    "tensorflow": ["tensorflow", "tf"],
    "pytorch": ["pytorch", "torch"],
    "scikit-learn": [
        "scikit-learn",
        "sklearn",
        "scikit learn",
    ],
    "pandas": ["pandas"],
    "numpy": ["numpy"],
    "react": ["react", "react.js", "reactjs"],
    "angular": ["angular", "angular.js"],
    "vue": ["vue", "vue.js"],
    "node.js": ["node.js", "node", "nodejs"],
    "django": ["django"],
    "flask": ["flask"],
    "fastapi": ["fastapi"],
    "streamlit": ["streamlit"],
    "docker": ["docker", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "aws": ["aws", "amazon web services"],
    "azure": ["azure", "microsoft azure"],
    "gcp": [
        "gcp",
        "google cloud",
        "google cloud platform",
    ],
    "git": [
        "git",
        "github",
        "gitlab",
        "version control",
    ],
    "ci/cd": [
        "ci/cd",
        "continuous integration",
        "continuous deployment",
    ],
    "rest api": [
        "rest api",
        "restful api",
        "rest apis",
    ],
    "graphql": ["graphql"],
    "agile": ["agile", "scrum", "kanban"],
    "project management": ["project management"],
    "communication": ["communication skills", "communication"],
    "leadership": ["leadership"],
    "problem solving": [
        "problem solving",
        "problem-solving",
    ],
    "teamwork": ["teamwork", "collaboration"],
    "excel": ["excel", "microsoft excel"],
    "tableau": ["tableau"],
    "power bi": ["power bi", "powerbi"],
    "spark": [
        "apache spark",
        "pyspark",
        "spark",
    ],
    "hadoop": ["hadoop"],
    "linux": ["linux", "unix"],
    "html": ["html", "html5"],
    "css": ["css", "css3"],
    "mongodb": ["mongodb", "mongo"],
    "redis": ["redis"],
    "microservices": ["microservices"],
    "api design": [
        "api design",
        "api development",
    ],
    "testing": [
        "unit testing",
        "test automation",
        "qa",
        "quality assurance",
    ],
}


_DISPLAY_NAMES: Dict[str, str] = {
    "python": "Python",
    "java": "Java",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "sql": "SQL",
    "c++": "C++",
    "c#": "C#",
    "r": "R",
    "go": "Go",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "natural language processing": "Natural Language Processing (NLP)",
    "computer vision": "Computer Vision",
    "data analysis": "Data Analysis",
    "data science": "Data Science",
    "artificial intelligence": "Artificial Intelligence (AI)",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit-learn": "scikit-learn",
    "pandas": "pandas",
    "numpy": "NumPy",
    "react": "React",
    "angular": "Angular",
    "vue": "Vue.js",
    "node.js": "Node.js",
    "django": "Django",
    "flask": "Flask",
    "fastapi": "FastAPI",
    "streamlit": "Streamlit",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "Google Cloud Platform (GCP)",
    "git": "Git",
    "ci/cd": "CI/CD",
    "rest api": "REST API",
    "graphql": "GraphQL",
    "agile": "Agile",
    "project management": "Project Management",
    "communication": "Communication",
    "leadership": "Leadership",
    "problem solving": "Problem Solving",
    "teamwork": "Teamwork",
    "excel": "Excel",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "spark": "Apache Spark",
    "hadoop": "Hadoop",
    "linux": "Linux",
    "html": "HTML",
    "css": "CSS",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "microservices": "Microservices",
    "api design": "API Design",
    "testing": "Testing / QA",
}


_SYNONYM_TO_CANONICAL: Dict[str, str] = {
    synonym.strip().lower(): canonical
    for canonical, synonyms in SKILL_SYNONYMS.items()
    for synonym in synonyms
}


_KNOWN_ACRONYMS = {
    "ai",
    "ml",
    "nlp",
    "cv",
    "sql",
    "aws",
    "gcp",
    "api",
    "css",
    "html",
    "qa",
    "pm",
    "dl",
    "ui",
    "ux",
    "iot",
    "etl",
    "sdk",
    "cli",
    "aci",
}


# ---------------------------------------------------------------------------
# GARBAGE FILTER
# ---------------------------------------------------------------------------

# These are NOT skills. This protects the application even if the LLM
# produces a poor answer.
_REJECT_EXACT = {
    "experience",
    "years",
    "year",
    "role",
    "the role",
    "responsibilities",
    "responsibility",
    "strong background",
    "background",
    "professional experience",
    "work experience",
    "industry experience",
    "job description",
    "candidate",
    "candidates",
    "professional",
    "professional background",
    "stakeholders",
    "cross-functional stakeholders",
    "team",
    "teams",
    "company",
    "organization",
    "organizations",
    "business",
    "businesses",
    "results",
    "findings",
    "present findings",
    "strong problem-solving skills",
    "ability to work independently",
    "ability to work",
    "work independently",
    "responsible for",
    "worked on",
    "working with",
    "developed",
    "designed",
    "built",
    "managed",
    "supported",
    "knowledge",
    "understanding",
    "etc",
}


_REJECT_PATTERNS = [
    r"^\d+\s*(years?|yrs?)$",
    r"^(the|a|an)\s+\w+$",
    r"^(strong|excellent|good|solid)\s+",
    r"\bexperience\b",
    r"\bresponsibilit(y|ies)\b",
    r"^ability to\b",
    r"^ability\b",
    r"^work(ing)?\b",
    r"^responsible\b",
    r"^experience with\b",
    r"^experience in\b",
    r"^knowledge of\b",
    r"^understanding of\b",
    r"^proven ability\b",
    r"^cross-functional\b",
    r"\bstakeholders?\b",
]


def _is_valid_skill(skill: str) -> bool:
    """Reject obvious non-skill phrases."""

    value = skill.strip().lower()

    if not value:
        return False

    if value in _REJECT_EXACT:
        return False

    if len(value) > 70:
        return False

    for pattern in _REJECT_PATTERNS:
        if re.search(pattern, value, flags=re.IGNORECASE):
            return False

    # Reject sentence-like output.
    if value.endswith((".", ":", ";")):
        return False

    if value.count(" ") > 7:
        return False

    return True


# ---------------------------------------------------------------------------
# NORMALIZATION
# ---------------------------------------------------------------------------

def _format_skill_name(raw: str) -> str:
    key = raw.strip().lower()
    canonical = _SYNONYM_TO_CANONICAL.get(key, key)

    if canonical in _DISPLAY_NAMES:
        return _DISPLAY_NAMES[canonical]

    words = canonical.split()

    formatted = [
        word.upper() if word in _KNOWN_ACRONYMS else word.capitalize()
        for word in words
    ]

    return " ".join(formatted) if formatted else raw.strip()


def _canonical_key(display_name: str) -> str:
    key = display_name.strip().lower()
    return _SYNONYM_TO_CANONICAL.get(key, key)


def _dedupe_and_format(raw_skills: List[str]) -> List[str]:
    """Normalize, filter and deduplicate extracted skills."""

    seen: Dict[str, str] = {}

    for raw in raw_skills:
        if not raw or not raw.strip():
            continue

        if not _is_valid_skill(raw):
            continue

        key = _canonical_key(raw)

        if key not in seen:
            seen[key] = _format_skill_name(raw)

    return sorted(seen.values())


def canonical_key(name: str) -> str:
    return _canonical_key(name)


# ---------------------------------------------------------------------------
# API / FALLBACK STATUS
# ---------------------------------------------------------------------------

def _get_api_key() -> Optional[str]:
    """Get Groq API key from environment or Streamlit secrets."""

    api_key = os.environ.get("GROQ_API_KEY")

    if api_key:
        return api_key

    try:
        import streamlit as st
        return st.secrets.get("GROQ_API_KEY", None)
    except Exception:
        return None


def is_llm_configured() -> bool:
    return bool(_get_api_key())


_fallback_used: bool = False
_fallback_reason: Optional[str] = None


def reset_fallback_tracking() -> None:
    global _fallback_used, _fallback_reason

    _fallback_used = False
    _fallback_reason = None


def get_fallback_status() -> tuple[bool, Optional[str]]:
    return _fallback_used, _fallback_reason


def _mark_fallback(reason: str) -> None:
    global _fallback_used, _fallback_reason

    _fallback_used = True

    if _fallback_reason is None:
        _fallback_reason = reason


# ---------------------------------------------------------------------------
# CONSERVATIVE FALLBACK
# ---------------------------------------------------------------------------

def _dictionary_match(text: str) -> Set[str]:
    """Find known skills using exact word-boundary matching."""

    lower_text = text.lower()
    found: Set[str] = set()

    for synonym, canonical in _SYNONYM_TO_CANONICAL.items():

        pattern = (
            r"(?<![a-zA-Z0-9])"
            + re.escape(synonym.strip())
            + r"(?![a-zA-Z0-9])"
        )

        if re.search(pattern, lower_text):
            found.add(canonical)

    return found


def _noun_chunk_candidates(
    text: str,
    max_chunks: int = 20,
) -> Set[str]:
    """Very conservative fallback candidate extraction."""

    try:
        nlp = load_spacy_model()
    except RuntimeError:
        return set()

    doc = nlp(text[:5000])
    candidates: Set[str] = set()

    for chunk in doc.noun_chunks:
        phrase = chunk.text.strip()

        if not _is_valid_skill(phrase):
            continue

        words = phrase.split()

        # Only allow short phrases.
        if 1 <= len(words) <= 3:
            candidates.add(phrase.lower())

        if len(candidates) >= max_chunks:
            break

    return candidates


# ---------------------------------------------------------------------------
# GROQ LLM EXTRACTION
# ---------------------------------------------------------------------------

def _llm_extract(
    text: str,
    source_label: str,
) -> Optional[List[str]]:
    """Extract skills using Groq GPT OSS 120B."""

    api_key = _get_api_key()

    if not api_key:
        _mark_fallback("No GROQ_API_KEY configured")
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        system_prompt = """
You are a strict ATS skill extraction engine.

Your ONLY job is to identify genuine skills from the supplied text.

A valid skill can be:
- programming language
- framework
- library
- software/tool
- cloud platform
- database
- technical technology
- technical methodology
- professional competency explicitly presented as a skill

DO NOT extract:
- job titles
- names
- companies
- industries
- responsibilities
- achievements
- sentences
- sentence fragments
- nouns that merely happen to appear in a sentence
- generic business phrases
- generic words
- years or numbers
- "experience"
- "background"
- "role"
- "responsibilities"
- "stakeholders"
- "findings"
- "ability to work"
- "cross-functional stakeholders"

For example:

BAD:
Experience
Years
Strong Background
The Role
Cross-functional Stakeholders
Present Findings
Ability to Work Independently

GOOD:
Python
AWS
Docker
Machine Learning
SQL
TensorFlow
Project Management

Only return skills explicitly supported by the text.
Do not invent skills.
Do not infer a skill simply because a responsibility sounds related.

Return JSON ONLY in exactly this format:

{
  "skills": ["Python", "AWS", "Machine Learning"]
}
"""

        user_prompt = f"""
Analyze this {source_label}.

Extract only genuine skills explicitly mentioned in the text.

TEXT:
{text[:8000]}
"""

        response = client.chat.completions.create(
            model="openai/gpt-oss-120b",
            temperature=0,
            max_tokens=600,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
        )

        raw = response.choices[0].message.content or ""

        data = json.loads(raw)

        skills = data.get("skills", [])

        if not isinstance(skills, list):
            _mark_fallback("LLM returned an invalid skills structure")
            return None

        cleaned = [
            str(skill).strip()
            for skill in skills
            if str(skill).strip()
        ]

        cleaned = _dedupe_and_format(cleaned)

        if not cleaned:
            _mark_fallback("LLM returned an empty skill list")
            return None

        return cleaned

    except Exception as exc:
        _mark_fallback(
            f"LLM call failed: {type(exc).__name__}: {exc}"
        )
        return None


# ---------------------------------------------------------------------------
# PUBLIC EXTRACTION FUNCTIONS
# ---------------------------------------------------------------------------

def extract_required_skills(jd_text: str) -> List[str]:
    """Extract required/desired skills from a job description."""

    llm_skills = _llm_extract(
        jd_text,
        "job description",
    )

    if llm_skills:
        return llm_skills

    dictionary_skills = _dictionary_match(jd_text)

    # We intentionally DO NOT add arbitrary spaCy noun chunks to the
    # final fallback anymore. That was the source of noisy results such as
    # "Experience", "Years", "The Role", and "Stakeholders".
    return _dedupe_and_format(
        [_DISPLAY_NAMES.get(skill, skill) for skill in dictionary_skills]
    )


def extract_resume_skills(resume_text: str) -> List[str]:
    """Extract skills present in a resume."""

    llm_skills = _llm_extract(
        resume_text,
        "resume",
    )

    if llm_skills:
        return llm_skills

    dictionary_skills = _dictionary_match(resume_text)

    return _dedupe_and_format(
        [_DISPLAY_NAMES.get(skill, skill) for skill in dictionary_skills]
    )