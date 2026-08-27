"""Hybrid matching engine: exact keyword matching + semantic similarity.

Why hybrid, not just one or the other:
- Pure keyword matching misses synonyms and related phrasing ("ML" vs
  "machine learning" vs "predictive modeling"). A candidate who is
  genuinely qualified can score poorly just because they phrased things
  differently than the JD.
- Pure semantic similarity (embeddings only) is too forgiving — it can
  rate loosely-related content as a "match" when the resume never
  actually mentions the skill, which defeats the point of an ATS check.

So we combine them: exact/dictionary matches are trusted directly, and
only the *remaining unmatched* required skills get a second chance via
embedding similarity against everything the resume does mention. This
keeps the precision of keyword matching while recovering the recall that
pure keyword matching loses.

The final match_score is a weighted blend of three explainable
components (see ScoreBreakdown) rather than a single opaque number, so
users can see *why* they got the score they did.
"""

from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np
import streamlit as st
from sentence_transformers import util

from matching.skill_extractor import (
    canonical_key,
    extract_required_skills,
    extract_resume_skills,
)
from utils.models import MatchResult, ScoreBreakdown, ContactInfo
from utils.nlp_models import load_sentence_transformer

# Cosine similarity above this threshold counts as a semantic skill match
# for an otherwise-unmatched required skill (e.g. JD says "cloud
# infrastructure", resume says "AWS/Azure deployment").
SEMANTIC_MATCH_THRESHOLD = 0.55

# Score weights — must sum to 1.0. Skills match is weighted highest since
# it's the most direct signal of JD fit; keyword density rewards genuine
# emphasis over a single passing mention; experience relevance captures
# overall contextual fit that skill lists alone can miss.
WEIGHT_SKILLS_MATCH = 0.5
WEIGHT_KEYWORD_DENSITY = 0.2
WEIGHT_EXPERIENCE_RELEVANCE = 0.3


@st.cache_data(show_spinner=False)
def _embed_text(text: str) -> np.ndarray:
    """Compute (and cache) a sentence embedding for a piece of text.

    Cached by st.cache_data, keyed on the text itself — so re-scoring the
    same resume against a different JD in the same session doesn't
    recompute the resume's embedding from scratch.
    """
    model = load_sentence_transformer()
    return model.encode(text, convert_to_numpy=True)


@st.cache_data(show_spinner=False)
def _embed_skills(skills: Tuple[str, ...]) -> np.ndarray:
    """Compute (and cache) embeddings for a list of skill phrases at once.

    Batched encoding is much faster than encoding one skill at a time.
    Takes a tuple (not a list) because st.cache_data requires hashable
    arguments.
    """
    if not skills:
        return np.empty((0, 384))  # all-MiniLM-L6-v2 embedding dimension
    model = load_sentence_transformer()
    return model.encode(list(skills), convert_to_numpy=True)


def _semantic_skill_match(
    unmatched_required: List[str], resume_skills: List[str]
) -> List[str]:
    """Recover semantic matches for required skills not found by exact match.

    For each unmatched required skill, compare its embedding against every
    skill phrase the resume actually contains. If the best match clears
    SEMANTIC_MATCH_THRESHOLD, treat it as matched.
    """
    if not unmatched_required or not resume_skills:
        return []

    required_emb = _embed_skills(tuple(unmatched_required))
    resume_emb = _embed_skills(tuple(resume_skills))

    similarity_matrix = util.cos_sim(required_emb, resume_emb).numpy()
    matched: List[str] = []
    for i, skill in enumerate(unmatched_required):
        best_score = float(np.max(similarity_matrix[i])) if similarity_matrix.shape[1] else 0.0
        if best_score >= SEMANTIC_MATCH_THRESHOLD:
            matched.append(skill)
    return matched


def _keyword_density_pct(matching_skills: List[str], resume_text: str) -> float:
    """% of matched skills that appear more than once in the resume text.

    This is distinct from skills_match_pct: it measures *emphasis* (does
    the resume genuinely develop the skill, or mention it once in a list?)
    rather than mere presence.
    """
    if not matching_skills:
        return 0.0
    lower_text = resume_text.lower()
    repeated = 0
    for skill in matching_skills:
        # Count occurrences of the skill phrase (word-boundary safe).
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(skill.lower()) + r"(?![a-zA-Z0-9])"
        occurrences = len(re.findall(pattern, lower_text))
        if occurrences >= 2:
            repeated += 1
    return round((repeated / len(matching_skills)) * 100, 1)


def _experience_relevance_pct(jd_text: str, resume_text: str) -> float:
    """Overall semantic similarity between the full JD and full resume.

    Captures contextual/experience fit beyond an itemized skill list —
    e.g. years of relevant industry experience described in prose that
    wouldn't show up as a discrete "skill".
    """
    jd_emb = _embed_text(jd_text)
    resume_emb = _embed_text(resume_text)
    similarity = float(util.cos_sim(jd_emb, resume_emb)[0][0])
    # Cosine similarity for unrelated professional text rarely goes below
    # ~0.1, so rescale to use the 0-100 range more meaningfully.
    scaled = max(0.0, min(1.0, (similarity - 0.1) / 0.6))
    return round(scaled * 100, 1)


def score_resume(
    resume_text: str,
    jd_text: str,
    filename: str,
    contact: ContactInfo,
    ats_issues: List[str],
    parse_warnings: List[str],
) -> MatchResult:
    """Compute the full hybrid match result for one resume against one JD."""
    required_skills = extract_required_skills(jd_text)
    resume_skills = extract_resume_skills(resume_text)

    # Compare by canonical key (case/source-insensitive) but always display
    # using the required-skill list's casing, since that's what the JD
    # actually asked for.
    required_by_key = {canonical_key(s): s for s in required_skills}
    resume_keys = {canonical_key(s) for s in resume_skills}

    # Step 1: exact/canonical matches.
    matched_keys = set(required_by_key) & resume_keys
    unmatched_keys = set(required_by_key) - matched_keys
    unmatched_display = [required_by_key[k] for k in unmatched_keys]

    # Step 2: recover semantic matches for anything exact matching missed.
    semantic_matches = _semantic_skill_match(unmatched_display, resume_skills)
    matched_keys |= {canonical_key(s) for s in semantic_matches}

    matching_skills = sorted(required_by_key[k] for k in matched_keys)
    missing_skills = sorted(
        required_by_key[k] for k in set(required_by_key) - matched_keys
    )

    skills_match_pct = (
        round((len(matching_skills) / len(required_by_key)) * 100, 1)
        if required_by_key else 0.0
    )
    keyword_density_pct = _keyword_density_pct(matching_skills, resume_text)
    experience_relevance_pct = _experience_relevance_pct(jd_text, resume_text)

    breakdown = ScoreBreakdown(
        skills_match_pct=skills_match_pct,
        keyword_density_pct=keyword_density_pct,
        experience_relevance_pct=experience_relevance_pct,
    )

    match_score = round(
        breakdown.skills_match_pct * WEIGHT_SKILLS_MATCH
        + breakdown.keyword_density_pct * WEIGHT_KEYWORD_DENSITY
        + breakdown.experience_relevance_pct * WEIGHT_EXPERIENCE_RELEVANCE,
        1,
    )

    return MatchResult(
        filename=filename,
        contact=contact,
        matching_skills=matching_skills,
        missing_skills=missing_skills,
        match_score=match_score,
        breakdown=breakdown,
        ats_issues=ats_issues,
        parse_warnings=parse_warnings,
    )
