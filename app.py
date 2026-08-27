"""ATS Resume Analyzer — Streamlit UI."""

from __future__ import annotations

import re
from typing import List, Optional

import pandas as pd
import streamlit as st

from matching.scorer import score_resume
from matching.skill_extractor import (
    get_fallback_status,
    is_llm_configured,
    reset_fallback_tracking,
)
from parsers import docx_parser, pdf_parser, txt_parser
from utils.contact_extractor import extract_contact_info
from utils.models import MatchResult, ParsedDocument


# ---------------------------------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {"pdf", "docx", "txt"}

st.set_page_config(
    page_title="ATS Resume Analyzer",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def raw_html(markup: str) -> None:
    """Render a raw HTML snippet via st.markdown.

    Streamlit's Markdown parser follows CommonMark: a blank line followed
    by indented text is parsed as an indented *code block*, not HTML — so
    a pretty-printed, multi-line HTML string with blank lines between tags
    can render as literal text instead of styled HTML. Collapsing the
    string to one line (no newlines, no leading whitespace) sidesteps that
    entirely, regardless of how the caller indents/formats the markup.
    """
    compact = re.sub(r"\n\s*", "", markup.strip())
    st.markdown(compact, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# CUSTOM UI
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>

    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&family=Inter:wght@400;500;600;700&display=swap');

    /* ---------- GLOBAL ---------- */

    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        background:
            radial-gradient(
                circle at 8% 0%,
                rgba(99, 102, 241, 0.22),
                transparent 42%
            ),
            radial-gradient(
                circle at 92% 8%,
                rgba(14, 165, 233, 0.20),
                transparent 40%
            ),
            radial-gradient(
                circle at 50% 100%,
                rgba(129, 140, 248, 0.14),
                transparent 55%
            ),
            #eef1fb;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ---------- SIDEBAR ---------- */

    section[data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #111827 0%,
            #172033 100%
        );
        border-right: 1px solid rgba(255,255,255,0.08);
    }

    section[data-testid="stSidebar"] * {
        color: #f8fafc !important;
    }

    /* Cards placed inside the dark sidebar keep their white background —
       without this override, the sidebar's blanket "everything is white
       text" rule above would make white text sit on a white card and
       vanish. */
    section[data-testid="stSidebar"] .info-card,
    section[data-testid="stSidebar"] .info-card * {
        color: #111827 !important;
    }
    section[data-testid="stSidebar"] .info-card-small {
        color: #475569 !important;
    }

    section[data-testid="stSidebar"] textarea {
        background: #1f2937 !important;
        border: 1px solid #374151 !important;
        color: #f8fafc !important;
        border-radius: 10px !important;
    }

    section[data-testid="stSidebar"] .stFileUploader {
        background: #1f2937;
        border-radius: 12px;
        padding: 8px;
    }

    /* ---------- HERO ---------- */

    .hero {
        background:
            linear-gradient(
                135deg,
                #111827 0%,
                #1e293b 55%,
                #312e81 100%
            );
        padding: 34px 38px;
        border-radius: 20px;
        margin-bottom: 28px;
        box-shadow: 0 15px 45px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255,255,255,0.08);
    }

    .hero-title {
        font-family: 'Poppins', 'Inter', sans-serif;
        color: white;
        font-size: 38px;
        font-weight: 800;
        margin: 0;
        letter-spacing: -1.2px;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 16px;
        margin-top: 10px;
        max-width: 850px;
        line-height: 1.6;
    }

    .hero-badge {
        display: inline-block;
        margin-top: 18px;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(255,255,255,0.10);
        color: #e0e7ff;
        font-size: 13px;
        font-weight: 600;
    }

    /* ---------- SECTION HEADINGS ---------- */

    .section-title {
        font-size: 24px;
        font-weight: 750;
        color: #111827;
        margin-top: 12px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #64748b;
        margin-bottom: 18px;
    }

    /* ---------- CARDS ---------- */

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.05);
        margin-bottom: 14px;
    }

    .info-card-title {
        font-size: 15px;
        font-weight: 700;
        color: #334155;
        margin-bottom: 6px;
    }

    .info-card-value {
        font-size: 26px;
        font-weight: 800;
        color: #111827;
    }

    .info-card-small {
        color: #64748b;
        font-size: 13px;
    }

    /* ---------- SKILL BADGES ---------- */

    .skill-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 10px;
    }

    .skill-badge {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        background: #eef2ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
        font-size: 13px;
        font-weight: 600;
    }

    .skill-badge-missing {
        display: inline-block;
        padding: 7px 11px;
        border-radius: 999px;
        background: #fff1f2;
        color: #be123c;
        border: 1px solid #fecdd3;
        font-size: 13px;
        font-weight: 600;
    }

    /* ---------- RESULT CARD ---------- */

    .resume-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 18px;
        padding: 22px;
        margin-bottom: 16px;
        box-shadow: 0 5px 20px rgba(15, 23, 42, 0.06);
    }

    .resume-name {
        font-size: 20px;
        font-weight: 750;
        color: #111827;
    }

    .resume-meta {
        color: #64748b;
        font-size: 13px;
        margin-top: 4px;
    }

    /* ---------- SCORE ---------- */

    .score-label {
        font-size: 13px;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .score-number {
        font-size: 38px;
        font-weight: 850;
        color: #111827;
        line-height: 1.1;
    }

    /* ---------- BUTTON ---------- */

    .stButton > button {
        border-radius: 10px;
        font-weight: 700;
        min-height: 44px;
    }

    /* ---------- DATAFRAME ---------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        border: 1px solid #e2e8f0;
    }

    /* ---------- FOOTER ---------- */

    .footer {
        text-align: center;
        color: #94a3b8;
        font-size: 12px;
        margin-top: 45px;
        padding-top: 20px;
        border-top: 1px solid #e2e8f0;
    }

</style>
""",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def get_extension(filename: str) -> str:
    return (
        filename.rsplit(".", 1)[-1].lower()
        if "." in filename
        else ""
    )


def parse_file(
    uploaded_file,
) -> Optional[ParsedDocument]:
    """Route uploaded file to the appropriate parser."""

    ext = get_extension(uploaded_file.name)
    file_bytes = uploaded_file.getvalue()

    if ext == "pdf":
        return pdf_parser.extract_text(
            file_bytes,
            uploaded_file.name,
        )

    if ext == "docx":
        return docx_parser.extract_text(
            file_bytes,
            uploaded_file.name,
        )

    if ext == "txt":
        return txt_parser.extract_text(
            file_bytes,
            uploaded_file.name,
        )

    st.error(
        f"Unsupported file type: '{uploaded_file.name}'. "
        f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}."
    )

    return None


def process_resumes(
    uploaded_files,
    jd_text: str,
) -> List[MatchResult]:

    results: List[MatchResult] = []

    progress = st.progress(
        0,
        text="Preparing analysis...",
    )

    total = len(uploaded_files)

    for index, uploaded_file in enumerate(uploaded_files):

        progress.progress(
            index / total,
            text=f"Analyzing {uploaded_file.name}...",
        )

        parsed = parse_file(uploaded_file)

        if parsed is None:
            continue

        if not parsed.success or not parsed.text.strip():

            for warning in parsed.warnings:
                st.warning(
                    f"{uploaded_file.name}: {warning}"
                )

            continue

        for warning in parsed.warnings:
            st.info(
                f"{uploaded_file.name}: {warning}"
            )

        contact = extract_contact_info(
            parsed.text
        )

        result = score_resume(
            resume_text=parsed.text,
            jd_text=jd_text,
            filename=uploaded_file.name,
            contact=contact,
            ats_issues=parsed.ats_issues,
            parse_warnings=parsed.warnings,
        )

        results.append(result)

    progress.progress(
        1.0,
        text="Analysis complete.",
    )

    return results


# ---------------------------------------------------------------------------
# UI COMPONENTS
# ---------------------------------------------------------------------------

def render_skill_badges(
    skills: List[str],
    missing: bool = False,
) -> None:

    if not skills:
        st.caption(
            "None identified."
            if not missing
            else "No missing skills identified."
        )
        return

    css_class = (
        "skill-badge-missing"
        if missing
        else "skill-badge"
    )

    badges_markup = '<div class="skill-wrap">'

    for skill in skills:
        badges_markup += (
            f'<span class="{css_class}">'
            f"{skill}"
            "</span>"
        )

    badges_markup += "</div>"

    raw_html(badges_markup)


def score_color(score: float) -> str:
    if score >= 80:
        return "Excellent"
    if score >= 65:
        return "Strong"
    if score >= 50:
        return "Moderate"
    return "Low"


def render_single_result(
    result: MatchResult,
) -> None:

    raw_html(
        f"""
        <div class="resume-card">
            <div class="resume-name">
                {result.contact.name}
            </div>
            <div class="resume-meta">
                {result.filename}
            </div>
        </div>
        """
    )

    # Contact information
    c1, c2, c3 = st.columns(3)

    with c1:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">Email</div>
                <div>{result.contact.email}</div>
            </div>
            """
        )

    with c2:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">Phone</div>
                <div>{result.contact.phone}</div>
            </div>
            """
        )

    with c3:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">Match Rating</div>
                <div class="info-card-value">
                    {score_color(result.match_score)}
                </div>
            </div>
            """
        )

    # Score
    raw_html('<div class="section-title">Resume Match Score</div>')

    score_col1, score_col2 = st.columns(
        [1, 4]
    )

    with score_col1:
        raw_html(
            f"""
            <div>
                <div class="score-label">Overall Match</div>
                <div class="score-number">
                    {result.match_score:.1f}
                </div>
                <div class="resume-meta">out of 100</div>
            </div>
            """
        )

    with score_col2:
        st.write("")
        st.progress(
            min(max(result.match_score / 100, 0), 1)
        )

    raw_html('<div class="section-title">Score Breakdown</div>')

    b1, b2, b3 = st.columns(3)

    with b1:
        st.metric(
            "Skills Match",
            f"{result.breakdown.skills_match_pct:.1f}%",
        )

    with b2:
        st.metric(
            "Keyword Density",
            f"{result.breakdown.keyword_density_pct:.1f}%",
        )

    with b3:
        st.metric(
            "Experience Relevance",
            f"{result.breakdown.experience_relevance_pct:.1f}%",
        )

    # Skills
    raw_html('<div class="section-title">Skill Analysis</div>')

    s1, s2 = st.columns(2)

    with s1:
        st.markdown(
            "**✅ Matching Skills**"
        )
        render_skill_badges(
            result.matching_skills
        )

    with s2:
        st.markdown(
            "**⚠️ Missing Skills**"
        )
        render_skill_badges(
            result.missing_skills,
            missing=True,
        )

    # ATS issues
    if result.ats_issues:

        raw_html('<div class="section-title">ATS Parseability</div>')

        for issue in result.ats_issues:
            st.warning(issue)


def render_multi_results(
    results: List[MatchResult],
) -> None:

    sorted_results = sorted(
        results,
        key=lambda result: result.match_score,
        reverse=True,
    )

    # Summary cards
    avg_score = sum(
        result.match_score
        for result in sorted_results
    ) / len(sorted_results)

    best_score = sorted_results[0].match_score

    c1, c2, c3 = st.columns(3)

    with c1:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">
                    Resumes Analyzed
                </div>
                <div class="info-card-value">
                    {len(sorted_results)}
                </div>
                <div class="info-card-small">
                    Successfully processed
                </div>
            </div>
            """
        )

    with c2:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">
                    Average Match
                </div>
                <div class="info-card-value">
                    {avg_score:.1f}
                </div>
                <div class="info-card-small">
                    Across all candidates
                </div>
            </div>
            """
        )

    with c3:
        raw_html(
            f"""
            <div class="info-card">
                <div class="info-card-title">
                    Top Candidate
                </div>
                <div class="info-card-value">
                    {best_score:.1f}
                </div>
                <div class="info-card-small">
                    Highest match score
                </div>
            </div>
            """
        )

    raw_html('<div class="section-title">Candidate Ranking</div>')

    raw_html(
        '<div class="section-description">'
        "Candidates ranked by overall resume-to-job match."
        "</div>"
    )

    table_data = []

    for rank, result in enumerate(
        sorted_results,
        start=1,
    ):
        table_data.append(
            {
                "Rank": rank,
                "Candidate": result.contact.name,
                "Email": result.contact.email,
                "Phone": result.contact.phone,
                "Match Score": round(result.match_score, 1),
                "Skills Match %": round(result.breakdown.skills_match_pct, 1),
                "Keyword Density %": round(result.breakdown.keyword_density_pct, 1),
                "Experience Relevance %": round(
                    result.breakdown.experience_relevance_pct, 1
                ),
                "Skills Matched": len(result.matching_skills),
                "Skills Missing": len(result.missing_skills),
                "ATS Issues": len(result.ats_issues),
                "Matching Skills": ", ".join(result.matching_skills),
                "Missing Skills": ", ".join(result.missing_skills),
            }
        )

    df = pd.DataFrame(table_data)

    display_columns = [
        "Rank", "Candidate", "Email", "Phone", "Match Score",
        "Skills Match %", "Keyword Density %", "Experience Relevance %",
        "Skills Matched", "Skills Missing", "ATS Issues",
    ]

    st.dataframe(
        df[display_columns],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match Score": st.column_config.ProgressColumn(
                "Match Score",
                min_value=0,
                max_value=100,
                format="%.1f",
            ),
            "Skills Match %": st.column_config.NumberColumn(format="%.1f%%"),
            "Keyword Density %": st.column_config.NumberColumn(format="%.1f%%"),
            "Experience Relevance %": st.column_config.NumberColumn(format="%.1f%%"),
        },
    )

    st.caption(
        "Full matching/missing skill names are listed in each "
        "candidate's details below, and in the CSV export."
    )

    csv_bytes = df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        "⬇️ Download Full Analysis as CSV",
        data=csv_bytes,
        file_name="ats_resume_analysis.csv",
        mime="text/csv",
        use_container_width=False,
    )

    raw_html('<div class="section-title">Candidate Details</div>')

    for result in sorted_results:

        with st.expander(
            f"{result.contact.name}  •  "
            f"{result.match_score:.1f}/100  •  "
            f"{result.filename}"
        ):
            render_single_result(result)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main() -> None:

    # Hero
    llm_status = (
        "AI-powered skill extraction"
        if is_llm_configured()
        else "Dictionary extraction mode"
    )

    raw_html(
        f"""
        <div class="hero">
            <div class="hero-title">
                📄 ATS Resume Analyzer
            </div>
            <div class="hero-subtitle">
                Analyze resumes against a job description using
                semantic matching, skill extraction, keyword analysis,
                and ATS parseability checks.
            </div>
            <div class="hero-badge">
                ✦ {llm_status}
            </div>
        </div>
        """
    )

    # Sidebar — informational cards only; inputs live in the main panel.
    with st.sidebar:

        st.markdown("## ✨ How It Works")
        st.caption(
            "A quick overview of what this tool checks for you."
        )

        raw_html(
            """
            <div class="info-card">
                <div class="info-card-title">🚀 Ready to Analyze</div>
                <div class="info-card-small">
                    Fill in the job description and upload candidate
                    resumes in the main panel, then click
                    <b>Analyze Resumes</b>.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">🎯 Skill Matching</div>
                <div class="info-card-small">
                    Identify skills that candidates have and skills
                    they are missing.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">📊 Candidate Ranking</div>
                <div class="info-card-small">
                    Compare multiple resumes and rank candidates by
                    match score.
                </div>
            </div>
            <div class="info-card">
                <div class="info-card-title">🛡️ ATS Analysis</div>
                <div class="info-card-small">
                    Detect resume parsing and ATS compatibility
                    issues.
                </div>
            </div>
            """
        )

        st.markdown("---")
        st.caption("ATS Resume Analyzer")
        st.caption("Hybrid keyword + semantic analysis")

    # Main panel — job description + resume upload live here now.
    col_jd, col_resume = st.columns([3, 2])

    with col_jd:
        raw_html('<div class="section-title">1 · Job Description</div>')
        jd_text = st.text_area(
            "Paste the job description",
            height=320,
            placeholder=(
                "Paste the complete job description here..."
            ),
            label_visibility="collapsed",
        )

    with col_resume:
        raw_html('<div class="section-title">2 · Candidate Resumes</div>')
        uploaded_files = st.file_uploader(
            "Upload resumes",
            type=list(SUPPORTED_EXTENSIONS),
            accept_multiple_files=True,
            help="Supported: PDF, DOCX and TXT",
            label_visibility="collapsed",
        )

        if uploaded_files:
            st.success(
                f"{len(uploaded_files)} resume(s) ready"
            )

        analyze_clicked = st.button(
            "🚀 Analyze Resumes",
            type="primary",
            use_container_width=True,
        )

    # Empty state
    if not analyze_clicked:
        raw_html(
            """
            <div class="footer">
                Built for practical resume screening and
                explainable candidate matching.
            </div>
            """
        )
        return

    # Validation
    if not jd_text or not jd_text.strip():
        st.error(
            "Please provide a job description."
        )
        return

    if len(jd_text.strip()) < 30:
        st.warning(
            "The job description is very short. "
            "For reliable matching, provide the complete JD."
        )

    if not uploaded_files:
        st.error(
            "Please upload at least one resume."
        )
        return

    # Analysis
    reset_fallback_tracking()

    results = process_resumes(
        uploaded_files,
        jd_text,
    )

    fallback_used, fallback_reason = (
        get_fallback_status()
    )

    if fallback_used:
        st.warning(
            "Skill extraction used the fallback method "
            "for at least one document. "
            f"Reason: {fallback_reason}"
        )
    else:
        st.success(
            "AI skill extraction completed successfully."
        )

    if not results:
        st.error(
            "No resumes could be successfully parsed. "
            "Check the warnings and try again."
        )
        return

    raw_html(
        f"""
        <div class="section-title">
            Analysis Results
        </div>
        <div class="section-description">
            Successfully analyzed
            <b>{len(results)}</b> of
            <b>{len(uploaded_files)}</b> uploaded resume(s).
        </div>
        """
    )

    if len(results) == 1:
        render_single_result(results[0])
    else:
        render_multi_results(results)

    raw_html(
        """
        <div class="footer">
            ATS Resume Analyzer · AI-assisted resume screening
        </div>
        """
    )


if __name__ == "__main__":
    main()
