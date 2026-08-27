# 📄 ATS Resume Analyzer

🔗 **Live demo:** [ats-resume-analyzer123.streamlit.app](https://ats-resume-analyzer123.streamlit.app)

![ATS Resume Analyzer Demo](https://github.com/user-attachments/assets/f5eb1222-8cf1-4d8f-9a87-0e1cab24aae2)


---
A Streamlit app that scores resumes against a job description the way a
real Applicant Tracking System (ATS) would — combining exact keyword
matching with semantic similarity, so it doesn't unfairly penalize
candidates who describe the same skill in different words. It also flags
resume-formatting issues that would confuse real ATS parsers.

## Features

- Upload one or many resumes (`.pdf`, `.docx`, `.txt`) plus a job description
- Automatic OCR fallback for scanned/image-based PDF resumes (Tesseract)
- Candidate name, email, and phone extraction (regex + spaCy NER)
- Hybrid skill matching: keyword/dictionary matching **and** semantic
  (embedding) similarity, so "ML" and "machine learning" — or "cloud
  infrastructure" and "AWS" — are correctly recognized as related
- Explainable 0–100 match score, broken into three visible components:
  skills match %, keyword density %, and experience relevance %
- Single-resume view (detailed) and multi-resume view (sortable table +
  CSV export)
- ATS parseability check: flags tables/text boxes, missing section
  headers, and other formatting that confuses real ATS software
- 2–4 actionable improvement suggestions per resume

## Tech Stack

| Layer | Technology |
|---|---|
| App framework | Streamlit (single app, no separate frontend/backend) |
| NLP / NER | spaCy (`en_core_web_sm`) |
| Semantic similarity | sentence-transformers (`all-MiniLM-L6-v2`) + cosine similarity |
| File parsing | pdfplumber / PyMuPDF (PDF), python-docx (DOCX), plain read (TXT) |
| OCR fallback | pytesseract (Tesseract OCR) |
| Storage | None — everything is processed in-memory via `st.session_state` per session |

## Project Structure

```
app.py                       # Streamlit UI only — no parsing/NLP logic
parsers/
    pdf_parser.py             # PDF text extraction + OCR fallback + table detection
    docx_parser.py             # DOCX text extraction + ATS issue detection
    txt_parser.py               # Plain text handling
matching/
    skill_extractor.py        # Skill dictionary + NER + optional LLM extraction
    scorer.py                  # Hybrid keyword + semantic scoring engine
utils/
    contact_extractor.py      # Name/email/phone extraction
    models.py                  # Shared dataclasses (ParsedDocument, MatchResult, etc.)
    nlp_models.py               # Cached model loading (spaCy, sentence-transformers)
requirements.txt
packages.txt                  # System package for Streamlit Cloud (tesseract-ocr)
README.md
```

## Local Setup (VS Code)

### 1. Prerequisites

- Python 3.10 or newer ([download here](https://www.python.org/downloads/))
- [VS Code](https://code.visualstudio.com/) with the Python extension installed
- (Optional, for OCR on scanned PDFs) [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) installed on your system:
  - **Windows**: download the installer from the [Tesseract wiki](https://github.com/UB-Mannheim/tesseract/wiki), and make sure it's added to your PATH
  - **macOS**: `brew install tesseract`
  - **Linux**: `sudo apt install tesseract-ocr`

### 2. Create the project folder

```bash
mkdir ats-resume-analyzer
cd ats-resume-analyzer
```

Copy all the project files (`app.py`, `parsers/`, `matching/`, `utils/`,
`requirements.txt`, `packages.txt`, `.gitignore`, `README.md`) into this
folder, preserving the folder structure above.

Open the folder in VS Code:

```bash
code .
```

### 3. Create and activate a virtual environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

You'll know it worked when your terminal prompt shows `(venv)` at the
start. In VS Code, also select this environment as your Python
interpreter: `Ctrl+Shift+P` → `Python: Select Interpreter` → choose the
one inside `venv`.

### 4. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

This will also install the spaCy `en_core_web_sm` model automatically
(it's listed as a direct wheel URL in `requirements.txt`). If it's ever
missing for any reason, install it manually with:

```bash
python -m spacy download en_core_web_sm
```

### 5. Run the app

```bash
streamlit run app.py
```

This opens the app in your browser at `http://localhost:8501`.

### Optional: LLM-based skill extraction

By default, skill extraction uses a built-in dictionary + spaCy NER — no
API key needed. If you want to try LLM-based extraction instead, create
`.streamlit/secrets.toml` in your project folder with:

```toml
ANTHROPIC_API_KEY = "your-key-here"
```

This file is already excluded via `.gitignore` so your key won't be
committed. You'll also need to `pip install anthropic` for this optional
path.

## Testing Checklist

Run through these to confirm each feature works before deploying:

- [ ] **Single resume**: upload one `.pdf` resume + a job description → see name/email/phone, prominent match score, score breakdown, and matching/missing skill tags
- [ ] **Multiple resumes**: upload 2–3 resumes of mixed formats (`.pdf`, `.docx`, `.txt`) → see a sortable table, sorted by match score descending
- [ ] **CSV export**: click "Download results as CSV" in multi-resume mode → confirm the file opens correctly with all expected columns
- [ ] **Unsupported file type**: try uploading a `.jpg` or `.png` → confirm `st.error` is shown and the app doesn't crash (the uploader's `type` filter should also prevent selection in the first place)
- [ ] **Missing job description**: click "Analyze" with the JD box empty → confirm `st.error` appears instead of a crash
- [ ] **Scanned/image-based PDF**: upload a resume that's a scanned image → confirm OCR fallback runs (or a clear warning appears if Tesseract isn't installed)
- [ ] **ATS issues**: upload a resume built with a table-based layout → confirm the ATS parseability warning appears
- [ ] **Improvement suggestions**: expand the "Improvement Suggestions" section for a resume with missing skills → confirm 2–4 relevant suggestions appear

## GitHub Setup

### 1. Initialize the repository

From your project folder (with `venv` activated):

```bash
git init
git add .
git commit -m "Initial commit: ATS Resume Analyzer"
```

### 2. Create the repo on GitHub

1. Go to [github.com/new](https://github.com/new)
2. Name it `ats-resume-analyzer` (or whatever you prefer)
3. Leave "Initialize with README" **unchecked** (you already have one)
4. Click **Create repository**

### 3. Push your code

GitHub will show you commands — they'll look like this (replace `YOUR-USERNAME`):

```bash
git remote add origin https://github.com/YOUR-USERNAME/ats-resume-analyzer.git
git branch -M main
git push -u origin main
```

## Free Deployment (Streamlit Community Cloud)

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account
2. Click **New app**
3. Select your `ats-resume-analyzer` repository, branch `main`, and set the main file path to `app.py`
4. Click **Deploy**

Streamlit Cloud will automatically:
- Install everything in `requirements.txt`, including the spaCy model (via the wheel URL)
- Install `tesseract-ocr` as a system package, since it's listed in `packages.txt` — this enables the OCR fallback for scanned PDFs in the deployed app

The first deploy takes a few minutes since it's downloading the
sentence-transformers model too. Subsequent restarts are faster.

If you configured an Anthropic API key for optional LLM-based skill
extraction, add it under your app's **Settings → Secrets** in the
Streamlit Cloud dashboard, in the same `ANTHROPIC_API_KEY = "..."` format
as your local `secrets.toml`.

## Key Design Decisions

**Why hybrid keyword + semantic matching, not just one or the other?**
Pure keyword matching misses synonyms and related phrasing — a candidate
who wrote "ML" instead of "machine learning," or "AWS" instead of "cloud
infrastructure," would be unfairly marked as missing that skill even
though they have it. But pure semantic similarity is *too* forgiving: it
can rate loosely related content as a "match" even when the resume never
actually mentions the skill, which defeats the purpose of an ATS check.

The app resolves this by using exact/dictionary matching first (trusted
directly, since it's precise), and only sending the *remaining unmatched*
required skills through a second semantic-similarity pass against
whatever skills the resume does contain. This keeps the precision of
keyword matching while recovering the recall that keyword-only matching
loses — matching how a well-built real-world ATS actually behaves.

**Why three separate score components instead of one number?**
A single match score hides *why* a resume scored the way it did. Skills
match % measures direct keyword/semantic coverage; keyword density %
measures whether matched skills are genuinely emphasized (mentioned more
than once) versus a token mention; experience relevance % captures
overall contextual fit between the full resume and full JD via document-
level embedding similarity — catching relevant experience described in
prose that wouldn't show up as a discrete "skill." Together they're more
explainable — and more actionable — than a single black-box percentage.

**Why in-memory processing only?**
This is a portfolio/demo tool, not a system meant to store candidate PII.
Keeping everything in `st.session_state` for the duration of a session
(and never writing resumes to disk or a database) avoids taking on data
retention and privacy obligations the app doesn't need.
