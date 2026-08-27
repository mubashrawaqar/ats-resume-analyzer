"""Centralized, cached loading of the heavy NLP models.

Both spaCy (NER) and sentence-transformers (embeddings) take real time
and memory to load. We load each exactly once per app process using
st.cache_resource, and every other module imports from here instead of
loading its own copy.
"""

from __future__ import annotations

import os

import streamlit as st

# Local path to a pre-downloaded sentence-transformers model folder.
# Set this if Hugging Face isn't reachable from your network (e.g. it's
# blocked) — point it at the folder containing config.json,
# model.safetensors, tokenizer.json, the 1_Pooling/ folder, etc.
# Override via env var; falls back to the path used during development.
LOCAL_SENTENCE_TRANSFORMER_PATH = os.environ.get(
    "LOCAL_SENTENCE_TRANSFORMER_PATH",
    r"C:\Users\User\models\all-MiniLM-L6-v2",
)


@st.cache_resource(show_spinner="Loading NLP models (first run only)...")
def load_spacy_model():
    """Load the spaCy English model used for name/skill NER.

    Cached with st.cache_resource because it's an unpicklable model
    object that should live once per app process, not per session.
    """
    import subprocess
    import sys

    import spacy

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        # Model isn't installed as a package (e.g. requirements.txt wheel
        # line was skipped). Try downloading it once at runtime before
        # giving up — this keeps local setup forgiving even if a step
        # was missed, without requiring it every deploy.
        try:
            subprocess.run(
                [sys.executable, "-m", "spacy", "download", "en_core_web_sm"],
                check=True,
                capture_output=True,
            )
            return spacy.load("en_core_web_sm")
        except Exception as exc:
            raise RuntimeError(
                "spaCy model 'en_core_web_sm' is not installed and could not "
                "be downloaded automatically. Run manually: "
                "python -m spacy download en_core_web_sm"
            ) from exc


@st.cache_resource(show_spinner="Loading semantic matching model (first run only)...")
def load_sentence_transformer():
    """Load the sentence-transformers model used for semantic similarity.

    all-MiniLM-L6-v2 is a good default: small (~80MB), fast on CPU, and
    accurate enough for short skill/phrase comparisons — no GPU required,
    which matters since this is meant to run on free hosting.

    Loads from LOCAL_SENTENCE_TRANSFORMER_PATH if that folder exists
    (useful when Hugging Face is blocked on your network) — otherwise
    downloads the model from Hugging Face by name as usual.
    """
    from sentence_transformers import SentenceTransformer

    if os.path.isdir(LOCAL_SENTENCE_TRANSFORMER_PATH):
        return SentenceTransformer(LOCAL_SENTENCE_TRANSFORMER_PATH)

    return SentenceTransformer("all-MiniLM-L6-v2")
