"""Plain-text (.txt) resume parsing.

Simplest of the three parsers — still guards against bad encodings so a
weird file can't crash the whole app.
"""

from __future__ import annotations

from utils.models import ParsedDocument


def extract_text(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Decode a .txt resume, trying utf-8 first and falling back gracefully."""
    text = ""
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            text = file_bytes.decode(encoding)
            break
        except (UnicodeDecodeError, LookupError):
            continue

    text = text.strip()
    if not text:
        return ParsedDocument(
            text="",
            success=False,
            warnings=[f"'{filename}' is empty or could not be decoded as text."],
        )

    return ParsedDocument(text=text, success=True)
