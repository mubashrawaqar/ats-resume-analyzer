"""PDF parsing for resumes.

Strategy:
1. Try pdfplumber first — fast, and gives us table detection for free
   (useful for the ATS-parseability check).
2. If a page yields almost no text, it's probably a scanned/image-based
   page. Fall back to rendering that page with PyMuPDF and running it
   through Tesseract OCR (pytesseract).
3. If OCR itself isn't available (Tesseract binary missing on the host),
   we don't crash — we surface a clear warning instead so the app keeps
   working for every other file.
"""

from __future__ import annotations

import io
from typing import List

import pdfplumber
import fitz  # PyMuPDF

from utils.models import ParsedDocument

# A page with fewer than this many extracted characters is treated as
# "probably scanned" and sent to OCR instead.
MIN_CHARS_PER_PAGE = 20

# Render resolution for OCR — higher = more accurate but slower.
OCR_ZOOM = 2.0


def _ocr_page(pdf_bytes: bytes, page_index: int) -> str:
    """Render a single PDF page to an image and OCR it with Tesseract.

    Raises if Tesseract isn't installed on the host — caller catches this.
    """
    import pytesseract
    from PIL import Image

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    page = doc[page_index]
    matrix = fitz.Matrix(OCR_ZOOM, OCR_ZOOM)
    pix = page.get_pixmap(matrix=matrix)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    doc.close()
    return pytesseract.image_to_string(img)


def _detect_ats_issues(pdf_bytes: bytes) -> List[str]:
    """Best-effort formatting checks that would trip up a real ATS parser."""
    issues: List[str] = []
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            table_pages = 0
            for page in pdf.pages:
                tables = page.find_tables()
                if tables:
                    table_pages += 1
            if table_pages:
                issues.append(
                    f"Detected table structures on {table_pages} page(s) — "
                    "tables/columns used for layout can cause ATS systems to "
                    "misread or drop content. Prefer plain single-column text."
                )
    except Exception:
        # Table detection is a nice-to-have, never let it break parsing.
        pass
    return issues


def extract_text(file_bytes: bytes, filename: str) -> ParsedDocument:
    """Extract text from a PDF resume, with OCR fallback for scanned pages.

    Never raises — any failure is captured in ParsedDocument.warnings and
    success=False so the UI can show a clear message instead of crashing.
    """
    warnings: List[str] = []
    used_ocr = False
    page_texts: List[str] = []

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").strip()
                if len(text) < MIN_CHARS_PER_PAGE:
                    # Likely a scanned/image page — try OCR.
                    try:
                        ocr_text = _ocr_page(file_bytes, i).strip()
                        if ocr_text:
                            page_texts.append(ocr_text)
                            used_ocr = True
                        else:
                            warnings.append(
                                f"Page {i + 1} appears to be a scanned image "
                                "and OCR found no readable text."
                            )
                    except Exception:
                        warnings.append(
                            f"Page {i + 1} appears to be a scanned image and "
                            "OCR is unavailable on this server (Tesseract not "
                            "installed) — this page's content could not be read."
                        )
                else:
                    page_texts.append(text)
    except Exception as exc:
        return ParsedDocument(
            text="",
            success=False,
            warnings=[f"Could not parse '{filename}' as a PDF: {exc}"],
        )

    full_text = "\n".join(page_texts).strip()
    ats_issues = _detect_ats_issues(file_bytes)

    if not full_text:
        warnings.append(
            f"Could not extract any readable text from '{filename}'. "
            "It may be a scanned document, corrupted, or password-protected."
        )
        return ParsedDocument(
            text="", success=False, used_ocr=used_ocr, warnings=warnings, ats_issues=ats_issues
        )

    return ParsedDocument(
        text=full_text,
        success=True,
        used_ocr=used_ocr,
        warnings=warnings,
        ats_issues=ats_issues,
    )
