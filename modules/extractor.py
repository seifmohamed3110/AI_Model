"""
modules/extractor.py

Responsibilities:
- Extract raw text from PDF files using PyMuPDF
- Extract raw text from DOCX files using python-docx
- Clean the extracted text (normalize whitespace, remove junk characters)

No Flask. No ML. Just text extraction.
"""

import re
import fitz          # PyMuPDF
from docx import Document


def extract_text(file_path: str) -> str:
    """
    Detect file type and extract plain text.
    Returns a clean string or raises ValueError for unsupported types.
    """
    path = file_path.strip().lower()

    if path.endswith(".pdf"):
        raw = _extract_pdf(file_path)
    elif path.endswith(".docx"):
        raw = _extract_docx(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return clean_text(raw)


def _extract_pdf(file_path: str) -> str:
    """Extract text from all pages of a PDF."""
    text_parts = []
    with fitz.open(file_path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n".join(text_parts)


def _extract_docx(file_path: str) -> str:
    """Extract text from all paragraphs of a DOCX file."""
    doc = Document(file_path)
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def clean_text(text: str) -> str:
    """
    Normalize extracted text:
    - Replace non-breaking spaces and other unicode spaces with regular space
    - Collapse multiple spaces into one
    - Collapse more than 2 consecutive newlines into 2
    - Strip leading/trailing whitespace
    """
    # Replace unicode whitespace variants with normal space
    text = re.sub(r"[^\S\n]", " ", text)

    # Remove null bytes and other control characters except newlines
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # Collapse multiple spaces into one
    text = re.sub(r" {2,}", " ", text)

    # Collapse more than 2 newlines into 2
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()