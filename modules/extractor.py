"""
modules/extractor.py
---------------------
Handles text extraction from:
  - PDF files  → using PyMuPDF (fitz)
  - Image files → using Tesseract OCR via pytesseract + Pillow
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

# ---------------------------------------------------------------------------
# Supported file type groups
# ---------------------------------------------------------------------------
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
ALLOWED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS

# Max file size allowed: 10 MB
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024


def validate_file(file_path: str) -> Tuple[bool, str]:
    """
    Validates that the file exists, is an allowed type, and is within size limits.

    Returns:
        (True, "") on success
        (False, error_message) on failure
    """
    path = Path(file_path)

    if not path.exists():
        return False, f"File not found: {file_path}"

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        return False, (
            f"Unsupported file type '{path.suffix}'. "
            f"Allowed types: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        )

    file_size = path.stat().st_size
    if file_size > MAX_FILE_SIZE_BYTES:
        size_mb = file_size / (1024 * 1024)
        return False, f"File too large ({size_mb:.1f} MB). Maximum allowed size is 10 MB."

    return True, ""


def extract_text_from_pdf(file_path: str) -> str:
    """
    Extracts plain text from all pages of a PDF using PyMuPDF.
    Pages are joined with double newlines to preserve paragraph structure.
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF is required for PDF extraction. Install with: pip install pymupdf")

    doc = fitz.open(file_path)
    pages_text = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # "text" mode preserves layout better than "raw"
        if text.strip():
            pages_text.append(text.strip())

    doc.close()

    if not pages_text:
        raise ValueError("No readable text found in the PDF. It may be image-based; try uploading as PNG/JPG.")

    return "\n\n".join(pages_text)


def extract_text_from_image(file_path: str) -> str:
    """
    Extracts text from an image file using Tesseract OCR.
    Applies basic preprocessing via Pillow to improve OCR accuracy.
    """
    try:
        import pytesseract
        from PIL import Image, ImageFilter, ImageOps
    except ImportError:
        raise ImportError(
            "pytesseract and Pillow are required for image OCR. "
            "Install with: pip install pytesseract pillow\n"
            "Also ensure Tesseract is installed on your OS."
        )

    image = Image.open(file_path)

    # Convert to grayscale – helps OCR accuracy
    image = ImageOps.grayscale(image)

    # Light sharpening pass
    image = image.filter(ImageFilter.SHARPEN)

    # Run OCR
    text = pytesseract.image_to_string(image, lang="eng")

    if not text.strip():
        raise ValueError("No text could be extracted from the image. Ensure the image contains readable text.")

    return text.strip()


def clean_text(raw_text: str) -> str:
    """
    Post-processes extracted text:
      - Collapses multiple blank lines into at most two
      - Removes stray non-printable characters
      - Normalises whitespace within lines
    """
    # Remove non-printable characters (except newlines and tabs)
    text = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\u00A0-\uFFFF]", "", raw_text)

    # Normalise intra-line whitespace
    lines = [" ".join(line.split()) for line in text.splitlines()]

    # Collapse runs of blank lines
    cleaned_lines: list[str] = []
    blank_count = 0
    for line in lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                cleaned_lines.append(line)
        else:
            blank_count = 0
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def extract_text(file_path: str) -> Tuple[str, str | None]:
    """
    Public interface: validates and extracts text from a PDF or image file.

    Returns:
        (extracted_text, None)        on success
        ("",            error_message) on failure
    """
    valid, error = validate_file(file_path)
    if not valid:
        return "", error

    path = Path(file_path)
    ext = path.suffix.lower()

    try:
        if ext in PDF_EXTENSIONS:
            raw_text = extract_text_from_pdf(file_path)
        else:
            raw_text = extract_text_from_image(file_path)

        text = clean_text(raw_text)
        return text, None

    except (ImportError, ValueError, OSError, RuntimeError) as exc:
        return "", str(exc)
