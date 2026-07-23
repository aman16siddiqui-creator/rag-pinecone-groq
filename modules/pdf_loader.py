from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import List

import pdfplumber
import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from config import settings

if settings.tesseract_cmd:
    pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


@dataclass
class PageContent:
    page_number: int          # 1-indexed
    text: str
    source_doc: str
    used_ocr: bool


class PDFExtractionError(Exception):
    """Raised when a PDF cannot be opened or processed at all."""


def _clean_text(raw: str) -> str:
    """Remove common PDF extraction artifacts: hyphenation across line
    breaks, repeated whitespace, stray form-feed / control characters,
    and duplicated running headers/footers on a single page."""
    if not raw:
        return ""

    text = raw.replace("\x0c", " ")
    # Fix hyphenated line-wraps: "histo-\npathology" -> "histopathology"
    text = re.sub(r"(\w+)-\n(\w+)", r"\1\2", text)
    # Collapse remaining newlines into spaces (paragraph structure isn't
    # critical for chunk/embedding purposes; we keep page-level granularity)
    text = re.sub(r"\s*\n\s*", " ", text)
    # Collapse multiple spaces/tabs
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _ocr_page_image(pil_image: Image.Image) -> str:
    try:
        return pytesseract.image_to_string(pil_image)
    except Exception as exc:  # pragma: no cover - defensive
        # OCR failures shouldn't crash the whole pipeline; log & continue.
        print(f"[pdf_loader] OCR failed on a page: {exc}")
        return ""


def _ocr_embedded_images(page: "pdfplumber.page.Page") -> str:
    """OCR any raster images embedded within an otherwise-text page
    (e.g. a microscope image pasted next to a paragraph)."""
    captions = []
    try:
        page_image = page.to_image(resolution=200).original
    except Exception:
        return ""

    for img in page.images:
        try:
            x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
            cropped = page_image.crop((x0, top, x1, bottom))
            if cropped.size[0] < 20 or cropped.size[1] < 20:
                continue  # skip tiny decorative icons
            ocr_text = _ocr_page_image(cropped)
            if ocr_text.strip():
                captions.append(ocr_text.strip())
        except Exception:
            continue
    return " ".join(captions)


def extract_text_from_pdf(file_path: str, doc_name: str) -> List[PageContent]:
    """
    Extract text page-by-page from a PDF, falling back to OCR (Tesseract)
    for image-only pages and for embedded raster images (micrographs,
    scanned figures, screenshots common in medical / histopathology PDFs).

    Returns a list of PageContent, one entry per page, in page order.
    """
    pages: List[PageContent] = []

    try:
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                native_text = page.extract_text() or ""
                native_text_clean = _clean_text(native_text)

                used_ocr = False
                combined_text = native_text_clean

                # Case 1: page is essentially image-only -> full-page OCR
                if len(native_text_clean) < settings.ocr_min_chars_per_page:
                    try:
                        images = convert_from_path(
                            file_path, first_page=i, last_page=i, dpi=250
                        )
                        if images:
                            ocr_text = _clean_text(_ocr_page_image(images[0]))
                            combined_text = (combined_text + " " + ocr_text).strip()
                            used_ocr = True
                    except Exception as exc:  # pragma: no cover
                        print(f"[pdf_loader] Full-page OCR failed on page {i}: {exc}")

                # Case 2: text page that also contains embedded images
                # (e.g. a captioned histology slide) -> OCR just the images
                elif page.images:
                    embedded_text = _clean_text(_ocr_embedded_images(page))
                    if embedded_text:
                        combined_text = (combined_text + " " + embedded_text).strip()
                        used_ocr = True

                pages.append(
                    PageContent(
                        page_number=i,
                        text=combined_text,
                        source_doc=doc_name,
                        used_ocr=used_ocr,
                    )
                )
    except Exception as exc:
        raise PDFExtractionError(f"Could not process '{doc_name}': {exc}") from exc

    return pages
