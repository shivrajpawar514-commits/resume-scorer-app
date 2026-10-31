"""
Document Text and Contact Information Extractor
Supports PDF (Multi-column aware, Scanned page detection, Ligature/Font normalization),
DOCX, and TXT files with comprehensive edge-case handling.
"""

import re
import io
import unicodedata
from typing import Dict, Any, Optional, List, Tuple

# Common typographical ligatures to normalize for robust ATS parsing
LIGATURE_MAP = {
    "\ufb00": "ff",
    "\ufb01": "fi",
    "\ufb02": "fl",
    "\ufb03": "ffi",
    "\ufb04": "ffl",
    "\ufb05": "ft",
    "\ufb06": "st",
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "–": "-",
    "—": "-",
    "•": "*",
    "‣": "*",
    "⁃": "*",
    "○": "*",
    "■": "*",
    "▪": "*",
    "\u00a0": " "
}

class DocumentExtractionError(Exception):
    """Custom exception raised when document extraction fails due to corruption or format issues."""
    pass


def normalize_pdf_typography(text: str) -> str:
    """Replaces typographical ligatures, smart quotes, and normalizes unicode."""
    if not text:
        return ""
    for ligature, replacement in LIGATURE_MAP.items():
        if ligature in text:
            text = text.replace(ligature, replacement)
    # NFKD decomposition normalizes embedded accented characters and font variants
    normalized = unicodedata.normalize("NFKD", text)
    return normalized


def extract_text_from_pdf_with_diagnostics(file_bytes: bytes) -> Dict[str, Any]:
    """
    Extracts text from PDF bytes with multi-column awareness, scanned document detection,
    and resilience against corrupted documents, unusual fonts, and empty files.
    """
    diagnostics = {
        "text": "",
        "page_count": 0,
        "char_count": 0,
        "is_scanned": False,
        "is_multi_column": False,
        "is_empty": False,
        "warnings": [],
        "extraction_method": "none",
        "error": None
    }

    if not file_bytes or len(file_bytes) == 0:
        diagnostics["is_empty"] = True
        diagnostics["error"] = "Empty PDF file (0 bytes)."
        return diagnostics

    # Method 1: PyMuPDF (fitz) with layout and block sorting
    try:
        import fitz  # PyMuPDF
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except (fitz.FileDataError, fitz.EmptyFileError) as e:
            diagnostics["error"] = f"Corrupted or invalid PDF file: {str(e)}"
            return diagnostics
        except Exception as e:
            if "encrypted" in str(e).lower() or "password" in str(e).lower():
                diagnostics["error"] = "Password-protected or encrypted PDF cannot be parsed."
                return diagnostics
            raise e

        if doc.is_encrypted:
            diagnostics["error"] = "Password-protected or encrypted PDF cannot be parsed."
            doc.close()
            return diagnostics

        diagnostics["page_count"] = len(doc)
        if len(doc) == 0:
            diagnostics["is_empty"] = True
            diagnostics["error"] = "PDF document contains 0 pages."
            doc.close()
            return diagnostics

        all_pages_text = []
        has_images_only = True
        multi_col_detected = False

        for page_idx, page in enumerate(doc):
            page_rect = page.rect
            page_width = page_rect.width
            blocks = page.get_text("blocks")
            images = page.get_images()

            # Separate text blocks (type 0)
            text_blocks = [b for b in blocks if b[6] == 0 and b[4].strip()]
            
            # Check for multi-column layout on this page
            # Check if there are distinct columns: left (x1 < 0.48*width) and right (x0 > 0.42*width)
            left_blocks = [b for b in text_blocks if b[2] <= page_width * 0.48]
            right_blocks = [b for b in text_blocks if b[0] >= page_width * 0.42]

            page_text = ""
            if len(left_blocks) >= 2 and len(right_blocks) >= 2:
                multi_col_detected = True
                # Sort left column top-to-bottom, then right column top-to-bottom
                left_blocks.sort(key=lambda b: (b[1], b[0]))
                right_blocks.sort(key=lambda b: (b[1], b[0]))
                
                # Any header spanning the full width
                header_blocks = [b for b in text_blocks if b not in left_blocks and b not in right_blocks and b[1] < page_rect.height * 0.25]
                header_blocks.sort(key=lambda b: b[1])

                for b in header_blocks:
                    page_text += b[4] + "\n"
                for b in left_blocks:
                    page_text += b[4] + "\n"
                for b in right_blocks:
                    page_text += b[4] + "\n"
            else:
                # Standard single column: sort by vertical y0 first, then x0
                text_blocks.sort(key=lambda b: (b[1], b[0]))
                for b in text_blocks:
                    page_text += b[4] + "\n"

            cleaned_page = clean_extracted_text(normalize_pdf_typography(page_text))
            if cleaned_page:
                has_images_only = False
                all_pages_text.append(cleaned_page)
            elif len(images) > 0:
                diagnostics["warnings"].append(f"Page {page_idx + 1} contains images with no selectable text layer.")

        doc.close()

        combined_text = "\n\n".join(all_pages_text).strip()
        if combined_text:
            diagnostics["text"] = combined_text
            diagnostics["char_count"] = len(combined_text)
            diagnostics["is_multi_column"] = multi_col_detected
            diagnostics["extraction_method"] = "pymupdf_blocks_sorted"
            if multi_col_detected:
                diagnostics["warnings"].append("Multi-column layout detected; extracted in preserved reading order.")
            return diagnostics
        elif has_images_only:
            diagnostics["is_scanned"] = True
            diagnostics["warnings"].append(
                "Scanned or image-only PDF detected. The document contains bitmap images without an embedded text layer. "
                "Applicant Tracking Systems (ATS) cannot parse image-only resumes without OCR."
            )
            return diagnostics

    except Exception as e:
        diagnostics["warnings"].append(f"PyMuPDF block extraction fallback: {str(e)}")

    # Method 2: pdfminer.six fallback
    try:
        from pdfminer.high_level import extract_text
        text = extract_text(io.BytesIO(file_bytes))
        clean_text = clean_extracted_text(normalize_pdf_typography(text))
        if clean_text:
            diagnostics["text"] = clean_text
            diagnostics["char_count"] = len(clean_text)
            diagnostics["extraction_method"] = "pdfminer.six"
            return diagnostics
    except Exception as e:
        diagnostics["warnings"].append(f"pdfminer fallback failed: {str(e)}")

    # Method 3: PyPDF2 fallback
    try:
        import PyPDF2
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages_text = []
        for p in pdf_reader.pages:
            t = p.extract_text()
            if t:
                pages_text.append(t)
        combined = clean_extracted_text(normalize_pdf_typography("\n".join(pages_text)))
        if combined:
            diagnostics["text"] = combined
            diagnostics["char_count"] = len(combined)
            diagnostics["extraction_method"] = "PyPDF2"
            return diagnostics
    except Exception as e:
        diagnostics["warnings"].append(f"PyPDF2 fallback failed: {str(e)}")

    if not diagnostics["text"]:
        diagnostics["is_empty"] = True
        diagnostics["error"] = "No readable text could be extracted from the PDF document."

    return diagnostics


def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract clean text from PDF bytes using robust multi-column aware parser."""
    diag = extract_text_from_pdf_with_diagnostics(file_bytes)
    return diag["text"]


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract clean text from DOCX bytes including paragraphs and table cells."""
    if not file_bytes:
        return ""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        for table in doc.tables:
            for row in table.rows:
                row_text = [cell.text.strip() for cell in row.cells if cell.text.strip()]
                if row_text:
                    paragraphs.append(" | ".join(row_text))
        return clean_extracted_text(normalize_pdf_typography("\n".join(paragraphs)))
    except Exception:
        return ""


def extract_text_from_file(uploaded_file) -> str:
    """Extract text based on file type with graceful error recovery."""
    if uploaded_file is None:
        return ""
    
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    filename = getattr(uploaded_file, "name", "").lower()

    if filename.endswith(".pdf"):
        return extract_text_from_pdf(file_bytes)
    elif filename.endswith(".docx"):
        return extract_text_from_docx(file_bytes)
    elif filename.endswith(".txt"):
        try:
            return clean_extracted_text(file_bytes.decode("utf-8", errors="ignore"))
        except Exception:
            return ""
    else:
        # Check magic bytes to determine format
        if file_bytes.startswith(b"%PDF-"):
            return extract_text_from_pdf(file_bytes)
        elif file_bytes.startswith(b"PK\x03\x04"):
            return extract_text_from_docx(file_bytes)
        else:
            try:
                return clean_extracted_text(file_bytes.decode("utf-8", errors="ignore"))
            except Exception:
                return ""


def clean_extracted_text(text: str) -> str:
    """Normalize whitespace, unicode artifacts, and multi-line breaks."""
    if not text:
        return ""
    # Replace control characters
    text = re.sub(r'[\r\f\v]', '\n', text)
    text = re.sub(r'[\t\xa0]', ' ', text)
    # Remove multiple horizontal spaces
    text = re.sub(r'[ ]{2,}', ' ', text)
    # Remove excessive blank lines
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def extract_contact_info(text: str) -> Dict[str, Any]:
    """Extract emails, phones, LinkedIn, GitHub, and portfolio URLs."""
    info = {
        "email": None,
        "phone": None,
        "linkedin": None,
        "github": None,
        "portfolio": None
    }
    if not text:
        return info

    # Email regex
    email_match = re.search(r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+', text)
    if email_match:
        info["email"] = email_match.group(0)

    # Phone regex
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}', text)
    if phone_match:
        info["phone"] = phone_match.group(0)

    # LinkedIn regex
    linkedin_match = re.search(r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|profile)\/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    if linkedin_match:
        info["linkedin"] = linkedin_match.group(0)

    # GitHub regex
    github_match = re.search(r'(?:https?:\/\/)?(?:www\.)?github\.com\/([a-zA-Z0-9_-]+)', text, re.IGNORECASE)
    if github_match:
        info["github"] = github_match.group(0)

    # Portfolio / Web regex
    portfolio_match = re.search(r'(?:https?:\/\/)?(?:www\.)?([a-zA-Z0-9-]+\.(?:io|me|dev|app|ai|com))(?:\/[^\s]*)?', text, re.IGNORECASE)
    if portfolio_match:
        url = portfolio_match.group(0)
        if "linkedin.com" not in url and "github.com" not in url:
            info["portfolio"] = url

    return info
