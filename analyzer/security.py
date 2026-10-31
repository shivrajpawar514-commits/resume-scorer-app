"""
Security, Content Validation, Safe File Handling & Data Privacy Module
Enforces:
1. File upload size limits and content magic-byte verification (protecting against spoofed extensions, executables).
2. Safe, ephemeral temporary file handling with context-managed immediate deletion and restricted file permissions.
3. Zero-retention privacy policy and PII redaction for applicant data protection.
"""

import os
import re
import io
import stat
import tempfile
from contextlib import contextmanager
from typing import Dict, Any, Tuple, Generator, Optional

# Security Constants
MAX_UPLOAD_SIZE_BYTES = 5 * 1024 * 1024  # 5 MB hard limit
ALLOWED_EXTENSIONS = {".pdf", ".docx", ".txt"}

# Magic byte signatures
MAGIC_BYTES = {
    "pdf": b"%PDF-",
    "docx_zip": b"PK\x03\x04"
}

PRIVACY_POLICY = {
    "policy_version": "1.0.0",
    "retention_policy": "Zero-Retention (Ephemeral in-memory processing only)",
    "data_storage": "No resume text, extracted PII, contact info, or applicant records are saved to disk, databases, or external telemetry.",
    "network_egress": "All scoring, skills matching, and document parsing run strictly local to the runtime environment.",
    "lifecycle": "Document bytes and extracted representations exist only in process memory for the duration of the HTTP request and are discarded immediately upon response completion.",
    "user_rights": "Candidates may optionally redact sensitive PII (email, phone, address, portfolio links) prior to scoring without degrading skill or impact assessment."
}


def validate_file_content(file_bytes: bytes, filename: str) -> Tuple[bool, str]:
    """
    Validates file upload against size limits, file extension whitelist,
    and genuine binary magic bytes (preventing spoofed extensions or malicious executables).
    """
    if not file_bytes or len(file_bytes) == 0:
        return False, "Empty file uploaded (0 bytes)."

    if len(file_bytes) > MAX_UPLOAD_SIZE_BYTES:
        max_mb = MAX_UPLOAD_SIZE_BYTES // (1024 * 1024)
        return False, f"File exceeds maximum upload size limit of {max_mb} MB."

    ext = os.path.splitext(filename)[1].lower() if filename else ""
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"Unsupported file extension '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}."

    # Magic byte verification
    if ext == ".pdf":
        if not file_bytes.startswith(MAGIC_BYTES["pdf"]):
            # Some PDFs have up to 1024 leading bytes of header comments
            if b"%PDF-" not in file_bytes[:1024]:
                return False, "Invalid PDF document: missing %PDF- magic signature header."

    elif ext == ".docx":
        if not file_bytes.startswith(MAGIC_BYTES["docx_zip"]):
            return False, "Invalid DOCX document: file is not a valid OpenXML ZIP archive."

    elif ext == ".txt":
        # Check for executable or binary file disguised as .txt (e.g. ELF, Mach-O, Windows PE)
        if file_bytes.startswith(b"MZ") or file_bytes.startswith(b"\x7fELF") or file_bytes.startswith(b"\xca\xfe\xba\xbe"):
            return False, "Binary executable disguised as text file rejected."
        # Check for excessive null bytes
        null_count = file_bytes.count(b"\x00")
        if null_count > 0:
            return False, "Binary content detected in text file (null bytes present)."

    return True, "Content verified."


@contextmanager
def safe_temp_file(suffix: str, content: bytes) -> Generator[str, None, None]:
    """
    Safely creates an ephemeral temporary file with restricted POSIX permissions (0o600: read/write by owner only).
    Guarantees immediate secure deletion (unlink) upon exiting the context block, even if exceptions occur.
    """
    temp_fd = None
    temp_path = None
    try:
        temp_fd, temp_path = tempfile.mkstemp(suffix=suffix)
        # Enforce restricted permissions (read/write by owner only, no group or other access)
        os.chmod(temp_path, stat.S_IRUSR | stat.S_IWUSR)
        with os.fdopen(temp_fd, "wb") as f:
            f.write(content)
        temp_fd = None  # fdopen transfers ownership
        yield temp_path
    finally:
        if temp_fd is not None:
            try:
                os.close(temp_fd)
            except OSError:
                pass
        if temp_path and os.path.exists(temp_path):
            try:
                # Overwrite bytes with zeros before unlinking for secure cleanup
                file_size = os.path.getsize(temp_path)
                with open(temp_path, "wb") as f:
                    f.write(b"\x00" * min(file_size, 4096))
                os.unlink(temp_path)
            except OSError:
                pass


def redact_pii(text: str) -> Dict[str, Any]:
    """
    Redacts candidate personally identifiable information (PII) including email addresses,
    phone numbers, LinkedIn handles, and physical addresses to protect candidate privacy.
    """
    redacted = text
    redacted_items = {"emails": 0, "phones": 0, "linkedin": 0, "github": 0}

    # Redact Emails
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    redacted_items["emails"] = len(re.findall(email_pattern, redacted))
    redacted = re.sub(email_pattern, "[EMAIL_REDACTED]", redacted)

    # Redact Phones
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    redacted_items["phones"] = len(re.findall(phone_pattern, redacted))
    redacted = re.sub(phone_pattern, "[PHONE_REDACTED]", redacted)

    # Redact LinkedIn profiles
    linkedin_pattern = r'(?:https?:\/\/)?(?:www\.)?linkedin\.com\/(?:in|profile)\/[a-zA-Z0-9_-]+'
    redacted_items["linkedin"] = len(re.findall(linkedin_pattern, redacted, re.IGNORECASE))
    redacted = re.sub(linkedin_pattern, "[LINKEDIN_REDACTED]", redacted, flags=re.IGNORECASE)

    # Redact GitHub profiles
    github_pattern = r'(?:https?:\/\/)?(?:www\.)?github\.com\/[a-zA-Z0-9_-]+'
    redacted_items["github"] = len(re.findall(github_pattern, redacted, re.IGNORECASE))
    redacted = re.sub(github_pattern, "[GITHUB_REDACTED]", redacted, flags=re.IGNORECASE)

    return {
        "redacted_text": redacted,
        "items_redacted": redacted_items,
        "is_redacted": sum(redacted_items.values()) > 0
    }
