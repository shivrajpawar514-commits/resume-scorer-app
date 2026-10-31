"""
Unit Tests for Security, Content Validation, Safe File Handling & Data Privacy
Verifies:
- File upload size limits (5 MB rejection)
- Content validation via magic bytes (PDF header, DOCX zip, binary executable detection)
- Safe ephemeral temporary file handling (0o600 permissions, guaranteed immediate unlinking)
- Zero-retention privacy policy and PII redaction
- Strict CORS configuration (trusted origins only, no credentialed wildcard)
"""

import unittest
import os
import stat
from fastapi.testclient import TestClient

from api import app, trusted_origins
from analyzer.security import (
    MAX_UPLOAD_SIZE_BYTES,
    validate_file_content,
    safe_temp_file,
    redact_pii,
    PRIVACY_POLICY
)

class TestSecurityAndPrivacy(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)

    def test_upload_size_limit_rejection(self):
        """Verify files exceeding 5MB are rejected by content validation and API."""
        oversized_bytes = b"0" * (MAX_UPLOAD_SIZE_BYTES + 1024)
        is_valid, msg = validate_file_content(oversized_bytes, "oversized_resume.pdf")
        self.assertFalse(is_valid)
        self.assertIn("exceeds maximum upload size", msg)

        # Test API endpoint rejection (HTTP 413)
        response = self.client.post(
            "/api/upload-and-score",
            files={"file": ("large.pdf", oversized_bytes, "application/pdf")}
        )
        self.assertEqual(response.status_code, 413)
        self.assertIn("exceeds maximum upload size", response.json()["detail"])

    def test_magic_byte_content_validation(self):
        """Verify magic byte verification prevents spoofed file extensions and disguised executables."""
        # 1. Fake PDF (text file renamed as .pdf without %PDF- magic bytes)
        fake_pdf = b"This is just plain text, not a real PDF document."
        is_valid, msg = validate_file_content(fake_pdf, "fake.pdf")
        self.assertFalse(is_valid)
        self.assertIn("missing %PDF- magic signature", msg)

        # 2. Genuine PDF header
        real_pdf_header = b"%PDF-1.4\n1 0 obj\n<<>>\nendobj\ntrailer\n<<>>\n%%EOF"
        is_valid, msg = validate_file_content(real_pdf_header, "real.pdf")
        self.assertTrue(is_valid)

        # 3. Disguised Windows PE executable renamed as .txt
        fake_txt_pe = b"MZ\x90\x00\x03\x00\x00\x00" + b"\x00" * 50
        is_valid, msg = validate_file_content(fake_txt_pe, "malware.txt")
        self.assertFalse(is_valid)
        self.assertIn("executable", msg.lower())

        # 4. Disguised Linux ELF binary renamed as .txt
        fake_txt_elf = b"\x7fELF\x02\x01\x01\x00" + b"\x00" * 50
        is_valid, msg = validate_file_content(fake_txt_elf, "malware.txt")
        self.assertFalse(is_valid)
        self.assertIn("executable", msg.lower())

        # 5. Unsupported extension
        is_valid, msg = validate_file_content(b"console.log('hi')", "script.js")
        self.assertFalse(is_valid)
        self.assertIn("Unsupported file extension", msg)

    def test_safe_temporary_file_handling(self):
        """Verify temporary files enforce restricted permissions (0o600) and are deleted immediately."""
        content = b"Candidate highly sensitive resume bytes"
        temp_file_recorded = None

        with safe_temp_file(suffix=".pdf", content=content) as temp_path:
            temp_file_recorded = temp_path
            self.assertTrue(os.path.exists(temp_path))
            
            # Check POSIX file permissions (must be readable/writable by owner only: 0o600)
            mode = stat.S_IMODE(os.stat(temp_path).st_mode)
            self.assertEqual(mode, stat.S_IRUSR | stat.S_IWUSR)

            # Check content was written
            with open(temp_path, "rb") as f:
                self.assertEqual(f.read(), content)

        # Once exited context block, file must be unlinked/deleted
        self.assertIsNotNone(temp_file_recorded)
        self.assertFalse(os.path.exists(temp_file_recorded), "Temporary file was not deleted upon exit!")

    def test_pii_redaction(self):
        """Verify candidate contact PII is cleanly scrubbed while preserving technical content."""
        raw_text = """
        John Doe
        john.doe@enterprise.com | +1-415-555-0199 | linkedin.com/in/johndoe | github.com/johndoe-ai
        Senior Machine Learning Engineer with PyTorch experience.
        """
        redaction_res = redact_pii(raw_text)
        redacted = redaction_res["redacted_text"]

        self.assertNotIn("john.doe@enterprise.com", redacted)
        self.assertNotIn("+1-415-555-0199", redacted)
        self.assertNotIn("linkedin.com/in/johndoe", redacted)
        self.assertNotIn("github.com/johndoe-ai", redacted)

        self.assertIn("[EMAIL_REDACTED]", redacted)
        self.assertIn("[PHONE_REDACTED]", redacted)
        self.assertIn("Senior Machine Learning Engineer with PyTorch experience.", redacted)

    def test_privacy_policy_endpoint(self):
        """Verify privacy policy endpoint publishes zero-retention data guarantees."""
        response = self.client.get("/api/privacy-policy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("policy", data)
        self.assertIn("Zero-Retention", data["policy"]["retention_policy"])
        self.assertIn("No resume text", data["policy"]["data_storage"])

    def test_cors_restriction(self):
        """Verify CORS is restricted to trusted origins and forbids credentialed wildcards."""
        self.assertNotIn("*", trusted_origins, "CORS allow_origins must not contain wildcard '*' when credentials are true!")
        self.assertGreater(len(trusted_origins), 0)
        
        # Test OPTIONS preflight request from a trusted origin
        response = self.client.options(
            "/api/score",
            headers={
                "Origin": "http://localhost:8501",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:8501")
        self.assertEqual(response.headers.get("access-control-allow-credentials"), "true")

        # Test request from an untrusted external origin
        untrusted_response = self.client.options(
            "/api/score",
            headers={
                "Origin": "https://malicious-third-party.attacker.com",
                "Access-Control-Request-Method": "POST"
            }
        )
        # Untrusted origin must not be allowed
        self.assertNotEqual(
            untrusted_response.headers.get("access-control-allow-origin"),
            "https://malicious-third-party.attacker.com"
        )

if __name__ == "__main__":
    unittest.main()
