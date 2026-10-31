"""
Unit Tests for FastAPI REST API Endpoints
"""

import unittest
from fastapi.testclient import TestClient
from api import app
from analyzer.sample_resumes import SAMPLE_RESUMES

class TestResumeScorerAPI(unittest.TestCase):

    def setUp(self):
        self.client = TestClient(app)
        self.sample_text = SAMPLE_RESUMES[list(SAMPLE_RESUMES.keys())[0]]

    def test_health_check_endpoint(self):
        """Test /api/health endpoint status and metadata."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertGreaterEqual(data["total_roles_benchmarked"], 5)
        self.assertGreaterEqual(data["preloaded_samples_count"], 1)

    def test_roles_endpoint(self):
        """Test /api/roles returns configured benchmarks."""
        response = self.client.get("/api/roles")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("roles", data)
        self.assertGreater(len(data["roles"]), 5)

    def test_samples_endpoint(self):
        """Test /api/samples returns sample resume list."""
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("samples", data)
        self.assertGreater(len(data["samples"]), 0)

    def test_privacy_policy_endpoint(self):
        """Test /api/privacy-policy returns zero-retention policy."""
        response = self.client.get("/api/privacy-policy")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("policy", data)
        self.assertEqual(data["max_upload_size_bytes"], 5 * 1024 * 1024)

    def test_rubric_validation_endpoint(self):
        """Test /api/rubric-validation returns statistical correlation data."""
        response = self.client.get("/api/rubric-validation")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "validated")
        self.assertGreaterEqual(data["metrics"]["pearson_correlation_r"], 0.85)

    def test_redact_pii_endpoint(self):
        """Test /api/redact-pii scrubs candidate contact data."""
        payload = {"text": "Alex Hunter | alex.hunter@example.com | +1-415-555-0199 | Python engineer"}
        response = self.client.post("/api/redact-pii", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn("alex.hunter@example.com", data["redacted_text"])
        self.assertIn("[EMAIL_REDACTED]", data["redacted_text"])

    def test_score_text_endpoint_success(self):
        """Test /api/score returns scoring breakdown, explanations, and rubric disclaimer."""
        payload = {
            "resume_text": self.sample_text,
            "target_role": "🤖 Generative AI / LLM Engineer"
        }
        response = self.client.post("/api/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("composite_score", data)
        self.assertIn("scores_breakdown", data)
        self.assertIn("explanations", data)
        self.assertIn("rubric_disclaimer", data)
        self.assertGreaterEqual(data["composite_score"], 60)

    def test_batch_score_endpoint(self):
        """Test /api/batch-score leaderboard calculation."""
        payload = {
            "candidates": [
                {"id": "cand_1", "resume_text": self.sample_text},
                {"id": "cand_2", "resume_text": "Junior Python developer with basic Git knowledge."}
            ],
            "target_role": "🤖 Generative AI / LLM Engineer"
        }
        response = self.client.post("/api/batch-score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["total_processed"], 2)
        self.assertEqual(len(data["leaderboard"]), 2)
        self.assertEqual(data["leaderboard"][0]["rank"], 1)
        self.assertGreater(data["leaderboard"][0]["composite_score"], data["leaderboard"][1]["composite_score"])

    def test_upload_and_score_valid_txt(self):
        """Test /api/upload-and-score with valid text file."""
        file_content = self.sample_text.encode("utf-8")
        response = self.client.post(
            "/api/upload-and-score",
            files={"file": ("candidate_resume.txt", file_content, "text/plain")},
            data={"target_role": "🤖 Generative AI / LLM Engineer"}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("composite_score", data)
        self.assertEqual(data["source_filename"], "candidate_resume.txt")

    def test_upload_and_score_invalid_extension(self):
        """Test /api/upload-and-score rejects invalid file types."""
        response = self.client.post(
            "/api/upload-and-score",
            files={"file": ("virus.exe", b"malicious content", "application/octet-stream")}
        )
        self.assertEqual(response.status_code, 400)
        self.assertIn("Unsupported file extension", response.json()["detail"])

    def test_markdown_report_endpoint(self):
        """Test /api/report/markdown endpoint."""
        payload = {
            "resume_text": self.sample_text,
            "target_role": "🤖 Generative AI / LLM Engineer"
        }
        response = self.client.post("/api/report/markdown", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("markdown", data)
        self.assertIn("# 📄 AI/ML Resume Scorer", data["markdown"])

    def test_pdf_report_endpoint(self):
        """Test /api/report/pdf returns binary PDF."""
        payload = {
            "resume_text": self.sample_text,
            "target_role": "🤖 Generative AI / LLM Engineer"
        }
        response = self.client.post("/api/report/pdf", json=payload)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "application/pdf")
        self.assertGreater(len(response.content), 1000)

if __name__ == "__main__":
    unittest.main()
