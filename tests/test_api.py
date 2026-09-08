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

    def test_score_text_endpoint_success(self):
        """Test /api/score returns scoring breakdown."""
        payload = {
            "resume_text": self.sample_text,
            "target_role": "🤖 Generative AI / LLM Engineer"
        }
        response = self.client.post("/api/score", json=payload)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("composite_score", data)
        self.assertIn("scores_breakdown", data)
        self.assertIn("radar_dimensions", data)
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
