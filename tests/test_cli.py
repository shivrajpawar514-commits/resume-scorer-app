"""
Unit Tests for CLI Execution and File Exports
"""

import unittest
import tempfile
import os
import csv
import json
from cli import export_batch_to_csv, evaluate_single_file
from analyzer.scoring_engine import score_resume
from analyzer.sample_resumes import SAMPLE_RESUMES

class TestResumeScorerCLI(unittest.TestCase):

    def setUp(self):
        self.sample_text = SAMPLE_RESUMES[list(SAMPLE_RESUMES.keys())[0]]
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_evaluate_single_file_txt(self):
        """Test file extraction and scoring on a temporary txt resume."""
        resume_file = os.path.join(self.temp_dir.name, "candidate.txt")
        with open(resume_file, "w", encoding="utf-8") as f:
            f.write(self.sample_text)

        results = evaluate_single_file(resume_file, role_name="🤖 Generative AI / LLM Engineer")
        self.assertIn("composite_score", results)
        self.assertGreaterEqual(results["composite_score"], 60)

    def test_export_batch_to_csv(self):
        """Test batch results export to CSV format."""
        csv_file = os.path.join(self.temp_dir.name, "leaderboard.csv")
        res1 = score_resume(self.sample_text, target_role_name="🤖 Generative AI / LLM Engineer")
        res2 = score_resume("Python and SQL developer", target_role_name="🤖 Generative AI / LLM Engineer")
        
        batch_results = [
            {"filename": "resume_top.pdf", "results": res1},
            {"filename": "resume_junior.pdf", "results": res2}
        ]

        export_batch_to_csv(batch_results, csv_file)
        self.assertTrue(os.path.exists(csv_file))

        with open(csv_file, "r", encoding="utf-8") as f:
            reader = list(csv.DictReader(f))
            self.assertEqual(len(reader), 2)
            self.assertEqual(reader[0]["Rank"], "1")
            self.assertEqual(reader[0]["Filename"], "resume_top.pdf")
            self.assertIn("Hard Skills Score", reader[0])
            self.assertIn("Composite Score (%)", reader[0])

if __name__ == "__main__":
    unittest.main()
