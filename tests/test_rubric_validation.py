"""
Unit Tests for Scoring Rubric Validation against Human-Reviewed Evaluation Set
Verifies:
- Statistical correlation between heuristic composite scores and human ground-truth evaluations (Pearson r > 0.85).
- Spearman rank correlation (rho > 0.85) ensuring accurate candidate ranking.
- Mean Absolute Error (MAE) calibration bounds.
- Concordance across Senior, Mid-level, Transitioner, and Low-match candidate profiles.
"""

import unittest
from analyzer.evaluation_benchmark import (
    HUMAN_EVALUATION_DATASET,
    run_rubric_validation
)

class TestRubricValidation(unittest.TestCase):

    def test_human_evaluation_dataset_integrity(self):
        """Verify the curated evaluation dataset is well-formed with diverse profiles."""
        self.assertGreaterEqual(len(HUMAN_EVALUATION_DATASET), 5)
        for item in HUMAN_EVALUATION_DATASET:
            self.assertIn("id", item)
            self.assertIn("target_role", item)
            self.assertIn("human_ground_truth_score", item)
            self.assertIn("human_verdict", item)
            self.assertIn("resume_text", item)
            self.assertGreater(len(item["resume_text"].strip()), 50)
            self.assertGreaterEqual(item["human_ground_truth_score"], 0.0)
            self.assertLessEqual(item["human_ground_truth_score"], 100.0)

    def test_rubric_statistical_soundness(self):
        """Verify rubric achieves high statistical correlation with human expert evaluations."""
        report = run_rubric_validation()
        m = report["metrics"]

        # Pearson correlation must be strong (r >= 0.85)
        self.assertGreaterEqual(
            m["pearson_correlation_r"], 0.85,
            f"Pearson correlation r={m['pearson_correlation_r']} fell below 0.85 threshold"
        )
        self.assertLess(m["pearson_p_value"], 0.01)

        # Spearman rank correlation must be strong (rho >= 0.85)
        self.assertGreaterEqual(
            m["spearman_rank_correlation_rho"], 0.85,
            f"Spearman rank correlation rho={m['spearman_rank_correlation_rho']} fell below 0.85 threshold"
        )

        # Mean Absolute Error should be reasonably bounded (MAE <= 25 points)
        self.assertLessEqual(
            m["mean_absolute_error_mae"], 25.0,
            f"MAE={m['mean_absolute_error_mae']} exceeded 25.0 points"
        )

        # Tier concordance rate
        self.assertGreaterEqual(m["tier_concordance_rate_pct"], 50.0)

    def test_candidate_relative_ranking(self):
        """Verify senior top-tier candidates score significantly higher than unrelated roles."""
        report = run_rubric_validation()
        evals = {c["id"]: c["system_composite_score"] for c in report["candidate_evaluations"]}

        # Senior GenAI must score higher than Junior Data Scientist
        self.assertGreater(evals["eval_01_senior_genai"], evals["eval_04_junior_ds"])

        # Junior Data Scientist must score higher than Unrelated Frontend Developer
        self.assertGreater(evals["eval_04_junior_ds"], evals["eval_06_unrelated_role"])

        # Unrelated Frontend must score higher than minimal empty text
        self.assertGreater(evals["eval_06_unrelated_role"], evals["eval_07_minimal_unformatted"])

if __name__ == "__main__":
    unittest.main()
