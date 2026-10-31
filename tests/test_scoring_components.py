"""
Unit Tests for Scoring Components & Defensible Explanations
Verifies:
- Every scoring component (Hard Skills, Dense Embeddings, TF-IDF Lexical, Impact, MLOps, ATS Health, Composite)
- Defensible reasoning for why each score was assigned
- Accurate lexical vs semantic distinction
- Rubric disclaimer presence
"""

import unittest
from analyzer.skills_taxonomy import ROLE_BENCHMARKS
from analyzer.sample_resumes import SAMPLE_RESUMES
from analyzer.scoring_engine import (
    score_resume,
    compute_tfidf_similarity,
    calculate_ats_readability,
    find_benchmark_role,
    RUBRIC_DISCLAIMER
)
from analyzer.semantic_embeddings import (
    compute_dense_embedding_similarity,
    compute_lexical_tfidf_similarity
)

class TestScoringComponents(unittest.TestCase):

    def setUp(self):
        self.sample_genai = SAMPLE_RESUMES["🤖 Senior Generative AI / LLM Engineer (High Match)"]
        self.sample_jr = SAMPLE_RESUMES["🔬 Junior Data Scientist / Analyst (Basic Match)"]
        self.role_genai = "🤖 Generative AI / LLM Engineer"

    def test_hard_skills_component_and_explanation(self):
        """Test hard skills scoring and verify detailed explanation of matched vs missing skills."""
        res = score_resume(self.sample_genai, target_role_name=self.role_genai)
        score = res["scores_breakdown"]["hard_skill_score"]
        self.assertGreaterEqual(score, 70.0)
        
        # Verify explanation
        exp = res["explanations"]["hard_skills"]
        self.assertIn("Assigned", exp)
        self.assertIn("core skills", exp)
        self.assertIn("secondary skills", exp)

    def test_dense_semantic_embedding_component(self):
        """Test genuine dense vector embedding comparison and synonym detection."""
        text_a = "Senior engineer skilled in Kubernetes, PyTorch, and large language models."
        text_b = "Specialist with k8s container orchestration, deep learning, and LLM systems."
        
        # Dense embeddings should detect strong semantic similarity despite vocabulary variance
        dense_eval = compute_dense_embedding_similarity(text_a, text_b)
        self.assertGreaterEqual(dense_eval["raw_similarity"], 0.60)
        self.assertGreaterEqual(dense_eval["calibrated_score"], 50.0)
        self.assertIn("explanation", dense_eval)
        self.assertFalse(dense_eval.get("is_fallback", False))

    def test_lexical_tfidf_similarity_component(self):
        """Verify TF-IDF measures surface lexical overlap and contains an accurate lexical disclaimer."""
        text_a = "Kubernetes orchestration container"
        text_b = "k8s deep learning neural networks"
        
        lex_eval = compute_lexical_tfidf_similarity(text_a, text_b)
        # Surface words don't match, so TF-IDF lexical overlap is low
        self.assertLess(lex_eval["raw_similarity"], 0.15)
        self.assertTrue(lex_eval["is_lexical_only"])
        self.assertIn("lexical", lex_eval["explanation"].lower())
        self.assertIn("not evaluate semantic", lex_eval["explanation"].lower())

    def test_quantifiable_impact_component_and_explanation(self):
        """Test impact analyzer detection of metrics and action verbs with defensible reasoning."""
        res = score_resume(self.sample_genai, target_role_name=self.role_genai)
        score = res["scores_breakdown"]["impact_score"]
        self.assertGreaterEqual(score, 60.0)
        
        exp = res["explanations"]["quantifiable_impact"]
        self.assertIn("Assigned", exp)
        self.assertIn("quantifiable metrics", exp)
        self.assertIn("action verbs", exp)

    def test_mlops_readiness_component_and_explanation(self):
        """Test MLOps tool extraction and production readiness scoring."""
        res = score_resume(self.sample_genai, target_role_name=self.role_genai)
        score = res["scores_breakdown"]["mlops_score"]
        self.assertGreaterEqual(score, 75.0)

        exp = res["explanations"]["mlops_readiness"]
        self.assertIn("Assigned", exp)
        self.assertIn("production", exp.lower())

    def test_ats_health_component_and_explanation(self):
        """Test ATS formatting health, section evaluation, contact verification, and readability."""
        res = score_resume(self.sample_genai, target_role_name=self.role_genai)
        score = res["scores_breakdown"]["ats_health_score"]
        self.assertGreaterEqual(score, 75.0)

        exp = res["explanations"]["ats_health"]
        self.assertIn("Assigned", exp)
        self.assertIn("section", exp.lower())
        self.assertIn("contact", exp.lower())

    def test_composite_score_formula_and_disclaimer(self):
        """Test overall composite aggregation, heuristic disclaimer, and tier verdict."""
        res = score_resume(self.sample_genai, target_role_name=self.role_genai)
        composite = res["composite_score"]
        self.assertGreaterEqual(composite, 70)
        self.assertLessEqual(composite, 100)

        # Rubric disclaimer must be explicitly present
        self.assertIn("rubric_disclaimer", res)
        self.assertEqual(res["rubric_disclaimer"], RUBRIC_DISCLAIMER)
        self.assertIn("heuristic model", res["rubric_disclaimer"].lower())

        # Explanations dictionary must have entries for all components
        explanations = res["explanations"]
        expected_keys = [
            "hard_skills", "semantic_similarity", "lexical_similarity",
            "quantifiable_impact", "mlops_readiness", "ats_health", "composite_score"
        ]
        for key in expected_keys:
            self.assertIn(key, explanations)
            self.assertGreater(len(explanations[key]), 20)

    def test_offline_readability_deterministic(self):
        """Verify calculate_ats_readability operates deterministically offline without socket hangs."""
        text = "Engineered real-time distributed microservices with automated deployment pipelines."
        metrics = calculate_ats_readability(text)
        self.assertIn("word_count", metrics)
        self.assertIn("flesch_score", metrics)
        self.assertGreater(metrics["flesch_score"], 0.0)

    def test_benchmark_role_fuzzy_resolution(self):
        """Verify fuzzy matching resolves role variants safely to exact benchmarks."""
        role1 = find_benchmark_role("MLOps")
        self.assertEqual(role1, "⚙️ MLOps Engineer")

        role2 = find_benchmark_role("Generative AI")
        self.assertEqual(role2, "🤖 Generative AI / LLM Engineer")

        role3 = find_benchmark_role("Nonexistent Role")
        self.assertIn(role3, ROLE_BENCHMARKS)

if __name__ == "__main__":
    unittest.main()
