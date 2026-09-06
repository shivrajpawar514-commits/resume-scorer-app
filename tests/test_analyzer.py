"""
Unit Tests for AI/ML Resume Scorer & Analytics Engine
"""

import unittest
from analyzer.skills_taxonomy import ROLE_BENCHMARKS, CATEGORIZED_SKILLS
from analyzer.sample_resumes import SAMPLE_RESUMES
from analyzer.matcher import extract_skills_from_text, extract_skills_from_job_description
from analyzer.section_parser import parse_resume_sections, evaluate_section_completeness
from analyzer.impact_analyzer import analyze_impact_and_verbs
from analyzer.scoring_engine import score_resume, compute_tfidf_similarity
from analyzer.report_builder import generate_pdf_report, generate_markdown_report

class TestResumeScorerEngine(unittest.TestCase):

    def setUp(self):
        self.sample_genai = SAMPLE_RESUMES["🤖 Senior Generative AI / LLM Engineer (High Match)"]
        self.sample_jr = SAMPLE_RESUMES["🔬 Junior Data Scientist / Analyst (Basic Match)"]

    def test_skills_taxonomy_integrity(self):
        """Verify role benchmarks and categorized skills exist and are non-empty."""
        self.assertGreater(len(ROLE_BENCHMARKS), 5)
        self.assertGreater(len(CATEGORIZED_SKILLS), 5)
        for role_name, data in ROLE_BENCHMARKS.items():
            self.assertIn("core_skills", data)
            self.assertIn("secondary_skills", data)
            self.assertGreater(len(data["core_skills"]), 0)

    def test_phrase_matcher_extraction(self):
        """Test SpaCy phrase matching for multi-word skills."""
        text = "Experienced in Large Language Models, Prompt Engineering, PyTorch, and Docker."
        extracted = extract_skills_from_text(text)
        skills = extracted["all_matched"]
        
        self.assertIn("large language models", skills)
        self.assertIn("prompt engineering", skills)
        self.assertIn("pytorch", skills)
        self.assertIn("docker", skills)

    def test_section_parser(self):
        """Test section chunking and completeness evaluation."""
        sections = parse_resume_sections(self.sample_genai)
        self.assertIn("Experience", sections)
        self.assertIn("Education", sections)
        self.assertIn("Skills", sections)

        eval_res = evaluate_section_completeness(sections)
        self.assertGreaterEqual(eval_res["score"], 80)

    def test_impact_and_action_verbs(self):
        """Test detection of action verbs and quantifiable metrics."""
        impact = analyze_impact_and_verbs(self.sample_genai)
        self.assertGreater(len(impact["strong_verbs_found"]), 5)
        self.assertGreater(len(impact["metrics_found"]), 0)
        self.assertGreater(impact["overall_impact_score"], 50)

    def test_tfidf_similarity(self):
        """Test TF-IDF cosine similarity calculation."""
        sim = compute_tfidf_similarity(self.sample_genai, "Generative AI LLM LangChain PyTorch Pinecone")
        self.assertGreater(sim, 0.0)
        self.assertLessEqual(sim, 1.0)

    def test_scoring_engine_composite(self):
        """Test composite scoring and role alignment."""
        res_genai = score_resume(self.sample_genai, target_role_name="🤖 Generative AI / LLM Engineer")
        self.assertGreaterEqual(res_genai["composite_score"], 75)
        self.assertIn("radar_dimensions", res_genai)
        self.assertEqual(len(res_genai["radar_dimensions"]), 5)

        # Junior should score lower on Senior GenAI role
        res_jr = score_resume(self.sample_jr, target_role_name="🤖 Generative AI / LLM Engineer")
        self.assertLess(res_jr["composite_score"], res_genai["composite_score"])

    def test_report_generation(self):
        """Test PDF and Markdown report builders."""
        res = score_resume(self.sample_genai, target_role_name="🤖 Generative AI / LLM Engineer")
        
        # Test PDF bytes
        pdf_bytes = generate_pdf_report(res)
        self.assertIsInstance(pdf_bytes, bytes)
        self.assertGreater(len(pdf_bytes), 1000)

        # Test Markdown text
        md_text = generate_markdown_report(res)
        self.assertIsInstance(md_text, str)
        self.assertIn("# 📄 AI/ML Resume Scorer", md_text)

if __name__ == "__main__":
    unittest.main()
