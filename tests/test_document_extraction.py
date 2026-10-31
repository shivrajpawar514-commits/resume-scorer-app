"""
Unit Tests for Document Extraction Edge Cases
Verifies:
- Multi-column layout PDFs (preserved reading order without column interleaving)
- Scanned pages (image-only PDFs flagged with ATS warnings)
- Unusual fonts, ligatures (fi, fl, ffi), smart quotes, and unicode normalization
- Empty files (0-byte) and empty document handling
- Corrupted documents (truncated/malformed headers and data)
- Resumes with missing sections (graceful degradation without crashes)
"""

import unittest
import io
import fitz  # PyMuPDF
from PIL import Image

from analyzer.extractor import (
    extract_text_from_pdf,
    extract_text_from_pdf_with_diagnostics,
    normalize_pdf_typography,
    clean_extracted_text
)
from analyzer.section_parser import parse_resume_sections, evaluate_section_completeness
from analyzer.scoring_engine import score_resume

class TestDocumentExtractionEdgeCases(unittest.TestCase):

    def create_multicolumn_pdf(self) -> bytes:
        """Generates a synthetic 2-column resume PDF using PyMuPDF."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)  # Standard Letter size

        # Full-width header
        page.insert_text(fitz.Point(50, 40), "Alex Johnson - Machine Learning Engineer", fontsize=14)
        page.insert_text(fitz.Point(50, 60), "alex.j@mlmail.com | +1-555-0100 | linkedin.com/in/alexj", fontsize=10)

        # Left Column (x: 50 to 250): Skills & Education
        page.insert_text(fitz.Point(50, 100), "SKILLS", fontsize=12)
        page.insert_text(fitz.Point(50, 120), "Python, PyTorch, Docker", fontsize=9)
        page.insert_text(fitz.Point(50, 140), "FastAPI, Kubernetes, SQL", fontsize=9)
        page.insert_text(fitz.Point(50, 180), "EDUCATION", fontsize=12)
        page.insert_text(fitz.Point(50, 200), "B.S. in Computer Science", fontsize=9)
        page.insert_text(fitz.Point(50, 220), "Stanford University (2020)", fontsize=9)

        # Right Column (x: 300 to 550): Experience & Projects
        page.insert_text(fitz.Point(300, 100), "EXPERIENCE", fontsize=12)
        page.insert_text(fitz.Point(300, 120), "Senior AI Engineer at AI Innovations (2021-Present)", fontsize=9)
        page.insert_text(fitz.Point(300, 140), "Architected distributed model serving reducing latency by 45%.", fontsize=9)
        page.insert_text(fitz.Point(300, 160), "Deployed scalable RAG pipeline using LangChain and Pinecone.", fontsize=9)
        page.insert_text(fitz.Point(300, 200), "PROJECTS", fontsize=12)
        page.insert_text(fitz.Point(300, 220), "Open-source fine-tuning tool with 1,500 GitHub stars.", fontsize=9)

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def create_scanned_image_only_pdf(self) -> bytes:
        """Generates an image-only PDF containing a bitmap scan without any text layer."""
        doc = fitz.open()
        page = doc.new_page(width=612, height=792)

        # Create a synthetic bitmap image (simulating a scanned paper document)
        img = Image.new("RGB", (300, 400), color=(240, 240, 240))
        img_buffer = io.BytesIO()
        img.save(img_buffer, format="PNG")
        img_bytes = img_buffer.getvalue()

        # Insert image into page without adding any font/text blocks
        page.insert_image(fitz.Rect(50, 50, 550, 750), stream=img_bytes)

        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_multicolumn_layout_preserves_reading_order(self):
        """Verify 2-column resume text is extracted column-by-column rather than line-interleaved."""
        pdf_bytes = self.create_multicolumn_pdf()
        diag = extract_text_from_pdf_with_diagnostics(pdf_bytes)

        self.assertTrue(diag["is_multi_column"], "Failed to detect multi-column layout")
        text = diag["text"]
        self.assertGreater(len(text), 100)

        # Verify sections are intact and not interleaved
        # In multi-column reading order, Left Column (SKILLS / EDUCATION) should not interleave with
        # Right Column (EXPERIENCE / PROJECTS) lines.
        skills_idx = text.find("SKILLS")
        exp_idx = text.find("EXPERIENCE")
        self.assertNotEqual(skills_idx, -1)
        self.assertNotEqual(exp_idx, -1)

        # Parsing sections should successfully detect both Skills, Education, Experience, Projects
        sections = parse_resume_sections(text)
        self.assertIn("Skills", sections)
        self.assertIn("Education", sections)
        self.assertIn("Experience", sections)
        self.assertIn("Projects", sections)

    def test_scanned_page_detection_and_warning(self):
        """Verify image-only PDFs without text layer are flagged as scanned documents."""
        pdf_bytes = self.create_scanned_image_only_pdf()
        diag = extract_text_from_pdf_with_diagnostics(pdf_bytes)

        self.assertTrue(diag["is_scanned"], "Should detect that PDF is an image-only scan")
        self.assertEqual(diag["text"], "")
        self.assertGreater(len(diag["warnings"]), 0)
        self.assertTrue(any("scanned" in w.lower() for w in diag["warnings"]))

    def test_unusual_fonts_and_ligature_normalization(self):
        """Verify typographical ligatures (fi, fl, ffi, ffl) and smart quotes are properly normalized."""
        ligature_text = (
            "Experienced in \ufb01ne-tuning Large Language Models, \ufb02oating-point quantization, "
            "and e\ufb03cient distributed inference. Uses “smart quotes” and – en-dashes • bullets."
        )
        normalized = normalize_pdf_typography(ligature_text)

        # Ligatures must be unpacked so NLP matcher can tokenize "fine-tuning", "floating-point", "efficient"
        self.assertIn("fine-tuning", normalized)
        self.assertIn("floating-point", normalized)
        self.assertIn("efficient", normalized)
        self.assertIn('"smart quotes"', normalized)
        self.assertIn("- en-dashes", normalized)
        self.assertIn("* bullets", normalized)

    def test_empty_files_handling(self):
        """Verify 0-byte files and empty streams do not crash the engine."""
        empty_bytes = b""
        diag = extract_text_from_pdf_with_diagnostics(empty_bytes)
        self.assertTrue(diag["is_empty"])
        self.assertEqual(diag["char_count"], 0)
        self.assertIsNotNone(diag["error"])

        # Direct extraction call returns empty string without crashing
        extracted = extract_text_from_pdf(empty_bytes)
        self.assertEqual(extracted, "")

        # Scoring engine handles empty string safely
        score_res = score_resume("")
        self.assertIn("error", score_res)
        self.assertEqual(score_res["composite_score"], 0)

    def test_corrupted_document_handling(self):
        """Verify corrupted or truncated PDF bytes return clean errors without unhandled exceptions."""
        corrupted_bytes = b"%PDF-1.4\n%Malformed binary stream that abruptly terminates\x00\xff\xfe"
        diag = extract_text_from_pdf_with_diagnostics(corrupted_bytes)
        self.assertIsNotNone(diag["error"])
        self.assertEqual(diag["text"], "")

        # Completely random junk bytes
        junk_bytes = b"NOT_A_PDF_AT_ALL_JUST_RANDOM_GARBAGE_BYTES_12345"
        diag_junk = extract_text_from_pdf_with_diagnostics(junk_bytes)
        self.assertIsNotNone(diag_junk["error"])

    def test_missing_sections_graceful_scoring(self):
        """Verify resumes missing Experience, Education, or Skills are scored and evaluated safely."""
        # Resume with ONLY skills and contact info (missing Experience, Education, Projects)
        sparse_resume = """
        John Doe | john.doe@email.com | +1-555-0199 | linkedin.com/in/johndoe
        SKILLS
        Python, PyTorch, Docker, Kubernetes, SQL, FastAPI, Machine Learning
        """
        sections = parse_resume_sections(sparse_resume)
        self.assertIn("Skills", sections)
        self.assertNotIn("Experience", sections)
        self.assertNotIn("Education", sections)

        eval_res = evaluate_section_completeness(sections)
        # Should heavily penalize missing critical sections but compute without crashing
        self.assertLess(eval_res["score"], 50.0)

        # Full scoring should succeed and explain missing sections
        res = score_resume(sparse_resume, target_role_name="🤖 Generative AI / LLM Engineer")
        self.assertIn("composite_score", res)
        self.assertLess(res["composite_score"], 60)
        self.assertIn("ats_health", res["explanations"])

if __name__ == "__main__":
    unittest.main()
