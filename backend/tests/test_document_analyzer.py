import unittest
import fitz
from backend.services.document_analyzer import DocumentAnalyzer, DocumentAnalysisResult
from backend.services.candidate_profile import CandidateProfileBuilder
from backend.models.schemas import CandidateProfile


class TestDocumentAnalyzer(unittest.TestCase):
    """Unit tests for DocumentAnalyzer and CandidateProfile integration."""

    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = DocumentAnalyzer()
        self.profile_builder = CandidateProfileBuilder()

    def _create_test_pdf(self, pages_text: list) -> bytes:
        """Helper to create an in-memory PDF with specified text per page."""
        doc = fitz.open()
        for text in pages_text:
            page = doc.new_page()
            if text:
                page.insert_text((50, 50), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_valid_single_page_pdf(self):
        """Test analyzing a valid single-page PDF."""
        resume_text = (
            "Jane Doe\nSoftware Engineer\njane.doe@example.com\n"
            "Summary: Experienced software engineer specializing in backend systems.\n"
            "Experience: Built distributed microservices in Python and Go at TechCorp from 2020 to 2024.\n"
            "Education: Bachelor of Science in Computer Science, State University, 2019.\n"
            "Skills: Python, Go, Docker, Kubernetes, PostgreSQL, FastAPI."
        )
        pdf_bytes = self._create_test_pdf([resume_text])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertIsInstance(result, DocumentAnalysisResult)
        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.pages_with_text, 1)
        self.assertEqual(result.pages_sparse_or_empty, 0)
        self.assertGreater(result.total_words, 20)
        self.assertGreater(result.total_characters, 100)
        self.assertGreater(result.character_density_per_page, 100.0)
        self.assertEqual(result.language, "en")
        self.assertIsNotNone(result.parser_confidence)
        self.assertGreaterEqual(result.parser_confidence, 0.7)
        self.assertLessEqual(result.parser_confidence, 1.0)

    def test_valid_multi_page_pdf(self):
        """Test analyzing a valid multi-page PDF."""
        page1 = (
            "Alex Smith - Senior Systems Architect\nalex@example.com\n"
            "Summary: Over 10 years of experience designing scalable distributed cloud infrastructure.\n"
            "Experience:\nLead Architect at Global Cloud Inc (2018-2024).\n"
            "Designed multi-region data storage platform serving 20M requests daily.\n"
            "Technologies: Kubernetes, Terraform, AWS, Python, Go, Kafka."
        )
        page2 = (
            "Education & Certifications\n"
            "M.S. in Computer Engineering from ABC Institute, 2014.\n"
            "B.S. in Software Systems, 2012.\n"
            "Certifications: AWS Solutions Architect Professional, CKA Kubernetes Administrator.\n"
            "Projects: Open-source distributed stream processing engine."
        )
        pdf_bytes = self._create_test_pdf([page1, page2])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.pages_with_text, 2)
        self.assertEqual(result.pages_sparse_or_empty, 0)
        self.assertEqual(result.language, "en")
        self.assertIsNotNone(result.parser_confidence)
        self.assertGreaterEqual(result.parser_confidence, 0.8)

    def test_empty_input(self):
        """Test empty bytes raise ValueError with clear message."""
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze(b"")
        self.assertIn("empty", str(context.exception).lower())

    def test_invalid_or_corrupt_pdf_bytes(self):
        """Test invalid or corrupt bytes raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze(b"not a valid pdf file")
        self.assertIn("invalid", str(context.exception).lower())

        # Test too short bytes (< 4 bytes)
        with self.assertRaises(ValueError) as context:
            self.analyzer.analyze(b"%P")
        self.assertIn("too short", str(context.exception).lower())

        # Test broken PDF header
        with self.assertRaises(ValueError):
            self.analyzer.analyze(b"%PDF-1.4\n" + b"\x00" * 500 + b"%%EOF")

    def test_document_with_readable_text(self):
        """Test text yield metrics on a document with readable text."""
        text = "Software Engineer with background in data engineering and machine learning."
        pdf_bytes = self._create_test_pdf([text])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertGreater(result.total_characters, 0)
        self.assertEqual(result.total_words, len(text.split()))
        self.assertEqual(result.pages_with_text, 1)
        self.assertEqual(result.pages_sparse_or_empty, 0)

    def test_document_with_very_little_readable_text(self):
        """Test handling of sparse document with little or no readable text."""
        # Only 5 characters, below the 20-character readable threshold
        sparse_text = "Hi"
        pdf_bytes = self._create_test_pdf([sparse_text])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertEqual(result.page_count, 1)
        self.assertEqual(result.pages_with_text, 0)
        self.assertEqual(result.pages_sparse_or_empty, 1)
        self.assertLess(result.total_words, 5)
        # Sparse sample should have unknown/None language
        self.assertIsNone(result.language)
        # Confidence score should reflect sparse content
        self.assertIsNotNone(result.parser_confidence)
        self.assertLess(result.parser_confidence, 0.4)

    def test_language_detection_informative_sample(self):
        """Test deterministic language detection on informative English, Spanish, French, German text."""
        # English sample
        en_text = (
            "Professional experience in software engineering with a degree in computer science. "
            "Led development of web applications using Python and PostgreSQL at university and tech startups."
        )
        self.assertEqual(self.analyzer.detect_language(en_text), "en")

        # Spanish sample
        es_text = (
            "Experiencia profesional en ingenieria de software con educacion universitaria en sistemas. "
            "Desarrollo de proyectos con habilidades avanzadas en bases de datos para el trabajo diario."
        )
        self.assertEqual(self.analyzer.detect_language(es_text), "es")

        # French sample
        fr_text = (
            "Experience professionnelle dans le developpement de projets logiciels avec une formation universitaire. "
            "Competences techniques solides pour le travail en equipe et la gestion des systemes."
        )
        self.assertEqual(self.analyzer.detect_language(fr_text), "fr")

        # German sample
        de_text = (
            "Berufserfahrung in der Softwareentwicklung mit einer Ausbildung an der Universitat. "
            "Kenntnisse in der Entwicklung von Projekten und der Arbeit mit modernen Technologien."
        )
        self.assertEqual(self.analyzer.detect_language(de_text), "de")

    def test_language_detection_short_or_ambiguous_sample(self):
        """Test that short or ambiguous samples return None rather than guessing."""
        self.assertIsNone(self.analyzer.detect_language(""))
        self.assertIsNone(self.analyzer.detect_language("John Doe Resume"))
        self.assertIsNone(self.analyzer.detect_language("Python, Java, C++, SQL, Git, Linux"))
        self.assertIsNone(self.analyzer.detect_language("12345 67890 2024"))

    def test_parsing_confidence_boundaries_and_range(self):
        """Test parsing confidence boundaries (always within 0.0 to 1.0)."""
        # Complete resume -> high confidence
        full_resume = (
            "John Doe - Senior Developer\njohn@example.com\n"
            "Summary: Experienced software engineer with extensive background in Python systems.\n"
            "Experience: Senior Engineer at CloudCorp (2020-2024). Led architecture of payment pipeline.\n"
            "Education: B.Tech in Computer Science from National Institute of Technology, 2019.\n"
            "Projects: High-performance cache proxy handling 50k QPS.\n"
            "Skills: Python, Rust, Docker, Kubernetes, AWS, SQL, Redis."
        )
        pdf_bytes = self._create_test_pdf([full_resume])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertGreaterEqual(result.parser_confidence, 0.0)
        self.assertLessEqual(result.parser_confidence, 1.0)
        self.assertGreaterEqual(result.parser_confidence, 0.8)

        # Blank page -> 0.0 confidence
        blank_pdf = self._create_test_pdf([""])
        blank_result = self.analyzer.analyze(blank_pdf)
        self.assertEqual(blank_result.parser_confidence, 0.0)

    def test_partial_text_extraction_across_multiple_pages(self):
        """Test multi-page document where one page has text and another is empty."""
        page1 = "Jane Doe\nSoftware Engineer with experience in cloud infrastructure (2021-2024)."
        page2 = ""  # Scanned or blank page
        pdf_bytes = self._create_test_pdf([page1, page2])
        result = self.analyzer.analyze(pdf_bytes)

        self.assertEqual(result.page_count, 2)
        self.assertEqual(result.pages_with_text, 1)
        self.assertEqual(result.pages_sparse_or_empty, 1)
        self.assertIsNotNone(result.parser_confidence)
        # Confidence should be penalized for the empty page compared to a 100% readable multi-page doc
        self.assertLess(result.parser_confidence, 0.8)

    def test_deterministic_results_for_identical_inputs(self):
        """Test that analyzing identical inputs yields identical, deterministic results."""
        resume_text = (
            "Alice Wonder\nData Scientist\nalice@example.com\n"
            "Summary: Data Scientist with 4 years experience building machine learning models.\n"
            "Experience: Machine Learning Engineer at DataCorp (2021-Present).\n"
            "Education: Master of Science in Data Science, 2020.\n"
            "Skills: Python, PyTorch, SQL, Pandas, Scikit-learn."
        )
        pdf_bytes = self._create_test_pdf([resume_text])

        res1 = self.analyzer.analyze(pdf_bytes)
        res2 = self.analyzer.analyze(pdf_bytes)

        self.assertEqual(res1.page_count, res2.page_count)
        self.assertEqual(res1.language, res2.language)
        self.assertEqual(res1.parser_confidence, res2.parser_confidence)
        self.assertEqual(res1.total_characters, res2.total_characters)
        self.assertEqual(res1.total_words, res2.total_words)
        self.assertEqual(res1.pages_with_text, res2.pages_with_text)
        self.assertEqual(res1.pages_sparse_or_empty, res2.pages_sparse_or_empty)
        self.assertEqual(res1.to_dict(), res2.to_dict())

    def test_candidate_profile_builder_integration_with_analysis(self):
        """Test CandidateProfileBuilder integration with DocumentAnalysisResult."""
        parsed_sections = {
            "summary": "Full stack engineer.",
            "experience": "Senior Developer at Tech (2020-2024)",
            "education": "B.Tech in CS",
            "projects": "Project A",
            "skills": "Python, React",
            "certifications": "AWS Certified",
            "raw_text": "Alex Doe\nalex@example.com\nSummary: Full stack engineer.",
        }

        analysis = DocumentAnalysisResult(
            page_count=2,
            language="en",
            parser_confidence=0.88,
            total_characters=500,
            total_words=80,
            pages_with_text=2,
            pages_sparse_or_empty=0,
            character_density_per_page=250.0,
        )

        # 1. Integration via explicit argument
        profile = self.profile_builder.build(parsed_sections, document_analysis=analysis)
        self.assertIsInstance(profile, CandidateProfile)
        self.assertEqual(profile.metadata.page_count, 2)
        self.assertEqual(profile.metadata.language, "en")
        self.assertEqual(profile.metadata.parser_confidence, 0.88)

        # 2. Integration via parsed_sections dictionary key
        sections_with_analysis = dict(parsed_sections)
        sections_with_analysis["document_analysis"] = analysis
        profile2 = self.profile_builder.build(sections_with_analysis)
        self.assertEqual(profile2.metadata.page_count, 2)
        self.assertEqual(profile2.metadata.language, "en")
        self.assertEqual(profile2.metadata.parser_confidence, 0.88)

    def test_candidate_profile_builder_backward_compatibility(self):
        """Test that CandidateProfileBuilder remains 100% backward compatible without analysis."""
        parsed_sections = {
            "summary": "Developer seeking roles.",
            "raw_text": "John Doe\njohn@example.com",
        }
        # Calling without document_analysis parameter
        profile = self.profile_builder.build(parsed_sections)
        self.assertIsInstance(profile, CandidateProfile)
        self.assertIsNone(profile.metadata.page_count)
        self.assertIsNone(profile.metadata.language)
        self.assertIsNone(profile.metadata.parser_confidence)
        self.assertEqual(profile.personal.email, "john@example.com")


if __name__ == "__main__":
    unittest.main()
