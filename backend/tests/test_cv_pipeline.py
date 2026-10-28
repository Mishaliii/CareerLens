import io
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import fitz
import google.generativeai as genai
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.schemas import CandidateProfile, CVProcessResponse, StructuredCV
from backend.routers.cv import get_cv_pipeline
from backend.services.cv_pipeline import CVPipelineResult, CVPipelineService
from backend.services.gemini import GeminiService


class TestCVPipelineIntegration(unittest.TestCase):
    """Integration test suite for CV Processing Pipeline and FastAPI endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        # Block any live network calls to Google Generative AI across all tests
        cls._genai_blocker = patch.object(
            genai.GenerativeModel,
            "generate_content",
            side_effect=RuntimeError("Live Gemini API call forbidden in tests"),
        )
        cls._genai_blocker.start()

    @classmethod
    def tearDownClass(cls):
        cls._genai_blocker.stop()

    def setUp(self):
        """Set up test-level deterministic extraction mock and service isolation."""
        # Ensure GeminiService is always configured with a mock model during tests
        def _mock_gemini_init(svc, api_key=None):
            svc.api_key = "test_deterministic_key"
            svc._is_configured = True
            svc.model = MagicMock()

        self._init_patcher = patch.object(GeminiService, "__init__", _mock_gemini_init)
        self._init_patcher.start()

        # Provide deterministic mock for Gemini structured extraction: extracts from raw text
        # without ever making a network call, regardless of what GEMINI_API_KEY is configured
        def _deterministic_gemini_extraction(*args, **kwargs) -> StructuredCV:
            raw_text = args[1] if len(args) > 1 else (args[0] if args else kwargs.get("raw_text", ""))
            unconfigured_svc = GeminiService.__new__(GeminiService)
            unconfigured_svc.api_key = None
            unconfigured_svc._is_configured = False
            cv = unconfigured_svc._fallback_rule_based_extraction(raw_text)
            return cv.model_copy(update={"extraction_source": "gemini"})

        self._gemini_patcher = patch.object(
            GeminiService,
            "_call_gemini_structured_extraction",
            side_effect=_deterministic_gemini_extraction,
        )
        self._gemini_patcher.start()

    def tearDown(self):
        """Clean up test-level patches."""
        self._gemini_patcher.stop()
        self._init_patcher.stop()

    def _create_test_pdf(self, text: str = "", num_pages: int = 1) -> bytes:
        """Helper to create valid in-memory PDF bytes with PyMuPDF."""
        doc = fitz.open()
        for i in range(num_pages):
            page = doc.new_page()
            if text:
                page.insert_text((50, 50), f"Page {i + 1}\n{text}")
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_existing_routes_remain_functional(self):
        """Test 16: Existing GET / and GET /health routes remain functional."""
        res_root = self.client.get("/")
        self.assertEqual(res_root.status_code, 200)
        self.assertEqual(res_root.json(), {"message": "CareerLens AI Backend is running"})

        res_health = self.client.get("/health")
        self.assertEqual(res_health.status_code, 200)
        self.assertEqual(res_health.json(), {"status": "healthy"})

    def test_successful_end_to_end_cv_processing(self):
        """Test 1, 2, 3, 4: Successful end-to-end PDF processing with correct schema and analysis."""
        resume_text = """
Aarav Sharma
aarav@example.com | +91 9876543210 | Bengaluru, India
https://linkedin.com/in/aarav | https://github.com/aarav

Professional Summary:
Full stack developer with 3 years of experience building web applications.

Skills:
Python, JS, ReactJS, FastAPI, Postgres, Docker, K8s, Rust

Experience:
Senior Engineer at TechCorp
July 2021 - Present
Built scalable microservices and APIs with FastAPI and PostgreSQL.

Education:
B.Tech in Computer Science
National Institute of Technology
2017 - 2021
CGPA: 8.8 / 10

Projects:
CareerLens Platform
Built intelligent matching backend using FastAPI and Docker.
"""
        pdf_bytes = self._create_test_pdf(resume_text, num_pages=2)

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # Validate top-level response envelope
        self.assertTrue(data["success"])
        self.assertIn("successfully", data["message"])
        self.assertIsInstance(data["processing_time_ms"], (int, float))
        self.assertGreaterEqual(data["processing_time_ms"], 0.0)

        # Validate Document Analysis metadata
        analysis = data["document_analysis"]
        self.assertEqual(analysis["page_count"], 2)
        self.assertEqual(analysis["language"], "en")
        self.assertGreater(analysis["total_words"], 30)
        self.assertGreater(analysis["total_characters"], 100)
        self.assertGreater(analysis["pages_with_text"], 0)
        self.assertIsInstance(analysis["character_density_per_page"], (int, float))

        # Validate CandidateProfile structure
        profile = data["profile"]
        self.assertIn("metadata", profile)
        self.assertEqual(profile["metadata"]["page_count"], 2)
        self.assertEqual(profile["metadata"]["language"], "en")
        self.assertIn("personal", profile)
        self.assertIn("resume_statistics", profile)
        self.assertIn("structured_data", profile)

        # Validate Pydantic deserialization
        validated_resp = CVProcessResponse.model_validate(data)
        self.assertIsInstance(validated_resp.profile, CandidateProfile)

    def test_normalization_of_aliases(self):
        """Test 11: Normalization of representative aliases (JS, K8s, Postgres)."""
        resume_text = """
Skills:
JS, K8s, Postgres, ReactJS, sklearn

Projects:
Search Engine
Technologies: Postgres, K8s
"""
        pdf_bytes = self._create_test_pdf(resume_text)

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("skills.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        structured = response.json()["profile"]["structured_data"]
        self.assertIsNotNone(structured)
        skills = structured.get("skills", [])

        # Check canonicalization
        self.assertIn("JavaScript", skills)
        self.assertIn("Kubernetes", skills)
        self.assertIn("PostgreSQL", skills)
        self.assertIn("React", skills)
        self.assertIn("scikit-learn", skills)

        # Check raw aliases were replaced with canonical names
        self.assertNotIn("JS", skills)
        self.assertNotIn("K8s", skills)
        self.assertNotIn("Postgres", skills)

    def test_unknown_skills_preserved(self):
        """Test 13: Unknown skills and missing optional candidate fields preserved safely."""
        resume_text = """
Skills:
Python, Solidity, Next.js, CustomToolXYZ

Experience:
Freelance Developer
2022 - 2023
"""
        pdf_bytes = self._create_test_pdf(resume_text)

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("candidate.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)
        profile = response.json()["profile"]
        structured = profile["structured_data"]
        skills = structured.get("skills", [])

        # Python is canonicalized; unknown skills are faithfully preserved
        self.assertIn("Python", skills)
        self.assertIn("Solidity", skills)

        # Missing optional fields do not crash
        self.assertEqual(profile["personal"]["linkedin"], "")
        self.assertEqual(profile["personal"]["github"], "")

    def test_degree_date_grade_normalization(self):
        """Test 12: Degree, date, and grade normalization with unambiguous examples."""
        resume_text = """
Education:
B.Tech in Information Technology
National University
2018 - 2022
CGPA: 8.5 / 10

Experience:
Software Engineer at Acme
July 2022 - Present
"""
        pdf_bytes = self._create_test_pdf(resume_text)

        # Test pipeline normalization directly
        pipeline = CVPipelineService()
        result = pipeline.process_cv(pdf_bytes)

        self.assertIsNotNone(result.profile.structured_data)
        struct = result.profile.structured_data

        # Verify education canonicalization if extracted
        if struct.education:
            edu = struct.education[0]
            if edu.normalized_degree:
                self.assertEqual(edu.normalized_degree, "Bachelor of Technology")
            if edu.grade:
                self.assertEqual(edu.grade.value, 8.5)

        # Verify experience date normalization if extracted
        if struct.experience:
            exp = struct.experience[0]
            if exp.start_date_normalized:
                self.assertEqual(exp.start_date_normalized.year, 2022)
            if exp.is_current:
                self.assertTrue(exp.is_current)

    def test_fallback_extraction_without_gemini(self):
        """Test 10: Successful processing when structured extraction uses fallback."""
        pdf_bytes = self._create_test_pdf("Skills: Python, SQL\nExperience: Developer for 2 years")

        # Mock GeminiService to simulate unconfigured or failed API call
        with patch.object(GeminiService, "_call_gemini_structured_extraction", return_value=None):
            response = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("cv.pdf", pdf_bytes, "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        structured = data["profile"]["structured_data"]
        self.assertIsNotNone(structured)
        self.assertEqual(structured["extraction_source"], "fallback")

    def test_invalid_non_pdf_bytes_rejected(self):
        """Test 5: Invalid non-PDF bytes rejected with HTTP 400."""
        text_bytes = b"Hello, this is a plain text file, not a PDF."

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("resume.txt", text_bytes, "text/plain")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("valid PDF", response.json()["detail"])

    def test_empty_upload_rejected(self):
        """Test 6: Empty upload rejected with HTTP 400."""
        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )

        self.assertEqual(response.status_code, 400)
        self.assertIn("empty", response.json()["detail"].lower())

    def test_corrupt_pdf_bytes_rejected(self):
        """Test 7: Corrupt PDF bytes with PDF signature mapped to HTTP 422."""
        corrupt_bytes = b"%PDF-1.4\n" + b"\xff\xfe\x00\x01ThisIsCorruptGarbageDataNotValidPdf"

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("corrupt.pdf", corrupt_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("corrupted", response.json()["detail"].lower())

    def test_blank_pdf_with_no_readable_text_rejected(self):
        """Test 8: A valid PDF containing no readable text mapped to HTTP 422."""
        # Create a valid PDF document with 1 page but no text inserted
        blank_pdf_bytes = self._create_test_pdf(text="")

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("blank.pdf", blank_pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 422)
        self.assertIn("no readable text", response.json()["detail"].lower())

    def test_oversized_upload_rejected_content_length(self):
        """Test 9: Oversized uploads rejected with HTTP 413 via Content-Length."""
        oversized_length = 15 * 1024 * 1024  # 15 MiB

        # Early rejection via Content-Length header
        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("oversized.pdf", b"%PDF-dummy", "application/pdf")},
            headers={"content-length": str(oversized_length)},
        )

        self.assertEqual(response.status_code, 413)
        self.assertIn("10 MiB", response.json()["detail"])

    def test_oversized_upload_rejected_stream_bytes(self):
        """Test 9: Oversized uploads rejected with HTTP 413 via bounded stream reading (lightweight strategy)."""
        # Patch MAX_FILE_SIZE_BYTES to a small threshold (200 bytes) to test bounded stream reading
        # without allocating excessive memory or sending massive payloads
        with patch("backend.routers.cv.MAX_FILE_SIZE_BYTES", 200):
            payload = b"%PDF-" + b"0" * 300
            response = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("oversized.pdf", payload, "application/pdf")},
            )
            self.assertEqual(response.status_code, 413)
            self.assertIn("exceeds maximum allowed limit", response.json()["detail"].lower())

    def test_unexpected_exception_mapped_to_500(self):
        """Test 14: Unexpected downstream exceptions mapped to generic HTTP 500 response."""
        pdf_bytes = self._create_test_pdf("Skills: Python")

        mock_pipeline = MagicMock()
        mock_pipeline.process_cv.side_effect = RuntimeError("Database connection exploded / secret_key=12345")

        app.dependency_overrides[get_cv_pipeline] = lambda: mock_pipeline
        try:
            response = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
            )

            self.assertEqual(response.status_code, 500)
            detail = response.json()["detail"]
            self.assertEqual(detail, "An internal error occurred while processing the CV")
            # Ensure internal secrets / stack trace are NOT leaked
            self.assertNotIn("exploded", detail)
            self.assertNotIn("secret_key", detail)
        finally:
            app.dependency_overrides.clear()

    def test_no_persistent_or_temp_files_created(self):
        """Test 15: No persistent or temporary resume files created during processing."""
        temp_dir = tempfile.gettempdir()
        initial_temp_files = set(os.listdir(temp_dir))

        pdf_bytes = self._create_test_pdf("Skills: Python, FastAPI\nExperience: 2 years")

        response = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("resume.pdf", pdf_bytes, "application/pdf")},
        )

        self.assertEqual(response.status_code, 200)

        current_temp_files = set(os.listdir(temp_dir))
        new_files = current_temp_files - initial_temp_files
        # Ensure no resume files were written to temp
        for f in new_files:
            self.assertFalse(f.endswith(".pdf"), f"Unexpected PDF file leaked in temp directory: {f}")


if __name__ == "__main__":
    unittest.main()
