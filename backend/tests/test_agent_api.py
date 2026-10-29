import io
import unittest
from unittest.mock import MagicMock, patch

import fitz
import google.generativeai as genai
from fastapi.testclient import TestClient

from backend.main import app
from backend.models.schemas import CVProcessResponse, ResumeQueryResponse
from backend.services.embedder import MockEmbedder
from backend.services.gemini import GeminiService
from backend.services.resume_agent import get_resume_agent, reset_resume_agent
from backend.services.retriever import get_retriever, reset_retriever
from backend.services.vector_store import get_vector_store, reset_vector_store


class TestAgentAPIIntegration(unittest.TestCase):
    """End-to-end integration tests for Resume Intelligence Agent endpoints and CV pipeline."""

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
        """Set up test environment and reset ephemeral stores."""
        from backend.db.database import Base, SessionLocal, engine
        from backend.db.models import User
        from backend.services.auth_service import get_current_user

        Base.metadata.create_all(bind=engine)
        with SessionLocal() as db_session:
            u = db_session.query(User).filter(User.id == "test_user_agent_api").first()
            if not u:
                self.mock_user = User(
                    id="test_user_agent_api",
                    email="agent_api_user@example.com",
                    hashed_password="hashed_pw",
                    full_name="Agent Test User",
                    is_active=True,
                )
                db_session.add(self.mock_user)
                db_session.commit()
            else:
                self.mock_user = u

        app.dependency_overrides[get_current_user] = lambda: self.mock_user

        reset_vector_store()
        reset_retriever()
        reset_resume_agent()

        # Force MockEmbedder so tests never attempt to load neural network weights
        self.mock_embedder = MockEmbedder(dimension=64)
        self._embedder_patcher = patch(
            "backend.services.retriever.get_embedder",
            return_value=self.mock_embedder,
        )
        self._embedder_patcher.start()

        # Deterministic mock for Gemini CV extraction and QA
        def _mock_gemini_init(svc, api_key=None):
            svc.api_key = "test_deterministic_key"
            svc._is_configured = True
            mock_model = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = '{"answer": "Alex Morgan is a Senior Backend Engineer at FinTech Labs [1].", "cited_source_ids": [1]}'
            mock_model.generate_content.return_value = mock_resp
            svc.model = mock_model

        self._init_patcher = patch.object(GeminiService, "__init__", _mock_gemini_init)
        self._init_patcher.start()

        def _deterministic_gemini_extraction(*args, **kwargs):
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
        from backend.services.auth_service import get_current_user
        app.dependency_overrides.pop(get_current_user, None)

        self._gemini_patcher.stop()
        self._init_patcher.stop()
        self._embedder_patcher.stop()
        reset_vector_store()
        reset_retriever()
        reset_resume_agent()

    def _create_test_pdf(self, text: str = "") -> bytes:
        """Helper to create valid in-memory PDF bytes."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), text or "Sample text")
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def test_status_endpoint_for_unindexed_resume(self):
        """Test GET /api/v1/agent/resume/{resume_id}/status for unindexed resume and 404 for unknown."""
        # Unknown/unowned resume returns 404
        res_404 = self.client.get("/api/v1/agent/resume/non_existent_id/status")
        self.assertEqual(res_404.status_code, 404)

        # Unindexed resume owned by authenticated user returns 200 with is_indexed=False
        from backend.db.database import SessionLocal
        from backend.db.models import Resume
        with SessionLocal() as db_session:
            existing = db_session.query(Resume).filter(Resume.id == "unindexed_user_resume").first()
            if not existing:
                db_session.add(Resume(
                    id="unindexed_user_resume",
                    user_id=self.mock_user.id,
                    filename="unindexed.pdf",
                    indexing_status="unindexed",
                    chunk_count=0,
                ))
                db_session.commit()

        res = self.client.get("/api/v1/agent/resume/unindexed_user_resume/status")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["resume_id"], "unindexed_user_resume")
        self.assertFalse(data["is_indexed"])
        self.assertEqual(data["chunk_count"], 0)
        self.assertEqual(data["sections_indexed"], [])

    def test_query_validation_errors(self):
        """Test POST /api/v1/agent/resume/query rejects invalid payloads."""
        # Query too short (< 2 chars)
        res_short = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": "res_1234", "query": "a"},
        )
        self.assertEqual(res_short.status_code, 422)

        # Missing resume_id
        res_missing = self.client.post(
            "/api/v1/agent/resume/query",
            json={"query": "Valid question here?"},
        )
        self.assertEqual(res_missing.status_code, 422)

    def test_end_to_end_cv_upload_indexing_and_agent_query(self):
        """
        Complete end-to-end user workflow:
        1. Upload PDF resume via POST /api/v1/cv/process
        2. Verify backward-compatible response contains resume_id and indexed=True
        3. Verify GET /api/v1/agent/resume/{resume_id}/status reports is_indexed=True
        4. Ask question via POST /api/v1/agent/resume/query
        5. Verify grounded answer and citation metadata
        """
        resume_text = (
            "Alex Morgan\n"
            "alex@example.com | Bengaluru, India\n\n"
            "Summary:\n"
            "Experienced Python developer with 4 years building backend systems.\n\n"
            "Experience:\n"
            "Senior Backend Engineer at FinTech Labs (2020 - Present)\n"
            "Designed and implemented distributed payment APIs using FastAPI and PostgreSQL.\n\n"
            "Education:\n"
            "B.Tech in Computer Science, Tech University (2016 - 2020)\n\n"
            "Skills:\n"
            "Python, FastAPI, PostgreSQL, Docker, Redis\n"
        )
        pdf_bytes = self._create_test_pdf(resume_text)

        # Step 1: Upload CV
        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("alex_resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        self.assertEqual(upload_res.status_code, 200)
        upload_data = upload_res.json()

        # Step 2: Verify existing fields remain intact
        self.assertTrue(upload_data["success"])
        self.assertIn("profile", upload_data)
        self.assertIn("document_analysis", upload_data)
        self.assertIn("processing_time_ms", upload_data)

        # Verify new RAG indexing fields
        self.assertIn("resume_id", upload_data)
        resume_id = upload_data["resume_id"]
        self.assertIsNotNone(resume_id)
        self.assertTrue(upload_data["indexed"])
        self.assertGreater(upload_data["chunk_count"], 0)

        # Step 3: Check index status endpoint
        status_res = self.client.get(f"/api/v1/agent/resume/{resume_id}/status")
        self.assertEqual(status_res.status_code, 200)
        status_data = status_res.json()
        self.assertTrue(status_data["is_indexed"])
        self.assertGreater(status_data["chunk_count"], 0)
        self.assertIn("experience", status_data["sections_indexed"])

        # Step 4: Query the agent about experience
        query_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={
                "resume_id": resume_id,
                "query": "Python FastAPI FinTech Labs backend systems",
                "top_k": 3,
            },
        )
        self.assertEqual(query_res.status_code, 200)
        query_data = query_res.json()

        self.assertTrue(query_data["success"])
        self.assertTrue(query_data["has_sufficient_evidence"])
        self.assertEqual(query_data["resume_id"], resume_id)
        self.assertIn("[1]", query_data["answer"])
        self.assertGreaterEqual(len(query_data["citations"]), 1)

        first_cit = query_data["citations"][0]
        self.assertEqual(first_cit["citation_id"], 1)
        self.assertIn(first_cit["section"], ["experience", "summary", "skills"])
        self.assertIsNotNone(first_cit["snippet"])

    def test_end_to_end_query_for_absent_information_returns_insufficient_evidence(self):
        """Test query for completely absent skills/topics returns has_sufficient_evidence=False."""
        resume_text = (
            "Sara Lee\nsara@example.com\n\n"
            "Summary:\nFrontend developer specializing in React and CSS styling.\n\n"
            "Skills:\nReact, CSS, HTML\n"
        )
        pdf_bytes = self._create_test_pdf(resume_text)

        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("sara_resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        self.assertEqual(upload_res.status_code, 200)
        resume_id = upload_res.json()["resume_id"]

        # Query about an unmentioned skill/domain
        query_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={
                "resume_id": resume_id,
                "query": "What quantum computing algorithms and particle accelerators have they used?",
            },
        )
        self.assertEqual(query_res.status_code, 200)
        data = query_res.json()
        self.assertTrue(data["success"])
        self.assertFalse(data["has_sufficient_evidence"])
        self.assertIn("does not contain information", data["answer"].lower())
        self.assertEqual(data["citations"], [])

    def test_end_to_end_gemini_failure_returns_defensive_error_response(self):
        """Test LLM generation failure returns graceful 200 error response without exposing tracebacks."""
        resume_text = "John Doe\njohn@example.com\n\nExperience:\nBackend Engineer at CloudCorp\n"
        pdf_bytes = self._create_test_pdf(resume_text)

        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("john_resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        resume_id = upload_res.json()["resume_id"]

        # Force the agent's LLM call to simulate a TimeoutError
        with patch.object(
            get_resume_agent().gemini_service.model,
            "generate_content",
            side_effect=TimeoutError("Mock timeout from Gemini API"),
        ):
            query_res = self.client.post(
                "/api/v1/agent/resume/query",
                json={
                    "resume_id": resume_id,
                    "query": "Backend Engineer CloudCorp experience details",
                },
            )
            self.assertEqual(query_res.status_code, 200)
            data = query_res.json()
            self.assertFalse(data["success"])
            self.assertIn("temporarily unavailable", data["answer"])
            self.assertEqual(data["citations"], [])

    def test_reindexing_changed_resume_clears_stale_chunks(self):
        """Test re-indexing with fewer chunks purges stale prior chunks."""
        v1_text = (
            "Candidate One\nc1@example.com\n\n"
            "Summary:\nSoftware Engineer.\n\n"
            "Experience:\nEngineer at Firm A\n\n"
            "Projects:\nProject Beta\n\n"
            "Education:\nB.Sc University\n\n"
            "Skills:\nPython, SQL\n"
        )
        res_v1 = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("v1.pdf", io.BytesIO(self._create_test_pdf(v1_text)), "application/pdf")},
        )
        r_id_1 = res_v1.json()["resume_id"]
        v1_count = res_v1.json()["chunk_count"]
        self.assertGreater(v1_count, 3)

        # Directly re-index the same resume_id with only 1 chunk
        from backend.models.schemas import ResumeChunk
        single_chunk = [
            ResumeChunk(
                chunk_id=f"{r_id_1}_only_one",
                resume_id=r_id_1,
                section="summary",
                content="Only one chunk remains after major revision.",
            )
        ]
        get_retriever().index_resume(r_id_1, single_chunk, purge_existing=True)

        status_res = self.client.get(f"/api/v1/agent/resume/{r_id_1}/status")
        status_data = status_res.json()
        self.assertEqual(status_data["chunk_count"], 1)
        self.assertEqual(status_data["sections_indexed"], ["summary"])


if __name__ == "__main__":
    unittest.main()
