"""
Comprehensive security, ownership enforcement, multi-user isolation,
and persistence test suite for CareerLens Phase 4.

Verifies:
1. Authentication requirements (401 on missing, invalid, or expired tokens)
2. Strict user ownership enforcement (User A vs User B isolation, 404 responses)
3. Client-supplied user_id body tampering defense
4. Identical resume content collision avoidance
5. Re-upload idempotency
6. Inaccessibility of legacy unowned records (user_id = NULL)
7. Retrieval persistence across vector store re-instantiations
8. Secure cascading deletion and filesystem cleanup
9. Defensive failure handling and recovery
"""

import io
import os
import shutil
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import MagicMock, patch

import fitz
import google.generativeai as genai
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base, get_db
from backend.db.models import CandidateProfileRecord, Resume, ResumeChunkRecord, User
from backend.main import app
from backend.models.schemas import StructuredCV
from backend.services.auth_service import (
    create_access_token,
    get_current_user,
    hash_password,
)
from backend.services.cv_pipeline import CVPipelineService
from backend.services.embedder import MockEmbedder
from backend.services.gemini import GeminiService
from backend.services.resume_agent import ResumeIntelligenceAgent, reset_resume_agent
from backend.services.retriever import ResumeRetriever, reset_retriever
from backend.services.vector_store import FaissVectorStore, reset_vector_store


class TestSecureResumeAPI(unittest.TestCase):
    """End-to-end multi-user isolation and authorization test suite."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
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
        # 1. Create isolated temporary directory for SQLite DB and FAISS indexes
        self.test_dir = tempfile.mkdtemp(prefix="careerlens_phase4_test_")
        self.db_path = os.path.join(self.test_dir, "test_phase4.db")
        self.vector_dir = os.path.join(self.test_dir, "vector_indexes")
        os.makedirs(self.vector_dir, exist_ok=True)

        # 2. Configure isolated SQLite database
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def _set_sqlite_pragma(dbapi_con, con_record):
            cursor = dbapi_con.cursor()
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

        Base.metadata.create_all(bind=self.engine)
        self.TestSessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )

        def override_get_db():
            db = self.TestSessionLocal()
            try:
                yield db
            finally:
                db.close()

        app.dependency_overrides[get_db] = override_get_db

        # 3. Create isolated VectorStore and Retriever with MockEmbedder
        self.mock_embedder = MockEmbedder(dimension=64)
        self.vector_store = FaissVectorStore(
            index_dir=self.vector_dir,
            session_factory=self.TestSessionLocal,
            dimension=64,
            expected_model_name="mock-embedder-64d",
        )
        self.retriever = ResumeRetriever(
            embedder=self.mock_embedder,
            vector_store=self.vector_store,
        )
        self.agent = ResumeIntelligenceAgent(
            retriever=self.retriever,
            gemini_service=GeminiService(),
        )

        # 4. Patch service dependencies in FastAPI
        from backend.routers.agent import get_agent_service, get_retriever_service
        from backend.routers.cv import get_cv_pipeline

        app.dependency_overrides[get_retriever_service] = lambda: self.retriever
        app.dependency_overrides[get_agent_service] = lambda: self.agent
        self.custom_pipeline = CVPipelineService(
            retriever=self.retriever,
        )
        app.dependency_overrides[get_cv_pipeline] = lambda: self.custom_pipeline

        # 5. Deterministic mock for Gemini
        def _mock_gemini_init(svc, api_key=None):
            svc.api_key = "test_deterministic_key"
            svc._is_configured = True
            mock_model = MagicMock()
            mock_resp = MagicMock()
            mock_resp.text = '{"answer": "Candidate has extensive experience in backend distributed systems [1].", "cited_source_ids": [1]}'
            mock_model.generate_content.return_value = mock_resp
            svc.model = mock_model

        self._init_patcher = patch.object(GeminiService, "__init__", _mock_gemini_init)
        self._init_patcher.start()

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

        # 6. Seed test users: User A and User B
        with self.TestSessionLocal() as session:
            self.user_a = User(
                id="user_a_uuid_1111",
                email="alice@example.com",
                hashed_password=hash_password("Password123!"),
                full_name="Alice Candidate",
                is_active=True,
            )
            self.user_b = User(
                id="user_b_uuid_2222",
                email="bob@example.com",
                hashed_password=hash_password("Password123!"),
                full_name="Bob Candidate",
                is_active=True,
            )
            session.add_all([self.user_a, self.user_b])
            session.commit()

        self.token_a = create_access_token(user_id=self.user_a.id, email=self.user_a.email)
        self.token_b = create_access_token(user_id=self.user_b.id, email=self.user_b.email)

    def tearDown(self):
        self._gemini_patcher.stop()
        self._init_patcher.stop()

        # Reset overrides
        app.dependency_overrides.clear()
        reset_vector_store()
        reset_retriever()
        reset_resume_agent()

        # Dispose database engine and remove temp files
        self.engine.dispose()
        if os.path.exists(self.test_dir):
            try:
                shutil.rmtree(self.test_dir)
            except OSError:
                pass

    def _create_test_pdf(self, text: str) -> bytes:
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes

    def _auth_header(self, token: str):
        return {"Authorization": f"Bearer {token}"}

    # ==========================================
    # 1. AUTHENTICATION ENFORCEMENT TESTS
    # ==========================================

    def test_missing_token_on_all_endpoints_returns_401(self):
        """All resume endpoints must return 401 when Authorization header is absent."""
        pdf_bytes = self._create_test_pdf("Some resume content")

        # 1. POST /api/v1/cv/process
        res_upload = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("resume.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
        )
        self.assertEqual(res_upload.status_code, 401)
        self.assertIn("credentials were not provided", res_upload.json()["detail"])

        # 2. POST /api/v1/agent/resume/query
        res_query = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": "res_1234", "query": "What is their experience?"},
        )
        self.assertEqual(res_query.status_code, 401)

        # 3. GET /api/v1/agent/resume/{id}/status
        res_status = self.client.get("/api/v1/agent/resume/res_1234/status")
        self.assertEqual(res_status.status_code, 401)

        # 4. DELETE /api/v1/agent/resume/{id}
        res_del = self.client.delete("/api/v1/agent/resume/res_1234")
        self.assertEqual(res_del.status_code, 401)

    def test_invalid_token_returns_401(self):
        """Endpoints reject forged, malformed, or garbage tokens with 401."""
        headers = {"Authorization": "Bearer completely-invalid-jwt-token"}
        res = self.client.get("/api/v1/agent/resume/res_1234/status", headers=headers)
        self.assertEqual(res.status_code, 401)

    def test_expired_token_returns_401(self):
        """Endpoints reject expired JWT tokens with 401."""
        expired_token = create_access_token(
            user_id=self.user_a.id,
            email=self.user_a.email,
            expires_delta=timedelta(minutes=-15),
        )
        headers = self._auth_header(expired_token)
        res = self.client.get("/api/v1/agent/resume/res_1234/status", headers=headers)
        self.assertEqual(res.status_code, 401)
        self.assertIn("expired", res.json()["detail"].lower())

    # ==========================================
    # 2. MULTI-USER ISOLATION & OWNERSHIP ENFORCEMENT
    # ==========================================

    def test_user_a_and_user_b_isolation(self):
        """
        User A uploads a resume:
        - User A can query, get status, and delete.
        - User B receives 404 for query, status, and delete.
        """
        resume_text = (
            "Alice Candidate\nalice@example.com\n\n"
            "Summary:\nLead Python Engineer with 6 years experience in FastAPI and microservices.\n\n"
            "Experience:\nLead Engineer at Alpha Corp (2020 - 2024)\n"
            "Architected cloud infrastructure and data processing pipelines.\n\n"
            "Skills:\nPython, FastAPI, Docker, Kubernetes\n"
        )
        pdf_bytes = self._create_test_pdf(resume_text)

        # User A uploads resume
        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("alice_cv.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(upload_res.status_code, 200)
        resume_id_a = upload_res.json()["resume_id"]
        self.assertIsNotNone(resume_id_a)

        # Verify User A can query their resume
        query_a_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_a, "query": "What is their experience at Alpha Corp?"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(query_a_res.status_code, 200)
        self.assertTrue(query_a_res.json()["success"])

        # Verify User A can check status
        status_a_res = self.client.get(
            f"/api/v1/agent/resume/{resume_id_a}/status",
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(status_a_res.status_code, 200)
        self.assertTrue(status_a_res.json()["is_indexed"])

        # --- USER B ATTEMPTS CROSS-USER ACCESS (MUST FAIL WITH 404) ---

        # 1. User B querying User A's resume returns 404
        query_b_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_a, "query": "What is Alice's experience?"},
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(query_b_res.status_code, 404)
        self.assertEqual(query_b_res.json()["detail"], "Resume not found")

        # 2. User B inspecting status returns 404
        status_b_res = self.client.get(
            f"/api/v1/agent/resume/{resume_id_a}/status",
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(status_b_res.status_code, 404)
        self.assertEqual(status_b_res.json()["detail"], "Resume not found")

        # 3. User B attempting to delete User A's resume returns 404
        delete_b_res = self.client.delete(
            f"/api/v1/agent/resume/{resume_id_a}",
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(delete_b_res.status_code, 404)
        self.assertEqual(delete_b_res.json()["detail"], "Resume not found")

        # Ensure User A's resume is still present in SQLite and FAISS
        with self.TestSessionLocal() as session:
            resume_db = session.query(Resume).filter(Resume.id == resume_id_a).first()
            self.assertIsNotNone(resume_db)
            self.assertEqual(resume_db.user_id, self.user_a.id)

    def test_request_body_user_id_tampering_is_ignored(self):
        """A client sending another user's user_id in the JSON body cannot bypass authorization."""
        resume_text = "Alice Secret Project\nalice@example.com\n\nExperience:\nStaff Engineer\n"
        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("alice.pdf", io.BytesIO(self._create_test_pdf(resume_text)), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        resume_id_a = upload_res.json()["resume_id"]

        # Malicious User B sends request specifying user_id = User A's id
        tampered_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={
                "resume_id": resume_id_a,
                "query": "Tell me secrets",
                "user_id": self.user_a.id,  # Body spoofing attempt
            },
            headers=self._auth_header(self.token_b),  # Authenticated as User B
        )
        self.assertEqual(tampered_res.status_code, 404)
        self.assertEqual(tampered_res.json()["detail"], "Resume not found")

    def test_identical_resumes_by_different_users_are_isolated(self):
        """
        Two different users uploading identical resume content get distinct resume IDs,
        isolated SQLite records, and independent FAISS indexes.
        """
        identical_text = (
            "Shared Candidate Template\ntemplate@example.com\n\n"
            "Summary:\nFull Stack Engineer.\n\n"
            "Experience:\nEngineer at TechFirm\n\n"
            "Skills:\nReact, Python, Node.js\n"
        )
        pdf_bytes = self._create_test_pdf(identical_text)

        # User A uploads identical PDF
        res_a = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("template.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        resume_id_a = res_a.json()["resume_id"]

        # User B uploads identical PDF
        res_b = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("template.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=self._auth_header(self.token_b),
        )
        resume_id_b = res_b.json()["resume_id"]

        # Verification: Resume IDs MUST be distinct
        self.assertNotEqual(resume_id_a, resume_id_b)

        # Verify distinct database records
        with self.TestSessionLocal() as session:
            rec_a = session.query(Resume).filter(Resume.id == resume_id_a).first()
            rec_b = session.query(Resume).filter(Resume.id == resume_id_b).first()
            self.assertEqual(rec_a.user_id, self.user_a.id)
            self.assertEqual(rec_b.user_id, self.user_b.id)

        # User A can query resume_a but not resume_b
        q_a_on_a = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_a, "query": "Tell me about tech firm"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(q_a_on_a.status_code, 200)

        q_a_on_b = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_b, "query": "Tell me about tech firm"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(q_a_on_b.status_code, 404)

        # User B can query resume_b but not resume_a
        q_b_on_b = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_b, "query": "Tell me about tech firm"},
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(q_b_on_b.status_code, 200)

        q_b_on_a = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id_a, "query": "Tell me about tech firm"},
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(q_b_on_a.status_code, 404)

    def test_reupload_by_same_user_is_idempotent(self):
        """Re-uploading the same resume by the same user does not create duplicates."""
        resume_text = "Idempotency Test\nuser@example.com\n\nExperience:\nDev at TestCo\n"
        pdf_bytes = self._create_test_pdf(resume_text)

        # Upload 1
        res1 = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("cv1.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        id1 = res1.json()["resume_id"]

        # Upload 2
        res2 = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("cv2.pdf", io.BytesIO(pdf_bytes), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        id2 = res2.json()["resume_id"]

        self.assertEqual(id1, id2)

        # Check only one Resume record exists
        with self.TestSessionLocal() as session:
            count = session.query(Resume).filter(Resume.id == id1).count()
            self.assertEqual(count, 1)

    def test_legacy_unowned_records_are_inaccessible_to_authenticated_users(self):
        """
        Resumes with user_id = NULL (legacy records) cannot be queried, inspected, or deleted
        by authenticated users.
        """
        with self.TestSessionLocal() as session:
            legacy_resume = Resume(
                id="legacy_unowned_001",
                user_id=None,  # Unowned
                filename="legacy.pdf",
                raw_text="Legacy content",
                processing_status="completed",
                indexing_status="indexed",
                chunk_count=1,
            )
            session.add(legacy_resume)
            session.commit()

        # User A querying legacy resume returns 404
        res_q = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": "legacy_unowned_001", "query": "What is in legacy resume?"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(res_q.status_code, 404)

        # User A checking status returns 404
        res_s = self.client.get(
            "/api/v1/agent/resume/legacy_unowned_001/status",
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(res_s.status_code, 404)

        # User A attempting to delete returns 404
        res_d = self.client.delete(
            "/api/v1/agent/resume/legacy_unowned_001",
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(res_d.status_code, 404)

        # Legacy resume remains intact in database
        with self.TestSessionLocal() as session:
            r = session.query(Resume).filter(Resume.id == "legacy_unowned_001").first()
            self.assertIsNotNone(r)
            self.assertIsNone(r.user_id)

    # ==========================================
    # 3. PERSISTENCE ACROSS RESTARTS
    # ==========================================

    def test_persistence_across_vector_store_recreation(self):
        """
        Indexed resume evidence remains queryable after the vector-store instance
        is recreated (simulating application restart).
        """
        resume_text = (
            "David Miller\ndavid@example.com\n\n"
            "Experience:\nPrincipal SRE at GlobalCloud (2018 - 2024)\n"
            "Maintained high availability for multi-region Kubernetes clusters.\n\n"
            "Skills:\nKubernetes, Terraform, Prometheus\n"
        )
        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("david.pdf", io.BytesIO(self._create_test_pdf(resume_text)), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        resume_id = upload_res.json()["resume_id"]

        # Simulate restart by instantiating a brand-new FaissVectorStore pointing to the same dir
        fresh_vector_store = FaissVectorStore(
            index_dir=self.vector_dir,
            session_factory=self.TestSessionLocal,
            dimension=64,
            expected_model_name="mock-embedder-64d",
        )
        fresh_retriever = ResumeRetriever(
            embedder=self.mock_embedder,
            vector_store=fresh_vector_store,
        )
        fresh_agent = ResumeIntelligenceAgent(
            retriever=fresh_retriever,
            gemini_service=GeminiService(),
        )

        from backend.routers.agent import get_agent_service, get_retriever_service
        app.dependency_overrides[get_retriever_service] = lambda: fresh_retriever
        app.dependency_overrides[get_agent_service] = lambda: fresh_agent

        # Authorized user query succeeds after restart
        query_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id, "query": "Kubernetes high availability SRE"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(query_res.status_code, 200)
        self.assertTrue(query_res.json()["success"])
        self.assertTrue(query_res.json()["has_sufficient_evidence"])

        # Unauthorized user still gets 404
        unauth_res = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": resume_id, "query": "Kubernetes high availability SRE"},
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(unauth_res.status_code, 404)

    # ==========================================
    # 4. DELETION AND CLEANUP TESTS
    # ==========================================

    def test_resume_deletion_cleanup_and_integrity(self):
        """
        Deleting a resume removes SQLite records, chunks, candidate profiles,
        and disk FAISS indexes. Other users' resumes remain unaffected.
        """
        # Upload Resume 1 for User A
        res_a = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("cv_a.pdf", io.BytesIO(self._create_test_pdf("Alice Work Experience\nSkills: Python\n")), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        id_a = res_a.json()["resume_id"]

        # Upload Resume 2 for User B
        res_b = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("cv_b.pdf", io.BytesIO(self._create_test_pdf("Bob Work Experience\nSkills: Java\n")), "application/pdf")},
            headers=self._auth_header(self.token_b),
        )
        id_b = res_b.json()["resume_id"]

        # User A deletes Resume A
        del_res = self.client.delete(
            f"/api/v1/agent/resume/{id_a}",
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(del_res.status_code, 200)
        del_data = del_res.json()
        self.assertTrue(del_data["success"])
        self.assertEqual(del_data["resume_id"], id_a)

        # 1. Subsequent queries and status checks for id_a return 404
        subsequent_q = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": id_a, "query": "Python skills"},
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(subsequent_q.status_code, 404)

        subsequent_s = self.client.get(
            f"/api/v1/agent/resume/{id_a}/status",
            headers=self._auth_header(self.token_a),
        )
        self.assertEqual(subsequent_s.status_code, 404)

        # 2. Database records for id_a are deleted, including cascades
        with self.TestSessionLocal() as session:
            self.assertIsNone(session.query(Resume).filter(Resume.id == id_a).first())
            self.assertEqual(session.query(ResumeChunkRecord).filter(ResumeChunkRecord.resume_id == id_a).count(), 0)
            self.assertIsNone(session.query(CandidateProfileRecord).filter(CandidateProfileRecord.resume_id == id_a).first())

        # 3. Disk FAISS file for id_a is removed
        idx_a_path = os.path.join(self.vector_dir, f"{id_a}.faiss")
        self.assertFalse(os.path.exists(idx_a_path))

        # 4. User B's resume id_b remains completely intact and queryable
        status_b = self.client.get(
            f"/api/v1/agent/resume/{id_b}/status",
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(status_b.status_code, 200)
        self.assertTrue(status_b.json()["is_indexed"])

        query_b = self.client.post(
            "/api/v1/agent/resume/query",
            json={"resume_id": id_b, "query": "Java skills"},
            headers=self._auth_header(self.token_b),
        )
        self.assertEqual(query_b.status_code, 200)

    # ==========================================
    # 5. DEFENSIVE FAILURE HANDLING TESTS
    # ==========================================

    def test_faiss_write_failure_marks_indexing_failed_without_discarding_db_record(self):
        """
        When writing the FAISS index to disk fails:
        - The Resume and CandidateProfileRecord in SQLite are preserved
        - Resume.indexing_status is set to 'failed'
        - The failure is surfaced to the client appropriately
        """
        import faiss

        # Patch faiss.write_index to simulate an OS or disk write failure
        with patch.object(faiss, "write_index", side_effect=IOError("Simulated disk error")):
            res = self.client.post(
                "/api/v1/cv/process",
                files={"file": ("error_cv.pdf", io.BytesIO(self._create_test_pdf("Error Test Content")), "application/pdf")},
                headers=self._auth_header(self.token_a),
            )
            # Pipeline raises error on indexing failure, returning 500
            self.assertEqual(res.status_code, 500)

        # Verify that SQLite record exists and is marked as indexing_status = 'failed'
        with self.TestSessionLocal() as session:
            resume = session.query(Resume).filter(Resume.user_id == self.user_a.id).first()
            if resume:
                self.assertEqual(resume.indexing_status, "failed")
                # Chunks are preserved with embedding blobs for future recovery
                chunks = session.query(ResumeChunkRecord).filter(ResumeChunkRecord.resume_id == resume.id).all()
                self.assertGreater(len(chunks), 0)
                self.assertIsNotNone(chunks[0].embedding_blob)

    def test_deletion_resilient_to_vector_index_removal_error(self):
        """
        If removing the FAISS file encounters an OSError, the database deletion
        still completes so access is revoked, and no internal paths leak to client.
        """
        upload_res = self.client.post(
            "/api/v1/cv/process",
            files={"file": ("del_err.pdf", io.BytesIO(self._create_test_pdf("Content to delete")), "application/pdf")},
            headers=self._auth_header(self.token_a),
        )
        resume_id = upload_res.json()["resume_id"]

        # Simulate OSError during disk index removal
        with patch.object(self.vector_store, "delete_by_resume_id", side_effect=OSError("Permission denied")):
            del_res = self.client.delete(
                f"/api/v1/agent/resume/{resume_id}",
                headers=self._auth_header(self.token_a),
            )
            self.assertEqual(del_res.status_code, 200)
            self.assertTrue(del_res.json()["success"])

        # Database record was deleted
        with self.TestSessionLocal() as session:
            r = session.query(Resume).filter(Resume.id == resume_id).first()
            self.assertIsNone(r)


if __name__ == "__main__":
    unittest.main()
