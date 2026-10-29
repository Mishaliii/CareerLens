"""
Comprehensive integration and security tests for POST /api/v1/jobs/match endpoint.

Verifies:
1. Authentication requirements:
   - 401 Unauthorized when Authorization header is missing.
   - 401 Unauthorized with malformed or invalid Bearer token.
   - 401 Unauthorized with expired Bearer token.
2. Multi-user isolation & ownership enforcement:
   - User B attempting to match against User A's resume receives 404 Not Found (never 403).
   - Inexistent resume_id returns 404 Not Found.
3. Input validation:
   - Job description < 20 characters returns 422 Unprocessable Entity.
   - Whitespace-only job description returns 422 Unprocessable Entity.
   - Blank or invalid resume_id returns 422 Unprocessable Entity.
4. End-to-end matching workflow:
   - Successfully matches candidate profile against a job description.
   - Returns 200 OK with complete explainable response payload.
   - Preserves hints (job_title_hint, company_hint).
"""

from datetime import timedelta
import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base, get_db
from backend.db.models import CandidateProfileRecord, Resume, ResumeChunkRecord, User
from backend.main import app
from backend.models.schemas import CandidateProfile, EducationItem, ExperienceItem, StructuredCV
from backend.services.auth_service import create_access_token, hash_password
from backend.services.gemini import GeminiService
from backend.services.job_matcher import JobMatcherService, reset_job_matcher
from backend.services.job_parser import JobParserService, reset_job_parser
from backend.services.skill_ontology import SkillOntology


class TestJobMatchAPI(unittest.TestCase):
    """Integration test suite for Job Matching API endpoint."""

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        reset_job_matcher()
        reset_job_parser()

        # 1. Setup isolated temporary directory for SQLite
        self.test_dir = tempfile.mkdtemp(prefix="careerlens_job_match_test_")
        self.db_path = os.path.join(self.test_dir, "test_jobs.db")

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

        # 2. Mock Gemini unconfigured to prevent network calls
        self._unconfigured_gemini = GeminiService.__new__(GeminiService)
        self._unconfigured_gemini.api_key = None
        self._unconfigured_gemini._is_configured = False

        from backend.routers.jobs import get_job_matcher_service, get_job_parser_service

        self.ontology = SkillOntology()
        self.parser_service = JobParserService(
            skill_ontology=self.ontology,
            gemini_service=self._unconfigured_gemini,
        )
        self.matcher_service = JobMatcherService(
            skill_ontology=self.ontology,
        )

        app.dependency_overrides[get_job_parser_service] = lambda: self.parser_service
        app.dependency_overrides[get_job_matcher_service] = lambda: self.matcher_service

        # 3. Seed User A and User B
        with self.TestSessionLocal() as session:
            self.user_a = User(
                id="user_a_uuid_1001",
                email="alice@example.com",
                hashed_password=hash_password("Password123!"),
                full_name="Alice Engineer",
                is_active=True,
            )
            self.user_b = User(
                id="user_b_uuid_2002",
                email="bob@example.com",
                hashed_password=hash_password("Password123!"),
                full_name="Bob Engineer",
                is_active=True,
            )
            session.add_all([self.user_a, self.user_b])
            session.commit()

            # Seed a resume for User A
            self.resume_a = Resume(
                id="resume_a_uuid_1001",
                user_id=self.user_a.id,
                filename="alice_resume.pdf",
                raw_text="Alice Engineer. Python, FastAPI, PostgreSQL, Docker.",
                processing_status="completed",
                indexing_status="indexed",
                chunk_count=1,
            )
            session.add(self.resume_a)
            session.commit()

            # Seed Candidate Profile Record for User A
            alice_profile = CandidateProfile(
                name="Alice Engineer",
                email="alice@example.com",
                skills="Python, FastAPI, PostgreSQL, Docker",
                experience="Backend Software Engineer at DataCloud Inc (2021 - 2024)",
                education="Bachelor of Science in Computer Science from Tech University",
                structured_data=StructuredCV(
                    full_name="Alice Engineer",
                    email="alice@example.com",
                    skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
                    experience=[
                        ExperienceItem(
                            title="Backend Software Engineer",
                            company="DataCloud Inc",
                            start_date="2021-01",
                            end_date="2024-01",
                            is_current=False,
                            description="Developed microservices with Python, FastAPI, and PostgreSQL.",
                        )
                    ],
                    education=[
                        EducationItem(
                            degree="Bachelor of Science",
                            field="Computer Science",
                            institution="Tech University",
                        )
                    ],
                ),
            )
            profile_dict = json.loads(alice_profile.model_dump_json())

            self.profile_record_a = CandidateProfileRecord(
                id="profile_a_uuid_1001",
                resume_id=self.resume_a.id,
                user_id=self.user_a.id,
                profile_json=profile_dict,
                doc_analysis_json={"page_count": 1, "language": "en"},
            )
            session.add(self.profile_record_a)

            # Seed Resume Chunk Records for User A
            self.chunk_record_a = ResumeChunkRecord(
                id="chunk_a_uuid_1001",
                resume_id=self.resume_a.id,
                user_id=self.user_a.id,
                chunk_index=0,
                section="skills",
                item_title="Core Skills",
                content="Python, FastAPI, PostgreSQL, Docker containerization.",
                metadata_json={},
            )
            session.add(self.chunk_record_a)
            session.commit()

        # Auth tokens
        self.token_a = create_access_token(user_id=self.user_a.id, email=self.user_a.email)
        self.token_b = create_access_token(user_id=self.user_b.id, email=self.user_b.email)

    def tearDown(self):
        app.dependency_overrides.clear()
        self.engine.dispose()
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # 1. Authentication Tests
    # =========================================================================

    def test_missing_auth_header_returns_401(self):
        """Request without Authorization header is rejected with 401."""
        response = self.client.post(
            "/api/v1/jobs/match",
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "We are seeking a Python and FastAPI developer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 401)
        self.assertIn("detail", response.json())

    def test_invalid_bearer_token_returns_401(self):
        """Malformed or forged Bearer token returns 401."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": "Bearer invalid.token.value"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "We are seeking a Python and FastAPI developer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 401)

    def test_expired_bearer_token_returns_401(self):
        """Expired Bearer token returns 401."""
        expired_token = create_access_token(
            user_id=self.user_a.id,
            email=self.user_a.email,
            expires_delta=timedelta(seconds=-60),
        )
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {expired_token}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "We are seeking a Python and FastAPI developer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 401)

    # =========================================================================
    # 2. Multi-User Isolation and Ownership Tests
    # =========================================================================

    def test_user_b_cannot_match_user_a_resume_returns_404(self):
        """User B querying User A's resume receives 404 (ownership isolation)."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_b}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "Looking for a Python and FastAPI Engineer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Resume not found")

    def test_non_existent_resume_returns_404(self):
        """Querying a non-existent resume ID returns 404."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "non_existent_resume_uuid_9999",
                "job_description": "Looking for a Python and FastAPI Engineer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["detail"], "Resume not found")

    # =========================================================================
    # 3. Input Validation Tests
    # =========================================================================

    def test_job_description_too_short_returns_422(self):
        """Job description with fewer than 20 characters is rejected with 422."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "Short text",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_job_description_whitespace_only_returns_422(self):
        """Job description consisting of only whitespace is rejected with 422."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": "                          ",
            },
        )
        self.assertEqual(response.status_code, 422)

    def test_resume_id_blank_returns_422(self):
        """Blank resume ID is rejected with 422."""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "   ",
                "job_description": "Looking for a Python and FastAPI Engineer with 2+ years experience.",
            },
        )
        self.assertEqual(response.status_code, 422)

    # =========================================================================
    # 4. Successful End-to-End Match Workflow
    # =========================================================================

    def test_successful_job_match_workflow(self):
        """User A performs matching against a valid job description and receives explainable results."""
        jd_text = """
Job Title: Senior Backend Engineer
Company: CloudScale Systems

Responsibilities:
• Develop scalable backend APIs with Python and FastAPI.
• Maintain database queries and migrations in PostgreSQL.

Requirements:
• 3+ years of professional software engineering experience.
• Strong hands-on proficiency with Python, FastAPI, and PostgreSQL.
• Bachelor's degree in Computer Science or related field.

Preferred Qualifications:
• Experience with Flask and Kubernetes.
• Familiarity with React.
"""
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": jd_text,
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()

        # 1. High-level structure
        self.assertTrue(data["has_sufficient_evidence"])
        self.assertGreater(data["overall_score"], 0.0)
        self.assertLessEqual(data["overall_score"], 100.0)
        self.assertIn("category_scores", data)
        self.assertIn("critical_gaps", data)
        self.assertIn("contextual_evidence", data)
        self.assertIn("uncertain_requirements", data)
        self.assertIn("scoring_explanation", data)
        self.assertIn("limitations", data)
        self.assertIn("parsed_job", data)

        # 2. Parsed Job verification
        parsed_job = data["parsed_job"]
        self.assertEqual(parsed_job["job_title"], "Senior Backend Engineer")
        self.assertEqual(parsed_job["company"], "CloudScale Systems")
        self.assertEqual(parsed_job["minimum_experience_years"], 3.0)
        self.assertIn("Python", parsed_job["required_skills"])
        self.assertIn("FastAPI", parsed_job["required_skills"])
        self.assertIn("PostgreSQL", parsed_job["required_skills"])
        self.assertIn("Flask", parsed_job["preferred_skills"])
        self.assertIn("Kubernetes", parsed_job["preferred_skills"])
        self.assertIn("React", parsed_job["preferred_skills"])

        # 3. Category scores
        cat_scores = data["category_scores"]
        self.assertIn("required_skills", cat_scores)
        self.assertIn("experience", cat_scores)
        self.assertIn("education", cat_scores)
        self.assertIn("preferred_skills", cat_scores)

        # Alice has Python, FastAPI, PostgreSQL (100% of required skills)
        self.assertEqual(cat_scores["required_skills"]["score"], 100.0)

        # 4. Matched vs missing requirements
        matched_reqs = [m["requirement"] for m in data["matched_requirements"]]
        self.assertIn("Python", matched_reqs)
        self.assertIn("FastAPI", matched_reqs)
        self.assertIn("PostgreSQL", matched_reqs)

        # Partially demonstrated: Flask (due to FastAPI framework transferability)
        partial_reqs = [m["requirement"] for m in data["partially_demonstrated_requirements"]]
        self.assertIn("Flask", partial_reqs)

        # Contextual evidence without substitution credit: Kubernetes (due to Docker)
        contextual_reqs = [m["requirement"] for m in data["contextual_evidence"]]
        self.assertIn("Kubernetes", contextual_reqs)

        # Missing requirements: React (no frontend skills) and Kubernetes (0.0 substitution credit)
        missing_reqs = [m["requirement"] for m in data["missing_requirements"]]
        self.assertIn("React", missing_reqs)
        self.assertIn("Kubernetes", missing_reqs)

        # Alice meets all mandatory requirements -> critical_gaps is empty
        self.assertEqual(len(data["critical_gaps"]), 0)

        # 5. Career gap recommendations
        recs = data["recommendations"]
        self.assertGreater(len(recs), 0)
        react_rec = next((r for r in recs if r["requirement"] == "React"), None)
        self.assertIsNotNone(react_rec)
        self.assertEqual(react_rec["gap_type"], "skill_gap")
        self.assertEqual(react_rec["action_type"], "learning_and_practice")
        self.assertIn("does not guarantee employment", react_rec["suggested_action"])

    def test_job_hints_override_extracted_meta(self):
        """User hints take precedence over parsed title and company."""
        jd_text = "We are seeking a Python engineer with 2+ years experience."
        response = self.client.post(
            "/api/v1/jobs/match",
            headers={"Authorization": f"Bearer {self.token_a}"},
            json={
                "resume_id": "resume_a_uuid_1001",
                "job_description": jd_text,
                "job_title": "Lead AI Engineer",
                "company": "CareerLens Inc",
            },
        )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["parsed_job"]["job_title"], "Lead AI Engineer")
        self.assertEqual(data["parsed_job"]["company"], "CareerLens Inc")


if __name__ == "__main__":
    unittest.main()
