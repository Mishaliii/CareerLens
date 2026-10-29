"""
Unit and integration tests for CareerLens persistent database storage.

Validates:
1. Table creation and migration execution.
2. SQLite foreign key enforcement (`PRAGMA foreign_keys = ON`).
3. Email and unique constraint enforcement.
4. Cascading deletion (User -> Resumes -> Profiles & Chunks).
5. Transaction rollback integrity.
6. Embedding blob validation, serialization, and deserialization.
7. Processing and indexing lifecycle status tracking.
"""

import math
import os
import tempfile
import unittest
from sqlalchemy import create_engine, event
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base
from backend.db.migrations import get_applied_migrations, run_migrations
from backend.db.models import (
    CandidateProfileRecord,
    Resume,
    ResumeChunkRecord,
    SchemaMigration,
    User,
)


class TestDatabaseStorage(unittest.TestCase):
    """Test suite for persistent relational models and SQLite configuration."""

    def setUp(self):
        # Create an isolated temporary SQLite file database for each test run
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db_url = f"sqlite:///{self.temp_db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
        )

        # Enforce foreign keys on the test engine
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.close()

        self.Session = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )

        # Run migrations to initialize schema
        run_migrations(self.engine)
        self.session = self.Session()

    def tearDown(self):
        self.session.close()
        self.engine.dispose()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except PermissionError:
                pass

    def test_migration_and_table_creation(self):
        """Verify tables are created and initial migration is recorded."""
        applied = get_applied_migrations(self.engine)
        self.assertTrue(any(v == "001_initial_schema" for v, _ in applied))

        # Re-running migrations should be idempotent
        second_run = run_migrations(self.engine)
        self.assertEqual(len(second_run), 0)

    def test_user_creation_and_email_uniqueness(self):
        """User creation succeeds and rejects duplicate email addresses."""
        user1 = User(
            id="user_test_001",
            email="candidate@example.com",
            hashed_password="bcrypt$fake_hash_value_here",
            full_name="Jane Doe",
        )
        self.session.add(user1)
        self.session.commit()

        # Attempt to insert user with identical email
        user2 = User(
            id="user_test_002",
            email="candidate@example.com",
            hashed_password="bcrypt$another_hash_value",
            full_name="Duplicate Email Candidate",
        )
        self.session.add(user2)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_foreign_key_enforcement_on_resume(self):
        """Inserting a resume referencing a non-existent user must fail."""
        resume = Resume(
            id="res_invalid_user",
            user_id="non_existent_user_id",
            filename="my_cv.pdf",
            processing_status="pending",
        )
        self.session.add(resume)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_foreign_key_enforcement_on_chunks(self):
        """Inserting a chunk with invalid resume_id or user_id must fail."""
        user = User(
            id="user_valid_001",
            email="valid@example.com",
            hashed_password="hash",
        )
        self.session.add(user)
        self.session.commit()

        chunk = ResumeChunkRecord(
            id="chunk_invalid_res",
            resume_id="non_existent_resume",
            user_id=user.id,
            chunk_index=0,
            section="summary",
            content="Summary content",
            metadata_json={},
        )
        self.session.add(chunk)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_candidate_profile_uniqueness_per_resume(self):
        """Only one CandidateProfileRecord can exist for a given resume_id."""
        user = User(id="user_profile_test", email="user_p@example.com", hashed_password="h")
        resume = Resume(id="res_profile_test", user_id=user.id, filename="cv.pdf")
        self.session.add_all([user, resume])
        self.session.commit()

        profile1 = CandidateProfileRecord(
            resume_id=resume.id,
            user_id=user.id,
            profile_json={"name": "Alice"},
            doc_analysis_json={"pages": 1},
        )
        self.session.add(profile1)
        self.session.commit()

        profile2 = CandidateProfileRecord(
            resume_id=resume.id,
            user_id=user.id,
            profile_json={"name": "Alice Duplicate"},
            doc_analysis_json={"pages": 1},
        )
        self.session.add(profile2)
        with self.assertRaises(IntegrityError):
            self.session.commit()
        self.session.rollback()

    def test_cascading_deletion_from_user(self):
        """Deleting a user automatically cascades and deletes their resumes, profiles, and chunks."""
        user = User(id="user_cascade_all", email="cascade@example.com", hashed_password="h")
        resume = Resume(id="res_cascade", user_id=user.id, filename="resume.pdf")
        profile = CandidateProfileRecord(
            resume_id=resume.id,
            user_id=user.id,
            profile_json={"skills": ["Python"]},
            doc_analysis_json={"pages": 2},
        )
        chunk1 = ResumeChunkRecord(
            id="chunk_c1",
            resume_id=resume.id,
            user_id=user.id,
            chunk_index=0,
            section="skills",
            content="Python",
            metadata_json={},
        )
        chunk2 = ResumeChunkRecord(
            id="chunk_c2",
            resume_id=resume.id,
            user_id=user.id,
            chunk_index=1,
            section="experience",
            content="Developer",
            metadata_json={},
        )

        self.session.add_all([user, resume, profile, chunk1, chunk2])
        self.session.commit()

        # Delete user
        self.session.delete(user)
        self.session.commit()

        # Confirm all downstream children were purged
        self.assertIsNone(self.session.query(Resume).filter_by(id="res_cascade").first())
        self.assertIsNone(self.session.query(CandidateProfileRecord).filter_by(resume_id="res_cascade").first())
        self.assertEqual(self.session.query(ResumeChunkRecord).filter_by(resume_id="res_cascade").count(), 0)

    def test_cascading_deletion_from_resume(self):
        """Deleting a resume removes profile and chunks, but leaves user intact."""
        user = User(id="user_keep", email="keep@example.com", hashed_password="h")
        resume = Resume(id="res_delete_me", user_id=user.id, filename="cv.pdf")
        chunk = ResumeChunkRecord(
            id="chunk_del_me",
            resume_id=resume.id,
            user_id=user.id,
            chunk_index=0,
            section="summary",
            content="Summary",
            metadata_json={},
        )
        self.session.add_all([user, resume, chunk])
        self.session.commit()

        # Delete resume
        self.session.delete(resume)
        self.session.commit()

        # User remains
        self.assertIsNotNone(self.session.query(User).filter_by(id="user_keep").first())
        # Chunk is removed
        self.assertIsNone(self.session.query(ResumeChunkRecord).filter_by(id="chunk_del_me").first())

    def test_transaction_rollback(self):
        """Failed transactions roll back completely, preventing partial writes."""
        user = User(id="user_rb", email="rb@example.com", hashed_password="h")
        self.session.add(user)
        self.session.commit()

        try:
            # Add valid resume and an invalid chunk in the same transaction
            resume = Resume(id="res_rb", user_id=user.id, filename="test.pdf")
            self.session.add(resume)
            invalid_chunk = ResumeChunkRecord(
                id="chunk_bad",
                resume_id="non_existent_parent",
                user_id=user.id,
                chunk_index=0,
                section="summary",
                content="text",
                metadata_json={},
            )
            self.session.add(invalid_chunk)
            self.session.commit()
        except IntegrityError:
            self.session.rollback()

        # The valid resume in the aborted transaction must NOT exist
        self.assertIsNone(self.session.query(Resume).filter_by(id="res_rb").first())

    def test_embedding_blob_serialization_and_validation(self):
        """Verify vector serialization, dimension enforcement, and non-finite rejection."""
        user = User(id="user_emb", email="emb@example.com", hashed_password="h")
        resume = Resume(id="res_emb", user_id=user.id, filename="emb.pdf")
        chunk = ResumeChunkRecord(
            id="chunk_emb_1",
            resume_id=resume.id,
            user_id=user.id,
            chunk_index=0,
            section="summary",
            content="FastAPI ML pipeline",
            metadata_json={},
        )
        self.session.add_all([user, resume, chunk])
        self.session.commit()

        # 1. Valid 384-dimensional vector
        valid_vec = [float(i) * 0.001 for i in range(384)]
        chunk.set_embedding(valid_vec, expected_dimension=384)
        self.session.commit()

        # Verify retrieval and float reconstruction
        loaded_chunk = self.session.query(ResumeChunkRecord).filter_by(id="chunk_emb_1").first()
        retrieved_vec = loaded_chunk.get_embedding()
        self.assertEqual(len(retrieved_vec), 384)
        for original, retrieved in zip(valid_vec, retrieved_vec):
            self.assertAlmostEqual(original, retrieved, places=5)

        # 2. Reject dimension mismatch
        with self.assertRaises(ValueError):
            chunk.set_embedding([1.0, 2.0], expected_dimension=384)

        # 3. Reject empty vector
        with self.assertRaises(ValueError):
            chunk.set_embedding([])

        # 4. Reject NaN / Inf values
        with self.assertRaises(ValueError):
            chunk.set_embedding([float("nan")] * 384)

        with self.assertRaises(ValueError):
            chunk.set_embedding([float("inf")] * 384)

    def test_status_tracking_and_error_capture(self):
        """Test processing and indexing status fields and failure recovery attributes."""
        user = User(id="user_status", email="status@example.com", hashed_password="h")
        resume = Resume(
            id="res_status_1",
            user_id=user.id,
            filename="cv.pdf",
            processing_status="pending",
            indexing_status="unindexed",
        )
        self.session.add_all([user, resume])
        self.session.commit()

        # Simulate failure during extraction
        resume.processing_status = "failed"
        resume.error_message = "Corrupted font dictionary in PDF page 2"
        self.session.commit()

        loaded = self.session.query(Resume).filter_by(id="res_status_1").first()
        self.assertEqual(loaded.processing_status, "failed")
        self.assertIn("Corrupted font dictionary", loaded.error_message)

        # Simulate recovery
        loaded.processing_status = "completed"
        loaded.indexing_status = "indexed"
        loaded.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
        loaded.embedding_dimension = 384
        loaded.chunk_count = 12
        loaded.error_message = None
        self.session.commit()

        verified = self.session.query(Resume).filter_by(id="res_status_1").first()
        self.assertEqual(verified.processing_status, "completed")
        self.assertEqual(verified.indexing_status, "indexed")
        self.assertEqual(verified.chunk_count, 12)
        self.assertIsNone(verified.error_message)


if __name__ == "__main__":
    unittest.main()
