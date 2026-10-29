"""
Unit and integration tests for CareerLens persistence portability, database configuration,
migration idempotency, and vector reconstruction capabilities (Phase 1).

Test coverage:
1. Local SQLite initialization and default configuration.
2. Missing or invalid production database configuration enforcement.
3. PostgreSQL URL normalization and connection pool configuration.
4. Full PostgreSQL DDL compilation for all SQLAlchemy models.
5. Repeatable migration behavior and idempotency.
6. Persistent vector embedding records (IEEE 754 float32 serialization).
7. Deterministic FAISS reconstruction from database records when local index files are missing.
8. Handling of corrupted or dimension-mismatched local vector index files.
9. Resilient operation when local disk cache is unwritable (ephemeral / read-only container filesystems).
10. Resume update and deletion lifecycles (stale index elimination).
11. Multi-user vector and database record isolation.
"""

import os
import shutil
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from sqlalchemy import create_engine, event, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import sessionmaker
from sqlalchemy.schema import CreateTable

from backend.db.database import (
    Base,
    DatabaseConfigurationError,
    DatabaseConnectionError,
    create_app_engine,
    resolve_database_url,
    verify_database_connectivity,
)
from backend.db.migrations import get_applied_migrations, run_migrations
from backend.db.models import (
    CandidateProfileRecord,
    Resume,
    ResumeChunkRecord,
    SchemaMigration,
    User,
)
from backend.models.schemas import ResumeChunk
from backend.services.vector_store import FaissVectorStore


class TestDatabasePortability(unittest.TestCase):
    """Tests for database URL resolution, driver configuration, and production safety."""

    def test_local_sqlite_default_in_development(self):
        """In development/test, omitting DATABASE_URL safely defaults to local SQLite."""
        resolved = resolve_database_url(None, environment="development")
        self.assertTrue(resolved.startswith("sqlite:///"))
        self.assertIn("careerlens.db", resolved)

        # Empty string also defaults to SQLite in development
        resolved_empty = resolve_database_url("", environment="development")
        self.assertTrue(resolved_empty.startswith("sqlite:///"))

    def test_production_missing_database_url_raises(self):
        """In production mode, missing or empty DATABASE_URL must raise DatabaseConfigurationError."""
        with self.assertRaises(DatabaseConfigurationError) as ctx:
            resolve_database_url(None, environment="production")
        self.assertIn("DATABASE_URL environment variable is required in production", str(ctx.exception))

        with self.assertRaises(DatabaseConfigurationError) as ctx:
            resolve_database_url("   ", environment="production")
        self.assertIn("DATABASE_URL environment variable is required in production", str(ctx.exception))

    def test_postgresql_url_normalization(self):
        """Normalizes 'postgres://' and 'postgresql://' to explicit 'postgresql+psycopg2://'."""
        neon_url = "postgres://user:secret123@ep-test-123456.us-east-2.aws.neon.tech/neondb?sslmode=require"
        normalized = resolve_database_url(neon_url, environment="production")
        self.assertTrue(normalized.startswith("postgresql+psycopg2://"))
        self.assertNotIn("postgres://", normalized)

        standard_pg = "postgresql://user:pass@localhost:5432/careerlens"
        normalized_std = resolve_database_url(standard_pg, environment="production")
        self.assertTrue(normalized_std.startswith("postgresql+psycopg2://"))

    def test_unsupported_database_scheme_rejected(self):
        """Unsupported database schemes (e.g. MySQL, MongoDB) are rejected with clear errors."""
        with self.assertRaises(DatabaseConfigurationError) as ctx:
            resolve_database_url("mysql://root:pass@localhost:3306/db", environment="development")
        self.assertIn("Unsupported database scheme", str(ctx.exception))

        with self.assertRaises(DatabaseConfigurationError) as ctx:
            resolve_database_url("mongodb://localhost:27017/db", environment="development")
        self.assertIn("Unsupported database scheme", str(ctx.exception))

    def test_postgresql_engine_configuration(self):
        """PostgreSQL engine is configured with pool settings and does NOT attach SQLite pragmas."""
        pg_url = "postgresql+psycopg2://user:pass@localhost:5432/careerlens"
        with patch("backend.db.database.create_engine") as mock_create_engine:
            mock_engine = MagicMock()
            mock_create_engine.return_value = mock_engine

            eng = create_app_engine(pg_url, environment="production")
            self.assertEqual(eng, mock_engine)
            mock_create_engine.assert_called_once()
            call_kwargs = mock_create_engine.call_args[1]

            # Verify pool configuration
            self.assertTrue(call_kwargs.get("pool_pre_ping"))
            self.assertIn("pool_size", call_kwargs)
            self.assertIn("max_overflow", call_kwargs)
            self.assertIn("pool_recycle", call_kwargs)
            # connect_args should not contain check_same_thread for postgres
            self.assertNotIn("check_same_thread", call_kwargs.get("connect_args", {}))

    def test_postgresql_ddl_compilation_all_models(self):
        """Verify all SQLAlchemy models compile valid DDL statements using PostgreSQL dialect."""
        pg_dialect = postgresql.dialect()
        tables = [
            User.__table__,
            Resume.__table__,
            CandidateProfileRecord.__table__,
            ResumeChunkRecord.__table__,
            SchemaMigration.__table__,
        ]

        for table in tables:
            compiled = str(CreateTable(table).compile(dialect=pg_dialect))
            self.assertIn(f"CREATE TABLE {table.name}", compiled)
            # Check specific PostgreSQL data types
            if table.name == "candidate_profiles":
                self.assertIn("JSON", compiled)
            elif table.name == "resume_chunks":
                self.assertIn("BYTEA", compiled)
            elif table.name == "users":
                self.assertIn("TIMESTAMP WITH TIME ZONE", compiled)

    def test_connectivity_verification_failure_handling(self):
        """Database connectivity failure raises DatabaseConnectionError and does not fallback to SQLite."""
        mock_broken_engine = MagicMock()
        mock_broken_engine.url.drivername = "psycopg2"
        mock_broken_engine.connect.side_effect = RuntimeError("Connection timed out to remote host")

        with self.assertRaises(DatabaseConnectionError) as ctx:
            verify_database_connectivity(target_engine=mock_broken_engine)
        self.assertIn("Failed to connect to database (psycopg2)", str(ctx.exception))


class TestSchemaMigrationsAndPersistence(unittest.TestCase):
    """Tests for migrations, repeatability, and persistent vector records."""

    def setUp(self):
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db_url = f"sqlite:///{self.temp_db_path}"
        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False})
        self.Session = sessionmaker(bind=self.engine)

    def tearDown(self):
        self.engine.dispose()
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except OSError:
                pass

    def test_migration_fresh_and_repeatable_startup(self):
        """Fresh database creates schema; second run is idempotent without modifying existing data."""
        # 1. Fresh database initialization
        applied_first = run_migrations(self.engine)
        self.assertIn("001_initial_schema", applied_first)

        with self.Session() as session:
            # Insert a record
            user = User(id="user_mig_1", email="mig@example.com", hashed_password="h")
            session.add(user)
            session.commit()

        # 2. Second run on existing database
        applied_second = run_migrations(self.engine)
        self.assertEqual(len(applied_second), 0)

        # Existing record is completely intact
        with self.Session() as session:
            saved_user = session.query(User).filter_by(id="user_mig_1").first()
            self.assertIsNotNone(saved_user)
            self.assertEqual(saved_user.email, "mig@example.com")


class TestVectorPersistenceAndDeterministicReconstruction(unittest.TestCase):
    """Tests verifying that database embedding records are the durable source of truth."""

    def setUp(self):
        self.temp_index_dir = tempfile.mkdtemp(prefix="careerlens_portability_faiss_")
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db_url = f"sqlite:///{self.temp_db_path}"
        self.engine = create_engine(self.db_url, connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(bind=self.engine)

        run_migrations(self.engine)

        self.dim = 32
        self.model_name = "test-model-32d"
        self.store = FaissVectorStore(
            index_dir=self.temp_index_dir,
            dimension=self.dim,
            expected_model_name=self.model_name,
            session_factory=self.SessionLocal,
        )

    def tearDown(self):
        self.store.clear()
        self.engine.dispose()
        if os.path.exists(self.temp_index_dir):
            shutil.rmtree(self.temp_index_dir, ignore_errors=True)
        if os.path.exists(self.temp_db_path):
            try:
                os.remove(self.temp_db_path)
            except OSError:
                pass

    def _sample_vector(self, seed: int) -> list[float]:
        rng = np.random.RandomState(seed)
        v = rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v /= norm
        return v.tolist()

    def test_reconstruct_faiss_when_local_index_missing(self):
        """When disk index is deleted or missing, search reconstructs index directly from DB."""
        resume_id = "res_missing_file_test"
        chunks = [
            ResumeChunk(chunk_id="c_0", resume_id=resume_id, section="skills", content="Python FastAPI"),
            ResumeChunk(chunk_id="c_1", resume_id=resume_id, section="experience", content="Backend Engineer"),
        ]
        embeddings = [self._sample_vector(10), self._sample_vector(20)]

        # Index resume
        self.store.add_chunks(chunks, embeddings)

        idx_path = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")
        self.assertTrue(os.path.exists(idx_path))

        # Clear RAM and purge disk index file
        self.store._indexes.clear()
        self.store._chunks_cache.clear()
        os.remove(idx_path)
        self.assertFalse(os.path.exists(idx_path))

        # Search should transparently rebuild index from DB records
        results = self.store.search(resume_id, embeddings[0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].chunk_id, "c_0")
        self.assertAlmostEqual(results[0][1], 1.0, places=4)

        # Index file is recreated in cache directory
        self.assertTrue(os.path.exists(idx_path))

    def test_reconstruct_faiss_when_local_index_corrupt(self):
        """When disk index is corrupted, search discards corrupt file and recovers from DB."""
        resume_id = "res_corrupt_test"
        chunks = [
            ResumeChunk(chunk_id="cc_0", resume_id=resume_id, section="summary", content="Lead AI Architect"),
        ]
        embeddings = [self._sample_vector(42)]
        self.store.add_chunks(chunks, embeddings)

        idx_path = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")

        # Corrupt disk index
        self.store._indexes.clear()
        self.store._chunks_cache.clear()
        with open(idx_path, "wb") as f:
            f.write(b"TRUNCATED_OR_GARBAGE_BINARY_DATA")

        # Search detects corrupt file and rebuilds from DB
        results = self.store.search(resume_id, embeddings[0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].chunk_id, "cc_0")

    def test_reconstruct_faiss_when_local_index_dimension_mismatched(self):
        """When disk index has wrong vector dimension, it is discarded and rebuilt from DB."""
        resume_id = "res_dim_mismatch_test"
        chunks = [
            ResumeChunk(chunk_id="dim_0", resume_id=resume_id, section="summary", content="Engineer"),
        ]
        embeddings = [self._sample_vector(99)]
        self.store.add_chunks(chunks, embeddings)

        # Replace disk index with a 16-dimension FAISS index
        idx_path = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")
        self.store._indexes.clear()
        self.store._chunks_cache.clear()

        import faiss
        wrong_idx = faiss.IndexFlatIP(16)
        wrong_vec = np.random.randn(1, 16).astype(np.float32)
        wrong_idx.add(wrong_vec)
        faiss.write_index(wrong_idx, idx_path)

        # Search checks dimension (32 vs 16), detects mismatch, and rebuilds 32-dim index from DB
        results = self.store.search(resume_id, embeddings[0], top_k=1)
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0][0].chunk_id, "dim_0")

    def test_rebuild_works_when_disk_cache_is_unwritable(self):
        """Rebuilding operates entirely in-memory if disk writes fail (ephemeral/read-only FS)."""
        resume_id = "res_readonly_test"
        chunks = [
            ResumeChunk(chunk_id="ro_0", resume_id=resume_id, section="projects", content="Serverless RAG"),
        ]
        embeddings = [self._sample_vector(77)]
        self.store.add_chunks(chunks, embeddings)

        # Clear RAM and delete file
        self.store._indexes.clear()
        self.store._chunks_cache.clear()
        idx_path = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")
        if os.path.exists(idx_path):
            os.remove(idx_path)

        # Mock faiss.write_index to raise PermissionError (simulating read-only disk)
        with patch("faiss.write_index", side_effect=OSError("Read-only filesystem")):
            results = self.store.search(resume_id, embeddings[0], top_k=1)
            # Search still succeeds using in-memory index!
            self.assertEqual(len(results), 1)
            self.assertEqual(results[0][0].chunk_id, "ro_0")
            self.assertTrue(self.store.is_indexed(resume_id))

    def test_resume_update_and_deletion_lifecycle(self):
        """Re-indexing replaces chunks; deleting resume purges DB and cache completely."""
        resume_id = "res_lifecycle"
        chunks_v1 = [ResumeChunk(chunk_id="v1_0", resume_id=resume_id, section="summary", content="Old Content")]
        emb_v1 = [self._sample_vector(1)]
        self.store.add_chunks(chunks_v1, emb_v1)

        # Re-index with v2
        chunks_v2 = [ResumeChunk(chunk_id="v2_0", resume_id=resume_id, section="summary", content="New Content")]
        emb_v2 = [self._sample_vector(2)]
        self.store.add_chunks(chunks_v2, emb_v2)

        res = self.store.search(resume_id, emb_v2[0], top_k=5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0][0].chunk_id, "v2_0")

        # Delete resume
        deleted = self.store.delete_by_resume_id(resume_id)
        self.assertTrue(deleted)
        self.assertFalse(self.store.is_indexed(resume_id))
        self.assertEqual(len(self.store.search(resume_id, emb_v2[0])), 0)

    def test_multi_user_isolation(self):
        """Resumes belonging to different users are strictly isolated in search and DB."""
        res_user_a = "resume_user_A"
        res_user_b = "resume_user_B"
        vec = self._sample_vector(123)

        chunks_a = [ResumeChunk(chunk_id="a_0", resume_id=res_user_a, section="skills", content="React JS")]
        chunks_b = [ResumeChunk(chunk_id="b_0", resume_id=res_user_b, section="skills", content="Angular JS")]

        self.store.add_chunks(chunks_a, [vec], user_id="user_A")
        self.store.add_chunks(chunks_b, [vec], user_id="user_B")

        # Querying within resume A never returns resume B chunks
        res_a = self.store.search(res_user_a, vec)
        self.assertEqual(len(res_a), 1)
        self.assertEqual(res_a[0][0].chunk_id, "a_0")

        res_b = self.store.search(res_user_b, vec)
        self.assertEqual(len(res_b), 1)
        self.assertEqual(res_b[0][0].chunk_id, "b_0")


if __name__ == "__main__":
    unittest.main()
