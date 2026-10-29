"""
Comprehensive unit and integration tests for FaissVectorStore.

Validates:
1. Persistence across process restarts.
2. Idempotent repeated indexing and re-indexing of modified resumes.
3. Deletion and cross-resume search isolation.
4. Auto-recovery from missing or corrupted index files using SQLite embeddings.
5. Incompatible embedding model and dimension detection.
6. Non-finite (NaN/Inf) vector rejection.
"""

import math
import os
import shutil
import tempfile
import unittest
import numpy as np
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker

from backend.db.database import Base
from backend.db.migrations import run_migrations
from backend.db.models import Resume, ResumeChunkRecord
from backend.models.schemas import ResumeChunk
from backend.services.vector_store import FaissVectorStore, IncompatibleEmbeddingError


class TestFaissVectorStore(unittest.TestCase):
    """Rigorous tests for FaissVectorStore persistence, isolation, and recovery."""

    def setUp(self):
        # 1. Create temporary directory for FAISS index files
        self.temp_index_dir = tempfile.mkdtemp(prefix="careerlens_faiss_")

        # 2. Create isolated temporary SQLite database
        self.temp_db_file = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.temp_db_path = self.temp_db_file.name
        self.temp_db_file.close()

        self.db_url = f"sqlite:///{self.temp_db_path}"
        self.engine = create_engine(
            self.db_url,
            connect_args={"check_same_thread": False},
        )

        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragmas(dbapi_connection, connection_record):
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys = ON;")
            cursor.execute("PRAGMA journal_mode = WAL;")
            cursor.execute("PRAGMA busy_timeout = 5000;")
            cursor.close()

        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
            expire_on_commit=False,
        )

        run_migrations(self.engine)

        self.dimension = 32
        self.model_name = "test-embedding-model-v1"
        self.store = FaissVectorStore(
            index_dir=self.temp_index_dir,
            dimension=self.dimension,
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

    def _generate_vector(self, seed: int, dim: int = 32) -> list[float]:
        rng = np.random.RandomState(seed)
        v = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(v)
        if norm > 1e-9:
            v /= norm
        return v.tolist()

    def _create_sample_chunks(self, resume_id: str, count: int = 3) -> tuple[list[ResumeChunk], list[list[float]]]:
        chunks = []
        embeddings = []
        sections = ["summary", "work_experience", "projects", "education", "skills"]
        for i in range(count):
            cid = f"{resume_id}_c_{i}"
            sec = sections[i % len(sections)]
            chunks.append(
                ResumeChunk(
                    chunk_id=cid,
                    resume_id=resume_id,
                    section=sec,
                    content=f"Content for chunk {i} of resume {resume_id} in section {sec}.",
                    item_title=f"Title {i}",
                    page_number=(i // 2) + 1,
                    metadata={"section": sec, "index": i},
                )
            )
            embeddings.append(self._generate_vector(seed=hash(cid) % (2**31), dim=self.dimension))
        return chunks, embeddings

    def test_persistence_across_restarts(self):
        """Chunks and FAISS vectors survive application restart (simulated store reconstruction)."""
        resume_id = "res_persist_001"
        chunks, embeddings = self._create_sample_chunks(resume_id, count=4)

        # 1. Index chunks in first store instance
        self.store.add_chunks(chunks, embeddings)
        self.assertTrue(self.store.is_indexed(resume_id))

        # Verify index file exists on disk
        idx_file = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")
        self.assertTrue(os.path.exists(idx_file))

        # 2. Simulate complete restart: instantiate a new store with empty in-memory caches
        new_store = FaissVectorStore(
            index_dir=self.temp_index_dir,
            dimension=self.dimension,
            expected_model_name=self.model_name,
            session_factory=self.SessionLocal,
        )

        # 3. Retrieve chunks using new store
        query_vec = embeddings[0]
        results = new_store.search(resume_id, query_vec, top_k=2)

        self.assertEqual(len(results), 2)
        top_chunk, score = results[0]
        self.assertEqual(top_chunk.chunk_id, chunks[0].chunk_id)
        self.assertEqual(top_chunk.section, chunks[0].section)
        self.assertEqual(top_chunk.page_number, chunks[0].page_number)
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_idempotent_repeated_indexing(self):
        """Indexing the identical resume twice does not duplicate records in SQLite or FAISS."""
        resume_id = "res_idempotent"
        chunks, embeddings = self._create_sample_chunks(resume_id, count=3)

        self.store.add_chunks(chunks, embeddings)
        status1 = self.store.get_status(resume_id)
        self.assertEqual(status1["chunk_count"], 3)

        # Re-index same chunks
        self.store.add_chunks(chunks, embeddings)
        status2 = self.store.get_status(resume_id)
        self.assertEqual(status2["chunk_count"], 3)

        # Verify SQLite row count
        db = self.SessionLocal()
        count = db.query(ResumeChunkRecord).filter_by(resume_id=resume_id).count()
        db.close()
        self.assertEqual(count, 3)

    def test_reindexing_modified_resume_purges_stale_vectors(self):
        """Re-indexing a modified resume with fewer chunks purges old chunks and vectors."""
        resume_id = "res_modified"
        chunks_v1, emb_v1 = self._create_sample_chunks(resume_id, count=5)
        self.store.add_chunks(chunks_v1, emb_v1)

        # Re-index with only 2 chunks
        chunks_v2 = [
            ResumeChunk(
                chunk_id=f"{resume_id}_v2_0",
                resume_id=resume_id,
                section="summary",
                content="Updated single summary",
            ),
            ResumeChunk(
                chunk_id=f"{resume_id}_v2_1",
                resume_id=resume_id,
                section="skills",
                content="Updated skills list",
            ),
        ]
        emb_v2 = [self._generate_vector(111, self.dimension), self._generate_vector(222, self.dimension)]
        self.store.add_chunks(chunks_v2, emb_v2)

        status = self.store.get_status(resume_id)
        self.assertEqual(status["chunk_count"], 2)

        # Search should only return v2 chunks
        results = self.store.search(resume_id, emb_v2[0], top_k=5)
        self.assertEqual(len(results), 2)
        returned_ids = [c.chunk_id for c, _ in results]
        self.assertIn(f"{resume_id}_v2_0", returned_ids)
        self.assertNotIn(f"{resume_id}_c_0", returned_ids)

    def test_deletion_and_isolation(self):
        """Deleting one resume removes its vectors while other resumes remain searchable."""
        res_a = "resume_A"
        res_b = "resume_B"
        chunks_a, emb_a = self._create_sample_chunks(res_a, count=3)
        chunks_b, emb_b = self._create_sample_chunks(res_b, count=3)

        self.store.add_chunks(chunks_a, emb_a)
        self.store.add_chunks(chunks_b, emb_b)

        # Delete resume A
        existed = self.store.delete_by_resume_id(res_a)
        self.assertTrue(existed)

        # Resume A is no longer searchable
        self.assertFalse(self.store.is_indexed(res_a))
        self.assertEqual(len(self.store.search(res_a, emb_a[0])), 0)

        # Resume B remains fully searchable
        self.assertTrue(self.store.is_indexed(res_b))
        results_b = self.store.search(res_b, emb_b[0], top_k=2)
        self.assertGreater(len(results_b), 0)
        self.assertEqual(results_b[0][0].resume_id, res_b)

    def test_cross_resume_isolation_guarantee(self):
        """Queries for resume A never return chunks belonging to resume B, even with identical vectors."""
        res_a = "resume_iso_A"
        res_b = "resume_iso_B"
        identical_vector = self._generate_vector(42, self.dimension)

        chunk_a = [ResumeChunk(chunk_id="a_1", resume_id=res_a, section="summary", content="Resume A Content")]
        chunk_b = [ResumeChunk(chunk_id="b_1", resume_id=res_b, section="summary", content="Resume B Content")]

        self.store.add_chunks(chunk_a, [identical_vector])
        self.store.add_chunks(chunk_b, [identical_vector])

        # Search within A with identical query vector
        results_a = self.store.search(res_a, identical_vector, top_k=5)
        self.assertEqual(len(results_a), 1)
        self.assertEqual(results_a[0][0].resume_id, res_a)
        self.assertEqual(results_a[0][0].chunk_id, "a_1")

        # Search within B with identical query vector
        results_b = self.store.search(res_b, identical_vector, top_k=5)
        self.assertEqual(len(results_b), 1)
        self.assertEqual(results_b[0][0].resume_id, res_b)
        self.assertEqual(results_b[0][0].chunk_id, "b_1")

    def test_auto_recovery_from_missing_index_file(self):
        """If FAISS index file is missing on disk, search auto-rebuilds index from SQLite."""
        resume_id = "res_recovery_missing"
        chunks, embeddings = self._create_sample_chunks(resume_id, count=3)
        self.store.add_chunks(chunks, embeddings)

        idx_file = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")
        self.assertTrue(os.path.exists(idx_file))

        # Clear in-memory cache and delete disk index file
        self.store._indexes.clear()
        self.store._chunks_cache.clear()
        os.remove(idx_file)
        self.assertFalse(os.path.exists(idx_file))

        # Search triggers automatic index reconstruction from SQLite embeddings
        results = self.store.search(resume_id, embeddings[0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].chunk_id, chunks[0].chunk_id)

        # Index file is recreated on disk
        self.assertTrue(os.path.exists(idx_file))

    def test_auto_recovery_from_corrupted_index_file(self):
        """If FAISS index file is corrupted on disk, search recovers gracefully from SQLite."""
        resume_id = "res_recovery_corrupt"
        chunks, embeddings = self._create_sample_chunks(resume_id, count=3)
        self.store.add_chunks(chunks, embeddings)

        idx_file = os.path.join(self.temp_index_dir, f"{resume_id}.faiss")

        # Overwrite file with garbage bytes
        self.store._indexes.clear()
        self.store._chunks_cache.clear()
        with open(idx_file, "wb") as f:
            f.write(b"CORRUPTED_FAISS_HEADER_GARBAGE_BYTES_123456789")

        # Search detects corruption and auto-rebuilds from SQLite
        results = self.store.search(resume_id, embeddings[0], top_k=2)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0][0].chunk_id, chunks[0].chunk_id)

    def test_reject_incompatible_embedding_model(self):
        """Searching a resume indexed with a different model raises IncompatibleEmbeddingError."""
        resume_id = "res_model_mismatch"
        chunks, embeddings = self._create_sample_chunks(resume_id, count=2)
        self.store.add_chunks(chunks, embeddings)

        # Instantiate store configured with a different model name
        different_store = FaissVectorStore(
            index_dir=self.temp_index_dir,
            dimension=self.dimension,
            expected_model_name="sentence-transformers/bge-m3",
            session_factory=self.SessionLocal,
        )

        with self.assertRaises(IncompatibleEmbeddingError):
            different_store.search(resume_id, embeddings[0])

    def test_reject_dimension_mismatch(self):
        """Vectors with incorrect dimensions fail validation with ValueError."""
        resume_id = "res_dim_err"
        chunks = [ResumeChunk(chunk_id="c1", resume_id=resume_id, section="summary", content="text")]
        wrong_dim_vec = [[1.0, 2.0, 3.0]]  # 3 dimensions instead of 32

        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks, wrong_dim_vec)

    def test_reject_non_finite_vectors(self):
        """Vectors containing NaN or Inf fail validation."""
        resume_id = "res_nan_err"
        chunks = [ResumeChunk(chunk_id="c1", resume_id=resume_id, section="summary", content="text")]
        nan_vec = [[float("nan")] * self.dimension]
        inf_vec = [[float("inf")] * self.dimension]

        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks, nan_vec)

        with self.assertRaises(ValueError):
            self.store.add_chunks(chunks, inf_vec)


if __name__ == "__main__":
    unittest.main()
