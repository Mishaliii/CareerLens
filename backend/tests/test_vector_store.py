import unittest

from backend.models.schemas import ResumeChunk
from backend.services.embedder import MockEmbedder
from backend.services.retriever import ResumeRetriever
from backend.services.vector_store import InMemoryVectorStore


class TestVectorStoreAndRetriever(unittest.TestCase):
    """Unit tests for InMemoryVectorStore and ResumeRetriever."""

    def setUp(self):
        self.store = InMemoryVectorStore()
        self.embedder = MockEmbedder(dimension=64)
        self.retriever = ResumeRetriever(embedder=self.embedder, vector_store=self.store)

    def _create_sample_chunks(self, resume_id: str) -> list[ResumeChunk]:
        return [
            ResumeChunk(
                chunk_id=f"{resume_id}_exp_1",
                resume_id=resume_id,
                section="experience",
                item_title="Software Engineer @ Acme",
                content="Built high-throughput payment pipelines using FastAPI and Python.",
                page_number=1,
            ),
            ResumeChunk(
                chunk_id=f"{resume_id}_proj_1",
                resume_id=resume_id,
                section="projects",
                item_title="Project Cloud Monitor",
                content="Real-time distributed metrics monitoring dashboard using Docker and React.",
                page_number=2,
            ),
            ResumeChunk(
                chunk_id=f"{resume_id}_edu_1",
                resume_id=resume_id,
                section="education",
                item_title="B.Tech Computer Science",
                content="Graduated from Institute of Technology with GPA 8.5.",
                page_number=2,
            ),
        ]

    def test_add_and_search_scoped_to_resume(self):
        """Test search strictly isolates chunks by resume_id."""
        chunks_a = self._create_sample_chunks("resume_A")
        chunks_b = self._create_sample_chunks("resume_B")

        self.retriever.index_resume("resume_A", chunks_a)
        self.retriever.index_resume("resume_B", chunks_b)

        # Search resume_A
        results_a = self.retriever.retrieve("resume_A", "FastAPI Python payment", top_k=2)
        self.assertGreater(len(results_a), 0)
        for chunk, score in results_a:
            self.assertEqual(chunk.resume_id, "resume_A")
            self.assertNotEqual(chunk.resume_id, "resume_B")

        # Search resume_B
        results_b = self.retriever.retrieve("resume_B", "Docker React monitoring", top_k=2)
        self.assertGreater(len(results_b), 0)
        for chunk, score in results_b:
            self.assertEqual(chunk.resume_id, "resume_B")

    def test_idempotent_repeated_indexing(self):
        """Test repeated indexing of identical chunks does not duplicate them."""
        chunks = self._create_sample_chunks("resume_repeat")
        self.retriever.index_resume("resume_repeat", chunks)
        status1 = self.retriever.get_index_status("resume_repeat")
        self.assertEqual(status1["chunk_count"], 3)

        # Index again with identical chunks
        self.retriever.index_resume("resume_repeat", chunks)
        status2 = self.retriever.get_index_status("resume_repeat")
        self.assertEqual(status2["chunk_count"], 3)

    def test_delete_by_resume_id(self):
        """Test deleting resume purge chunks and vectors."""
        chunks = self._create_sample_chunks("resume_del")
        self.retriever.index_resume("resume_del", chunks)
        self.assertTrue(self.store.is_indexed("resume_del"))

        deleted = self.retriever.delete_resume_index("resume_del")
        self.assertTrue(deleted)
        self.assertFalse(self.store.is_indexed("resume_del"))

        # Subsequent retrieval returns empty
        results = self.retriever.retrieve("resume_del", "FastAPI")
        self.assertEqual(results, [])

    def test_vector_dimension_mismatch_raises(self):
        """Test query vector with wrong dimension raises ValueError."""
        chunks = self._create_sample_chunks("resume_dim")
        self.retriever.index_resume("resume_dim", chunks)

        wrong_dim_query = [0.1] * 128  # Store is 64 dims
        with self.assertRaises(ValueError) as ctx:
            self.store.search("resume_dim", wrong_dim_query)
        self.assertIn("dimension", str(ctx.exception).lower())

    def test_search_nonexistent_or_empty_resume(self):
        """Test searching non-indexed resume returns empty list without error."""
        results = self.retriever.retrieve("non_existent_resume", "query")
        self.assertEqual(results, [])

    def test_zero_norm_query_vector_handled_safely(self):
        """Test zero-norm query vector does not crash or divide by zero."""
        chunks = self._create_sample_chunks("resume_zero")
        self.retriever.index_resume("resume_zero", chunks)

        zero_vec = [0.0] * 64
        results = self.store.search("resume_zero", zero_vec)
        self.assertEqual(results, [])

    def test_get_status_metadata(self):
        """Test get_status reports correct sections and chunk counts."""
        chunks = self._create_sample_chunks("resume_status")
        self.retriever.index_resume("resume_status", chunks)

        status = self.retriever.get_index_status("resume_status")
        self.assertTrue(status["is_indexed"])
        self.assertEqual(status["chunk_count"], 3)
        self.assertIn("experience", status["sections_indexed"])
        self.assertIn("projects", status["sections_indexed"])
        self.assertIn("education", status["sections_indexed"])


if __name__ == "__main__":
    unittest.main()
