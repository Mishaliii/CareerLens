import os
import unittest
from unittest.mock import patch

import numpy as np

from backend.services.embedder import (
    BaseEmbedder,
    MockEmbedder,
    SentenceTransformerEmbedder,
    get_embedder,
)


class TestEmbedder(unittest.TestCase):
    """Unit tests for BaseEmbedder, MockEmbedder, and SentenceTransformerEmbedder."""

    def setUp(self):
        self.embedder = MockEmbedder(dimension=128)

    def test_mock_embedder_dimension_and_normalization(self):
        """Test MockEmbedder produces vectors with exact dimension and unit norm."""
        self.assertEqual(self.embedder.dimension, 128)
        vec = self.embedder.embed_text("FastAPI backend developer")
        self.assertEqual(len(vec), 128)
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)

    def test_mock_embedder_determinism(self):
        """Test identical text yields identical embeddings across calls."""
        vec1 = self.embedder.embed_text("Python engineer with PostgreSQL experience")
        vec2 = self.embedder.embed_text("Python engineer with PostgreSQL experience")
        self.assertEqual(vec1, vec2)

    def test_mock_embedder_batch_consistency(self):
        """Test embed_batch matches embed_text for individual items."""
        texts = ["Python", "FastAPI", "Docker"]
        batch_vecs = self.embedder.embed_batch(texts)
        self.assertEqual(len(batch_vecs), 3)
        for i, text in enumerate(texts):
            single_vec = self.embedder.embed_text(text)
            self.assertEqual(batch_vecs[i], single_vec)

    def test_mock_embedder_empty_text(self):
        """Test empty string produces zero vector without error."""
        vec = self.embedder.embed_text("")
        self.assertEqual(len(vec), 128)
        self.assertEqual(vec, [0.0] * 128)

    def test_semantic_overlap_similarity(self):
        """Test texts with common keywords have higher similarity than unrelated texts."""
        v_python1 = np.array(self.embedder.embed_text("Python software development"))
        v_python2 = np.array(self.embedder.embed_text("Backend Python development APIs"))
        v_unrelated = np.array(self.embedder.embed_text("Culinary chef French pastries"))

        sim_related = np.dot(v_python1, v_python2)
        sim_unrelated = np.dot(v_python1, v_unrelated)

        self.assertGreater(sim_related, sim_unrelated)

    def test_invalid_dimension_raises(self):
        """Test zero or negative dimension raises ValueError."""
        with self.assertRaises(ValueError):
            MockEmbedder(dimension=0)
        with self.assertRaises(ValueError):
            MockEmbedder(dimension=-10)

    def test_sentence_transformer_lazy_loading(self):
        """Test SentenceTransformerEmbedder does not instantiate model on __init__."""
        st_embedder = SentenceTransformerEmbedder(model_name="dummy-model")
        self.assertIsNone(st_embedder._model)

    def test_sentence_transformer_failure_raises_runtime_error(self):
        """Test that failed real model load raises RuntimeError without silent fallback."""
        st_embedder = SentenceTransformerEmbedder(model_name="nonexistent-model-xyz")
        from unittest.mock import MagicMock
        mock_st_mod = MagicMock()
        mock_st_mod.SentenceTransformer.side_effect = RuntimeError("Mock model load failure")
        with patch.dict("sys.modules", {"sentence_transformers": mock_st_mod}):
            with self.assertRaises(RuntimeError) as ctx:
                st_embedder.embed_text("test")
            self.assertIn("could not be loaded", str(ctx.exception))

    def test_get_embedder_factory(self):
        """Test factory returns MockEmbedder when configured."""
        mock_instance = get_embedder(use_mock=True)
        self.assertIsInstance(mock_instance, MockEmbedder)

        with patch.dict(os.environ, {"CAREERLENS_USE_MOCK_EMBEDDER": "1"}):
            env_mock = get_embedder()
            self.assertIsInstance(env_mock, MockEmbedder)


if __name__ == "__main__":
    unittest.main()
