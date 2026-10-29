import abc
import hashlib
import logging
import os
import re
import sys
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Environment variable to configure the embedding model name.
# Supported examples: 'sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-small-en-v1.5', 'BAAI/bge-m3'
EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL_NAME"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MOCK_DIMENSION = 384


class BaseEmbedder(abc.ABC):
    """Abstract interface defining the CareerLens text embedding service."""

    @abc.abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """Embed a single text string into a normalized dense vector."""
        pass

    @abc.abstractmethod
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Embed a batch of text strings into normalized dense vectors."""
        pass

    @property
    @abc.abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass


class MockEmbedder(BaseEmbedder):
    """
    Deterministic mock embedder for tests and local environments without neural model weights.

    Generates deterministic, L2-normalized dense vectors. Uses term-hash projection
    so texts sharing keywords exhibit higher cosine similarity than unrelated texts,
    enabling realistic retrieval testing without downloading gigabytes of weights.
    """

    def __init__(self, dimension: int = DEFAULT_MOCK_DIMENSION):
        if dimension <= 0:
            raise ValueError(f"Embedding dimension must be positive, got {dimension}")
        self._dimension = dimension
        logger.info("MockEmbedder initialized (dimension=%d)", dimension)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        """Generate a deterministic normalized vector for a single text."""
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate deterministic normalized vectors for a batch of texts."""
        vectors: List[List[float]] = []

        for text in texts:
            clean = text.strip().lower() if text else ""
            if not clean:
                # Zero vector for empty text
                vectors.append([0.0] * self._dimension)
                continue

            # Deterministic base vector seeded from sha256 of text
            base_seed = int(hashlib.sha256(clean.encode("utf-8")).hexdigest()[:8], 16)
            rng = np.random.RandomState(base_seed)
            vec = rng.randn(self._dimension) * 0.1

            # Keyword-based semantic projections for shared vocabulary overlap
            tokens = [w for w in re.split(r'\W+', clean) if len(w) > 2]
            for token in tokens:
                token_seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
                token_rng = np.random.RandomState(token_seed)
                vec += token_rng.randn(self._dimension) * 0.4

            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = np.zeros(self._dimension)

            vectors.append(vec.tolist())

        return vectors


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Local embedding implementation using SentenceTransformers.

    Loads model lazily on first embedding call to prevent import-time overhead.
    Does NOT download weights or fall back silently if model loading fails in production.
    """

    def __init__(self, model_name: Optional[str] = None):
        self.model_name = model_name or os.getenv(EMBEDDING_MODEL_ENV_VAR, DEFAULT_EMBEDDING_MODEL)
        self._model = None
        self._dimension: Optional[int] = None
        logger.info("SentenceTransformerEmbedder configured (model_name=%s, lazy_load=True)", self.model_name)

    def _get_model(self):
        """Lazy load the sentence transformer model on demand."""
        if self._model is None:
            logger.info("Loading SentenceTransformer model: %s", self.model_name)
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self.model_name)
                # Determine embedding dimension
                test_vec = self._model.encode("dim_probe", normalize_embeddings=True)
                self._dimension = len(test_vec)
                logger.info("SentenceTransformer model '%s' loaded successfully (dimension=%d)", self.model_name, self._dimension)
            except Exception as e:
                logger.error("Failed to load SentenceTransformer model '%s': %s", self.model_name, type(e).__name__)
                raise RuntimeError(
                    f"Embedding model '{self.model_name}' could not be loaded: {str(e)}. "
                    f"Ensure model weights are available or set CAREERLENS_USE_MOCK_EMBEDDER=1 for offline tests."
                ) from e
        return self._model

    @property
    def dimension(self) -> int:
        if self._dimension is None:
            self._get_model()
        return self._dimension or DEFAULT_MOCK_DIMENSION

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []

        model = self._get_model()
        try:
            embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return embeddings.tolist()
        except Exception as e:
            logger.error("Failed to generate embeddings with model '%s': %s", self.model_name, type(e).__name__)
            raise RuntimeError(f"Embedding generation failed: {str(e)}") from e


def get_embedder(
    use_mock: Optional[bool] = None,
    model_name: Optional[str] = None,
) -> BaseEmbedder:
    """
    Factory function returning the configured embedding service.

    If use_mock is True, or if CAREERLENS_USE_MOCK_EMBEDDER env var is set ('1', 'true', 'yes'),
    or if executing inside an automated test runner without explicit override,
    returns deterministic MockEmbedder. Otherwise, returns SentenceTransformerEmbedder.
    """
    if use_mock is None:
        env_val = os.getenv("CAREERLENS_USE_MOCK_EMBEDDER", "").strip().lower()
        if env_val:
            use_mock = env_val in ("1", "true", "yes")
        elif "unittest" in sys.modules and any("test" in str(arg).lower() for arg in sys.argv):
            use_mock = True

    if use_mock:
        return MockEmbedder()
    return SentenceTransformerEmbedder(model_name=model_name)
