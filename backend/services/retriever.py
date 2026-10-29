import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.models.schemas import ResumeChunk
from backend.services.embedder import BaseEmbedder, get_embedder
from backend.services.vector_store import VectorStore, get_vector_store

logger = logging.getLogger(__name__)


class ResumeRetriever:
    """
    Retrieval coordinator providing semantic search scoped to individual resumes.
    """

    def __init__(
        self,
        embedder: Optional[BaseEmbedder] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.embedder = embedder or get_embedder()
        self.vector_store = vector_store or get_vector_store()
        logger.info("ResumeRetriever initialized")

    def index_resume(
        self,
        resume_id: str,
        chunks: List[ResumeChunk],
        purge_existing: bool = True,
        user_id: Optional[str] = None,
    ) -> int:
        """
        Embed and index resume chunks.

        Args:
            resume_id: Deterministic document identifier.
            chunks: List of ResumeChunk instances.
            purge_existing: Whether to clear prior index entries for this resume_id before indexing.
            user_id: Optional owner user ID.

        Returns:
            Number of successfully indexed chunks.
        """
        if not resume_id or not chunks:
            logger.warning("index_resume called with empty resume_id or no chunks")
            return 0

        logger.info("Indexing %d chunks for resume_id=%s", len(chunks), resume_id)
        texts = [c.content for c in chunks]
        embeddings = self.embedder.embed_batch(texts)
        if purge_existing:
            self.vector_store.delete_by_resume_id(resume_id)
        self.vector_store.add_chunks(chunks, embeddings, user_id=user_id)
        return len(chunks)

    def retrieve(
        self,
        resume_id: str,
        query: str,
        top_k: int = 4,
    ) -> List[Tuple[ResumeChunk, float]]:
        """
        Retrieve top_k most relevant chunks for a query from a specific resume.

        Args:
            resume_id: Identifier of the resume to search within.
            query: User's question or search query.
            top_k: Maximum number of chunks to return.

        Returns:
            List of (ResumeChunk, cosine_similarity_score) sorted descending.
        """
        if not resume_id or not query or not query.strip():
            logger.warning("retrieve called with invalid resume_id or empty query")
            return []

        if not self.vector_store.is_indexed(resume_id):
            logger.info("Resume %s is not indexed; returning empty retrieval results", resume_id)
            return []

        clean_query = query.strip()
        query_embedding = self.embedder.embed_text(clean_query)
        results = self.vector_store.search(
            resume_id=resume_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        logger.info("Retrieved %d candidate chunks for resume_id=%s", len(results), resume_id)
        return results

    def get_index_status(self, resume_id: str) -> Dict[str, Any]:
        """Check if resume is indexed and return chunk metadata."""
        return self.vector_store.get_status(resume_id)

    def delete_resume_index(self, resume_id: str) -> bool:
        """Remove resume from vector store."""
        return self.vector_store.delete_by_resume_id(resume_id)


_GLOBAL_RETRIEVER: Optional[ResumeRetriever] = None


def get_retriever(
    embedder: Optional[BaseEmbedder] = None,
    vector_store: Optional[VectorStore] = None,
) -> ResumeRetriever:
    """Dependency provider returning singleton retriever instance."""
    global _GLOBAL_RETRIEVER
    if _GLOBAL_RETRIEVER is None:
        _GLOBAL_RETRIEVER = ResumeRetriever(
            embedder=embedder or get_embedder(),
            vector_store=vector_store or get_vector_store(),
        )
    return _GLOBAL_RETRIEVER


def reset_retriever() -> None:
    """Utility function to reset global retriever for tests."""
    global _GLOBAL_RETRIEVER
    _GLOBAL_RETRIEVER = None
