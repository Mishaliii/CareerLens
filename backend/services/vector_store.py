"""
Vector store abstraction, in-memory implementation, and persistent FAISS vector store.

Entities:
- VectorStore: Abstract interface for vector similarity search and index management.
- InMemoryVectorStore: Ephemeral in-memory vector store for isolated unit tests.
- FaissVectorStore: Persistent vector store backed by SQLite (source of truth) and disk-persisted FAISS.
- IncompatibleEmbeddingError: Raised when querying an index built with an incompatible embedding model.
"""

import abc
import logging
import math
import os
import threading
from typing import Any, Dict, List, Optional, Tuple
import uuid

import faiss
import numpy as np

from backend.db.database import Base, SessionLocal, engine
from backend.db.models import Resume, ResumeChunkRecord
from backend.models.schemas import ResumeChunk

logger = logging.getLogger(__name__)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_INDEX_DIR = os.path.join(PROJECT_ROOT, "data", "vector_indexes")
VECTOR_INDEX_DIR_ENV = "VECTOR_INDEX_DIR"
EMBEDDING_MODEL_ENV_VAR = "EMBEDDING_MODEL_NAME"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class IncompatibleEmbeddingError(ValueError):
    """Raised when querying an index with an incompatible embedding model or dimension."""
    pass


class VectorStore(abc.ABC):
    """
    Abstract interface for document and chunk vector storage.
    Supports in-memory test implementation and persistent FAISS/pgvector backends.
    """

    @abc.abstractmethod
    def add_chunks(
        self,
        chunks: List[ResumeChunk],
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
    ) -> None:
        """Add or update chunks and their corresponding embedding vectors."""
        pass

    @abc.abstractmethod
    def search(
        self,
        resume_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Tuple[ResumeChunk, float]]:
        """Search for top_k relevant chunks strictly within the specified resume_id."""
        pass

    @abc.abstractmethod
    def delete_by_resume_id(self, resume_id: str) -> bool:
        """Delete all chunks and vectors associated with a resume_id."""
        pass

    @abc.abstractmethod
    def is_indexed(self, resume_id: str) -> bool:
        """Check if a resume has indexed chunks."""
        pass

    @abc.abstractmethod
    def get_status(self, resume_id: str) -> Dict[str, Any]:
        """Return index status, chunk count, and indexed sections."""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clear all in-memory or persisted indexes."""
        pass


class InMemoryVectorStore(VectorStore):
    """
    In-memory vector store implementation with document-level isolation and idempotency.

    Suitable for isolated fast unit tests without disk or database access.
    """

    def __init__(self):
        self._chunks: Dict[str, Dict[str, ResumeChunk]] = {}
        self._vectors: Dict[str, Dict[str, np.ndarray]] = {}
        logger.info("InMemoryVectorStore initialized (ephemeral)")

    def add_chunks(
        self,
        chunks: List[ResumeChunk],
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
    ) -> None:
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        resume_id = chunks[0].resume_id
        if resume_id not in self._chunks:
            self._chunks[resume_id] = {}
            self._vectors[resume_id] = {}

        for chunk, emb in zip(chunks, embeddings):
            if chunk.resume_id != resume_id:
                raise ValueError("All chunks in a batch must belong to the same resume_id")

            vec = np.array(emb, dtype=np.float32)
            self._chunks[resume_id][chunk.chunk_id] = chunk
            self._vectors[resume_id][chunk.chunk_id] = vec

        logger.info(
            "Indexed %d chunks for resume_id=%s (total chunks for resume: %d)",
            len(chunks),
            resume_id,
            len(self._chunks[resume_id]),
        )

    def search(
        self,
        resume_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Tuple[ResumeChunk, float]]:
        if not resume_id or resume_id not in self._chunks:
            return []

        resume_chunks = self._chunks[resume_id]
        resume_vectors = self._vectors[resume_id]

        if not resume_chunks or not resume_vectors:
            return []

        q_vec = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_vec))
        if q_norm < 1e-9:
            logger.warning("Zero-norm query vector supplied to vector search")
            return []

        chunk_ids = list(resume_vectors.keys())
        matrix = np.stack([resume_vectors[cid] for cid in chunk_ids])

        if matrix.shape[1] != len(q_vec):
            raise ValueError(
                f"Query vector dimension ({len(q_vec)}) does not match stored vector dimension ({matrix.shape[1]})"
            )

        doc_norms = np.linalg.norm(matrix, axis=1)
        valid_mask = doc_norms > 1e-9
        similarities = np.zeros(len(chunk_ids), dtype=np.float32)
        if np.any(valid_mask):
            dots = np.dot(matrix[valid_mask], q_vec)
            similarities[valid_mask] = dots / (doc_norms[valid_mask] * q_norm)

        similarities = np.clip(similarities, -1.0, 1.0)
        ranked_indices = np.argsort(-similarities)
        k = min(top_k, len(chunk_ids))

        results: List[Tuple[ResumeChunk, float]] = []
        for i in range(k):
            idx = ranked_indices[i]
            cid = chunk_ids[idx]
            chunk = resume_chunks[cid]
            score = float(similarities[idx])
            results.append((chunk, score))

        return results

    def delete_by_resume_id(self, resume_id: str) -> bool:
        existed = resume_id in self._chunks
        if existed:
            del self._chunks[resume_id]
            del self._vectors[resume_id]
            logger.info("Deleted index for resume_id=%s", resume_id)
        return existed

    def is_indexed(self, resume_id: str) -> bool:
        return resume_id in self._chunks and len(self._chunks[resume_id]) > 0

    def get_status(self, resume_id: str) -> Dict[str, Any]:
        if not self.is_indexed(resume_id):
            return {
                "resume_id": resume_id,
                "is_indexed": False,
                "chunk_count": 0,
                "sections_indexed": [],
            }

        chunks = self._chunks[resume_id].values()
        sections = sorted(list(set(c.section for c in chunks)))
        return {
            "resume_id": resume_id,
            "is_indexed": True,
            "chunk_count": len(chunks),
            "sections_indexed": sections,
        }

    def clear(self) -> None:
        self._chunks.clear()
        self._vectors.clear()
        logger.info("InMemoryVectorStore cleared")


class FaissVectorStore(VectorStore):
    """
    Persistent vector store using FAISS with SQLite as the canonical source of truth.

    Guarantees:
    - Persistence across restarts: FAISS indexes are saved to disk with atomic file replacement.
    - Consistency: Chunks and embedding blobs are persisted in SQLite; missing/corrupted FAISS
      indexes are automatically rebuilt from SQLite.
    - Resume isolation: Each resume possesses a dedicated IndexFlatIP, preventing cross-resume leakage.
    - Idempotency: Re-indexing updates SQLite and rebuilds the resume's FAISS index cleanly.
    - Concurrency safety: Uses thread re-entrant locks around index mutations and disk writes.
    """

    def __init__(
        self,
        index_dir: Optional[str] = None,
        dimension: Optional[int] = None,
        expected_model_name: Optional[str] = None,
        session_factory=None,
    ):
        self.index_dir = index_dir or os.getenv(VECTOR_INDEX_DIR_ENV, DEFAULT_INDEX_DIR)
        try:
            os.makedirs(self.index_dir, exist_ok=True)
        except OSError as e:
            logger.warning("Could not create vector index directory %s: %s. Operating in-memory.", self.index_dir, e)

        self.dimension = dimension
        self.expected_model_name = expected_model_name or os.getenv(
            EMBEDDING_MODEL_ENV_VAR, DEFAULT_EMBEDDING_MODEL
        )
        self.session_factory = session_factory or SessionLocal
        self._lock = threading.RLock()

        # In-memory caches for fast query resolution:
        # Maps resume_id -> faiss.IndexFlatIP
        self._indexes: Dict[str, faiss.IndexFlatIP] = {}
        # Maps resume_id -> List[ResumeChunk] (ordered by chunk_index)
        self._chunks_cache: Dict[str, List[ResumeChunk]] = {}

        # Ensure database tables exist for SQLite persistence
        try:
            bind = getattr(self.session_factory, "kw", {}).get("bind") or engine
            Base.metadata.create_all(bind=bind)
        except Exception as e:
            logger.warning("Could not auto-create tables in FaissVectorStore: %s", e)

        logger.info(
            "FaissVectorStore initialized (index_dir=%s, default_dimension=%s, model=%s)",
            self.index_dir,
            self.dimension,
            self.expected_model_name,
        )

    def _get_resume_index_path(self, resume_id: str) -> str:
        return os.path.join(self.index_dir, f"{resume_id}.faiss")

    def _load_chunks_from_db(self, resume_id: str) -> List[ResumeChunk]:
        """Fetch all chunks for a resume from SQLite ordered by chunk_index."""
        with self.session_factory() as session:
            rows = (
                session.query(ResumeChunkRecord)
                .filter(ResumeChunkRecord.resume_id == resume_id)
                .order_by(ResumeChunkRecord.chunk_index)
                .all()
            )
            chunks: List[ResumeChunk] = []
            for r in rows:
                chunks.append(
                    ResumeChunk(
                        chunk_id=r.id,
                        resume_id=r.resume_id,
                        section=r.section,
                        content=r.content,
                        item_title=r.item_title,
                        page_number=r.page_number,
                        metadata=r.metadata_json or {},
                    )
                )
            return chunks

    def _load_or_rebuild_index(self, resume_id: str) -> Optional[faiss.IndexFlatIP]:
        """
        Load index from memory, disk, or rebuild from SQLite.

        Thread-safe under self._lock.
        """
        with self._lock:
            # 1. In-memory cache hit
            if resume_id in self._indexes:
                return self._indexes[resume_id]

            idx_path = self._get_resume_index_path(resume_id)

            # 2. Disk hit
            if os.path.exists(idx_path):
                try:
                    index = faiss.read_index(idx_path)
                    chunks = self._load_chunks_from_db(resume_id)

                    expected_dim = self.dimension
                    if expected_dim is None:
                        with self.session_factory() as session:
                            r = session.query(Resume).filter(Resume.id == resume_id).first()
                            if r and r.embedding_dimension:
                                expected_dim = r.embedding_dimension

                    dim_ok = (expected_dim is None) or (index.d == expected_dim)
                    count_ok = chunks and index.ntotal == len(chunks)

                    if count_ok and dim_ok:
                        self._indexes[resume_id] = index
                        self._chunks_cache[resume_id] = chunks
                        return index
                    else:
                        logger.warning(
                            "Disk index mismatch for resume %s (index ntotal=%d, db chunks=%d, index.d=%d, expected_dim=%s). Rebuilding from DB...",
                            resume_id,
                            index.ntotal,
                            len(chunks),
                            index.d,
                            expected_dim,
                        )
                except Exception as e:
                    logger.warning("Failed to read FAISS index from disk for %s: %s. Rebuilding from DB...", resume_id, e)

            # 3. Rebuild from SQLite
            success = self.rebuild_resume_index(resume_id)
            if success:
                return self._indexes.get(resume_id)

            return None

    def add_chunks(
        self,
        chunks: List[ResumeChunk],
        embeddings: List[List[float]],
        user_id: Optional[str] = None,
    ) -> None:
        """
        Persist chunks and embeddings to SQLite, build FAISS index, and atomically save to disk.
        """
        if not chunks:
            return

        if len(chunks) != len(embeddings):
            raise ValueError(
                f"Chunks count ({len(chunks)}) must match embeddings count ({len(embeddings)})"
            )

        resume_id = chunks[0].resume_id
        dim = len(embeddings[0])

        if dim <= 0:
            raise ValueError("Embedding dimension must be positive")

        if self.dimension is not None and dim != self.dimension:
            raise ValueError(
                f"Embedding dimension mismatch: store configured for {self.dimension}, got {dim}"
            )

        # Validate non-empty, uniform dimension, and finite floats
        for i, emb in enumerate(embeddings):
            if chunks[i].resume_id != resume_id:
                raise ValueError("All chunks in a batch must belong to the same resume_id")
            if len(emb) != dim:
                raise ValueError(f"Inconsistent vector dimension at index {i}: expected {dim}, got {len(emb)}")
            for val in emb:
                if math.isnan(val) or math.isinf(val):
                    raise ValueError(f"Embedding vector at index {i} contains non-finite values (NaN or Inf)")

        with self._lock:
            # 1. Persist to SQLite (Canonical Source of Truth)
            with self.session_factory() as session:
                resume = session.query(Resume).filter(Resume.id == resume_id).first()
                if not resume:
                    resume = Resume(
                        id=resume_id,
                        user_id=user_id,
                        filename=f"resume_{resume_id}.pdf",
                        processing_status="completed",
                        indexing_status="indexing",
                        embedding_model=self.expected_model_name,
                        embedding_dimension=dim,
                        chunk_count=len(chunks),
                    )
                    session.add(resume)
                    session.flush()
                else:
                    if user_id and not resume.user_id:
                        resume.user_id = user_id
                    resume.indexing_status = "indexing"
                    resume.embedding_model = self.expected_model_name
                    resume.embedding_dimension = dim
                    resume.chunk_count = len(chunks)

                # Delete existing chunks for this resume to guarantee idempotency
                session.query(ResumeChunkRecord).filter(ResumeChunkRecord.resume_id == resume_id).delete()

                # Insert new chunk records with serialized embedding blobs
                for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
                    rec = ResumeChunkRecord(
                        id=chunk.chunk_id,
                        resume_id=resume_id,
                        user_id=resume.user_id,
                        chunk_index=idx,
                        section=chunk.section,
                        item_title=chunk.item_title,
                        page_number=chunk.page_number,
                        content=chunk.content,
                        metadata_json=chunk.metadata or {},
                        vector_index_id=idx,
                    )
                    rec.set_embedding(emb, expected_dimension=dim)
                    session.add(rec)

                resume.indexing_status = "indexed"
                session.commit()

            # 2. Build FAISS Index (IndexFlatIP with normalized vectors for cosine similarity)
            arr = np.array(embeddings, dtype=np.float32)
            faiss.normalize_L2(arr)

            index = faiss.IndexFlatIP(dim)
            index.add(arr)

            # 3. Atomic Disk Write using a temporary file and atomic replace
            idx_path = self._get_resume_index_path(resume_id)
            tmp_path = f"{idx_path}.tmp.{uuid.uuid4().hex[:8]}"
            try:
                os.makedirs(os.path.dirname(idx_path), exist_ok=True)
                faiss.write_index(index, tmp_path)
                os.replace(tmp_path, idx_path)
            except Exception as e:
                logger.error("Failed to write FAISS index to disk for %s: %s", resume_id, e)
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                try:
                    with self.session_factory() as session:
                        res = session.query(Resume).filter(Resume.id == resume_id).first()
                        if res:
                            res.indexing_status = "failed"
                            session.commit()
                except Exception as db_err:
                    logger.warning("Could not set indexing_status='failed' for %s: %s", resume_id, db_err)
                raise

            # 4. Update in-memory caches
            self.dimension = dim
            self._indexes[resume_id] = index
            self._chunks_cache[resume_id] = list(chunks)

            logger.info("Indexed and persisted %d chunks for resume_id=%s in FAISS", len(chunks), resume_id)

    def search(
        self,
        resume_id: str,
        query_embedding: List[float],
        top_k: int = 4,
    ) -> List[Tuple[ResumeChunk, float]]:
        """
        Search for top_k relevant chunks strictly within the specified resume_id.
        """
        if not resume_id or not query_embedding:
            return []

        # Validate query vector
        q_arr = np.array(query_embedding, dtype=np.float32)
        q_norm = float(np.linalg.norm(q_arr))
        if q_norm < 1e-9:
            logger.warning("Zero-norm query vector supplied to vector search")
            return []

        # Check embedding compatibility against SQLite metadata
        with self.session_factory() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            if resume and resume.embedding_model:
                if (
                    self.expected_model_name
                    and resume.embedding_model != self.expected_model_name
                ):
                    raise IncompatibleEmbeddingError(
                        f"Resume {resume_id} was indexed with model '{resume.embedding_model}', "
                        f"which is incompatible with active model '{self.expected_model_name}'"
                    )

        index = self._load_or_rebuild_index(resume_id)
        if index is None or index.ntotal == 0:
            return []

        if len(query_embedding) != index.d:
            raise ValueError(
                f"Query vector dimension ({len(query_embedding)}) does not match index dimension ({index.d})"
            )

        chunks = self._chunks_cache.get(resume_id)
        if not chunks:
            chunks = self._load_chunks_from_db(resume_id)
            self._chunks_cache[resume_id] = chunks

        if not chunks:
            return []

        # L2-normalize query vector for inner product cosine search
        q_matrix = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(q_matrix)

        k = min(top_k, index.ntotal)
        distances, indices = index.search(q_matrix, k)

        results: List[Tuple[ResumeChunk, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(chunks):
                results.append((chunks[idx], float(dist)))

        return results

    def delete_by_resume_id(self, resume_id: str) -> bool:
        """
        Delete a resume's chunks from SQLite, evict from memory cache, and remove disk index.
        """
        with self._lock:
            idx_path = self._get_resume_index_path(resume_id)
            had_index = os.path.exists(idx_path) or resume_id in self._indexes

            # 1. Remove from memory
            self._indexes.pop(resume_id, None)
            self._chunks_cache.pop(resume_id, None)

            # 2. Remove from disk
            if os.path.exists(idx_path):
                try:
                    os.remove(idx_path)
                except OSError as e:
                    logger.warning("Failed to remove FAISS index file %s: %s", idx_path, e)

            # 3. Remove chunks from SQLite
            had_db_records = False
            with self.session_factory() as session:
                del_count = (
                    session.query(ResumeChunkRecord)
                    .filter(ResumeChunkRecord.resume_id == resume_id)
                    .delete()
                )
                resume = session.query(Resume).filter(Resume.id == resume_id).first()
                if resume:
                    resume.indexing_status = "unindexed"
                    resume.chunk_count = 0
                session.commit()
                had_db_records = del_count > 0 or resume is not None

            logger.info("Deleted vector index for resume_id=%s", resume_id)
            return had_index or had_db_records

    def is_indexed(self, resume_id: str) -> bool:
        """Check if resume is indexed in memory, on disk, or in SQLite."""
        if resume_id in self._indexes and self._indexes[resume_id].ntotal > 0:
            return True
        if os.path.exists(self._get_resume_index_path(resume_id)):
            return True
        with self.session_factory() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            return resume is not None and resume.indexing_status == "indexed" and resume.chunk_count > 0

    def get_status(self, resume_id: str) -> Dict[str, Any]:
        """Return index status, chunk count, sections, and embedding metadata."""
        if not self.is_indexed(resume_id):
            return {
                "resume_id": resume_id,
                "is_indexed": False,
                "chunk_count": 0,
                "sections_indexed": [],
            }

        with self.session_factory() as session:
            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            chunk_records = (
                session.query(ResumeChunkRecord.section)
                .filter(ResumeChunkRecord.resume_id == resume_id)
                .all()
            )
            sections = sorted(list(set(r[0] for r in chunk_records)))
            count = len(chunk_records)

            return {
                "resume_id": resume_id,
                "is_indexed": True,
                "chunk_count": count,
                "sections_indexed": sections,
                "embedding_model": resume.embedding_model if resume else self.expected_model_name,
                "dimension": resume.embedding_dimension if resume else self.dimension,
            }

    def clear(self) -> None:
        """Clear all in-memory caches and delete all persisted FAISS index files on disk."""
        with self._lock:
            self._indexes.clear()
            self._chunks_cache.clear()

            if os.path.exists(self.index_dir):
                for filename in os.listdir(self.index_dir):
                    if filename.endswith(".faiss") or ".tmp" in filename:
                        filepath = os.path.join(self.index_dir, filename)
                        try:
                            os.remove(filepath)
                        except OSError as e:
                            logger.warning("Could not delete index file %s: %s", filepath, e)

            logger.info("FaissVectorStore cleared")

    def rebuild_resume_index(self, resume_id: str) -> bool:
        """
        Reconstruct a FAISS index directly from stored SQLite embedding blobs.
        """
        with self._lock:
            with self.session_factory() as session:
                records = (
                    session.query(ResumeChunkRecord)
                    .filter(ResumeChunkRecord.resume_id == resume_id)
                    .order_by(ResumeChunkRecord.chunk_index)
                    .all()
                )

                if not records or not records[0].embedding_blob:
                    logger.info("Cannot rebuild index for %s: no chunks or embedding blobs in DB", resume_id)
                    return False

                vectors: List[List[float]] = []
                chunks: List[ResumeChunk] = []

                for r in records:
                    emb = r.get_embedding()
                    if emb is None:
                        logger.warning("Missing embedding blob for chunk %s; cannot rebuild index", r.id)
                        return False
                    vectors.append(emb)
                    chunks.append(
                        ResumeChunk(
                            chunk_id=r.id,
                            resume_id=r.resume_id,
                            section=r.section,
                            content=r.content,
                            item_title=r.item_title,
                            page_number=r.page_number,
                            metadata=r.metadata_json or {},
                        )
                    )

                dim = len(vectors[0])
                arr = np.array(vectors, dtype=np.float32)
                faiss.normalize_L2(arr)

                index = faiss.IndexFlatIP(dim)
                index.add(arr)

                # 1. Update in-memory caches
                self._indexes[resume_id] = index
                self._chunks_cache[resume_id] = chunks

                # 2. Update database resume status
                resume = session.query(Resume).filter(Resume.id == resume_id).first()
                if resume:
                    resume.indexing_status = "indexed"
                    resume.chunk_count = len(chunks)
                    resume.embedding_dimension = dim
                session.commit()

                # 3. Attempt atomic disk cache write (gracefully tolerates read-only or ephemeral filesystems)
                idx_path = self._get_resume_index_path(resume_id)
                tmp_path = f"{idx_path}.tmp.{uuid.uuid4().hex[:8]}"
                try:
                    os.makedirs(os.path.dirname(idx_path), exist_ok=True)
                    faiss.write_index(index, tmp_path)
                    os.replace(tmp_path, idx_path)
                except Exception as e:
                    logger.warning("Could not cache rebuilt FAISS index to disk for %s: %s. Operating with in-memory index.", resume_id, e)
                    if os.path.exists(tmp_path):
                        try:
                            os.remove(tmp_path)
                        except OSError:
                            pass

                logger.info("Successfully rebuilt FAISS index from DB records for resume_id=%s (%d chunks)", resume_id, len(chunks))
                return True

    def rebuild_all_from_db(self) -> int:
        """Rebuild FAISS indexes for all indexed resumes in SQLite."""
        with self.session_factory() as session:
            resume_ids = [
                r[0]
                for r in session.query(Resume.id)
                .filter(Resume.indexing_status == "indexed")
                .all()
            ]
        rebuilt = 0
        for rid in resume_ids:
            if self.rebuild_resume_index(rid):
                rebuilt += 1
        return rebuilt


# Global singleton vector store instance
_GLOBAL_VECTOR_STORE: Optional[VectorStore] = None


def get_vector_store(
    backend: Optional[str] = None,
    index_dir: Optional[str] = None,
    dimension: Optional[int] = None,
) -> VectorStore:
    """
    Dependency provider returning the configured vector store.

    Defaults to persistent FaissVectorStore.
    If CAREERLENS_VECTOR_STORE is 'in_memory' or explicitly requested,
    returns InMemoryVectorStore.
    """
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is None:
        target = backend or os.getenv("CAREERLENS_VECTOR_STORE", "faiss").strip().lower()
        if target in ("in_memory", "memory", "mock"):
            _GLOBAL_VECTOR_STORE = InMemoryVectorStore()
        else:
            _GLOBAL_VECTOR_STORE = FaissVectorStore(
                index_dir=index_dir,
                dimension=dimension,
            )
    return _GLOBAL_VECTOR_STORE


def reset_vector_store() -> None:
    """Utility function to clear vector store during tests."""
    global _GLOBAL_VECTOR_STORE
    if _GLOBAL_VECTOR_STORE is not None:
        _GLOBAL_VECTOR_STORE.clear()
    _GLOBAL_VECTOR_STORE = None
