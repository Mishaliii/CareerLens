"""
SQLAlchemy 2.0 relational models for CareerLens persistent storage.

Entities:
- User: Candidate account and authentication identity.
- Resume: Document metadata, processing status, and embedding lifecycle.
- CandidateProfileRecord: Structured candidate profile and document analysis JSON.
- ResumeChunkRecord: Semantic resume chunks with metadata and serialized vector blobs for index recovery.
- SchemaMigration: Tracks applied schema migration versions.
"""

import math
import struct
import uuid
from typing import Any, Dict, List, Optional
import numpy as np
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    JSON,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import relationship

from backend.db.database import Base


class User(Base):
    """Candidate account and identity record."""
    __tablename__ = "users"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(255), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships with cascading deletion
    resumes = relationship(
        "Resume",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    candidate_profiles = relationship(
        "CandidateProfileRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    resume_chunks = relationship(
        "ResumeChunkRecord",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"


class Resume(Base):
    """
    Resume document lifecycle tracking.

    Tracks processing state (pending, extracted, failed) and RAG indexing state (unindexed, indexed, stale).
    """
    __tablename__ = "resumes"

    id = Column(String(64), primary_key=True)  # Deterministic hash or UUID
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    filename = Column(String(255), nullable=False)
    raw_text = Column(Text, nullable=True)
    processing_status = Column(String(32), default="pending", nullable=False, index=True)
    indexing_status = Column(String(32), default="unindexed", nullable=False, index=True)
    embedding_model = Column(String(128), nullable=True)
    embedding_dimension = Column(Integer, nullable=True)
    chunk_count = Column(Integer, default=0, nullable=False)
    error_message = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    user = relationship("User", back_populates="resumes")
    profile = relationship(
        "CandidateProfileRecord",
        back_populates="resume",
        uselist=False,
        cascade="all, delete-orphan",
        passive_deletes=True,
    )
    chunks = relationship(
        "ResumeChunkRecord",
        back_populates="resume",
        cascade="all, delete-orphan",
        passive_deletes=True,
        order_by="ResumeChunkRecord.chunk_index",
    )

    def __repr__(self) -> str:
        return f"<Resume(id={self.id}, user_id={self.user_id}, status={self.processing_status})>"


class CandidateProfileRecord(Base):
    """Stores structured candidate profile and document analysis payloads."""
    __tablename__ = "candidate_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    resume_id = Column(
        String(64),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    profile_json = Column(JSON, nullable=False)
    doc_analysis_json = Column(JSON, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)

    # Relationships
    resume = relationship("Resume", back_populates="profile")
    user = relationship("User", back_populates="candidate_profiles")

    def __repr__(self) -> str:
        return f"<CandidateProfileRecord(id={self.id}, resume_id={self.resume_id})>"


class ResumeChunkRecord(Base):
    """
    Section-aware resume chunks for vector retrieval and grounded citations.

    Includes an optional `embedding_blob` column (serialized float32 array)
    used to rebuild external FAISS indexes without re-running embedding models.
    """
    __tablename__ = "resume_chunks"

    id = Column(String(128), primary_key=True)
    resume_id = Column(
        String(64),
        ForeignKey("resumes.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    user_id = Column(
        String(36),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False)
    section = Column(String(64), nullable=False, index=True)
    item_title = Column(String(255), nullable=True)
    page_number = Column(Integer, nullable=True)
    content = Column(Text, nullable=False)
    metadata_json = Column(JSON, nullable=False, default=dict)
    vector_index_id = Column(Integer, nullable=True)
    embedding_blob = Column(LargeBinary, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    __table_args__ = (
        Index("idx_resume_chunk_order", "resume_id", "chunk_index"),
        Index("idx_user_resume_chunks", "user_id", "resume_id"),
        UniqueConstraint("resume_id", "vector_index_id", name="uq_resume_vector_index_id"),
    )

    # Relationships
    resume = relationship("Resume", back_populates="chunks")
    user = relationship("User", back_populates="resume_chunks")

    def set_embedding(
        self,
        vector: List[float],
        expected_dimension: Optional[int] = None,
    ) -> None:
        """
        Validate vector dimensionality and serialize float32 array to binary bytes.

        Raises:
            ValueError: If vector is empty, contains non-finite floats (NaN/Inf),
                        or fails expected dimension constraints.
        """
        if not vector:
            raise ValueError("Embedding vector must not be empty")

        if expected_dimension is not None and len(vector) != expected_dimension:
            raise ValueError(
                f"Embedding dimension mismatch: expected {expected_dimension}, got {len(vector)}"
            )

        for val in vector:
            if math.isnan(val) or math.isinf(val):
                raise ValueError("Embedding vector contains non-finite values (NaN or Inf)")

        # Convert to float32 numpy array and serialize to raw bytes
        arr = np.array(vector, dtype=np.float32)
        self.embedding_blob = arr.tobytes()

    def get_embedding(self) -> Optional[List[float]]:
        """Deserialize stored binary blob back into a list of Python floats."""
        if not self.embedding_blob:
            return None
        arr = np.frombuffer(self.embedding_blob, dtype=np.float32)
        return arr.tolist()

    def __repr__(self) -> str:
        return f"<ResumeChunkRecord(id={self.id}, resume_id={self.resume_id}, section={self.section})>"


class SchemaMigration(Base):
    """Tracks applied schema migrations for version integrity."""
    __tablename__ = "schema_migrations"

    version = Column(String(64), primary_key=True)
    name = Column(String(255), nullable=False)
    applied_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    def __repr__(self) -> str:
        return f"<SchemaMigration(version={self.version}, name={self.name})>"
