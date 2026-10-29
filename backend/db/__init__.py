"""
CareerLens Database Package.

Exports database engine, declarative Base, session dependency,
and SQLAlchemy models for persistent storage.
"""

from backend.db.database import Base, SessionLocal, engine, get_db
from backend.db.migrations import get_applied_migrations, run_migrations
from backend.db.models import (
    CandidateProfileRecord,
    Resume,
    ResumeChunkRecord,
    SchemaMigration,
    User,
)

__all__ = [
    "Base",
    "CandidateProfileRecord",
    "Resume",
    "ResumeChunkRecord",
    "SchemaMigration",
    "SessionLocal",
    "User",
    "engine",
    "get_applied_migrations",
    "get_db",
    "run_migrations",
]
