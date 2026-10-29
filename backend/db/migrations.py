"""
Database migration and schema versioning management for CareerLens.

Provides deterministic migration execution and tracking using the
`schema_migrations` table, establishing a clean upgrade path to Alembic.
"""

import logging
from typing import List, Tuple
from sqlalchemy import inspect
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from backend.db.database import Base, engine as default_engine
from backend.db.models import SchemaMigration

logger = logging.getLogger(__name__)

# Registry of migrations in sequential order: (version, description)
MIGRATIONS_REGISTRY: List[Tuple[str, str]] = [
    ("001_initial_schema", "Initial schema: users, resumes, candidate_profiles, resume_chunks"),
]


def run_migrations(target_engine: Engine = default_engine) -> List[str]:
    """
    Apply any pending schema migrations and record them in `schema_migrations`.

    Returns:
        List of newly applied migration version strings.
    """
    logger.info("Checking database schema and migrations...")

    # Ensure all tables registered with declarative Base exist
    Base.metadata.create_all(bind=target_engine)

    applied_versions: List[str] = []
    with Session(target_engine) as session:
        # Query currently applied migrations
        existing_rows = session.query(SchemaMigration.version).all()
        existing_versions = {row[0] for row in existing_rows}

        for version, name in MIGRATIONS_REGISTRY:
            if version not in existing_versions:
                logger.info("Applying migration %s: %s", version, name)
                record = SchemaMigration(version=version, name=name)
                session.add(record)
                applied_versions.append(version)

        session.commit()

    if applied_versions:
        logger.info("Successfully applied %d migrations: %s", len(applied_versions), applied_versions)
    else:
        logger.info("Database schema is up to date. No pending migrations.")

    return applied_versions


def get_applied_migrations(target_engine: Engine = default_engine) -> List[Tuple[str, str]]:
    """Return all recorded migrations as (version, name) tuples."""
    with Session(target_engine) as session:
        rows = session.query(SchemaMigration.version, SchemaMigration.name).order_by(SchemaMigration.applied_at).all()
        return [(r[0], r[1]) for r in rows]
