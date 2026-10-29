"""
Database configuration and session management for CareerLens AI.

Configures SQLAlchemy 2.0 engine, declarative base, session factory,
and connection parameters across local SQLite and production PostgreSQL.

Key capabilities:
- Local SQLite persistence by default in development mode.
- PostgreSQL support with standard pooling and connection health checks.
- URL normalization (e.g. converting deprecated 'postgres://' to 'postgresql+psycopg2://').
- Strict production safety: fails loudly when production configuration is missing or invalid;
  never silently falls back to SQLite.
- Dialect-isolated pragma enforcement: executes SQLite WAL and foreign key pragmas ONLY
  for SQLite connections, never on PostgreSQL.
"""

import logging
import os
from typing import Generator, Optional, Tuple
from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import declarative_base, sessionmaker, Session

logger = logging.getLogger(__name__)

# Default to local SQLite database in project data/ directory
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "careerlens.db"
)
DATABASE_URL_ENV = "DATABASE_URL"
ENVIRONMENT_ENV = "ENVIRONMENT"


class DatabaseConfigurationError(ValueError):
    """Raised when database configuration is missing, malformed, or unsupported."""
    pass


class DatabaseConnectionError(RuntimeError):
    """Raised when active connectivity to the configured database fails."""
    pass


def _default_sqlite_url() -> str:
    """Return local SQLite database URL, ensuring directory exists."""
    db_dir = os.path.dirname(DEFAULT_DB_PATH)
    os.makedirs(db_dir, exist_ok=True)
    return f"sqlite:///{DEFAULT_DB_PATH}"


def resolve_database_url(
    url: Optional[str] = None,
    environment: Optional[str] = None,
) -> str:
    """
    Resolve, validate, and normalize the database URL based on the environment.

    Behavior:
    1. If `environment` is 'production':
       - Requires a non-empty `DATABASE_URL`.
       - Rejects defaulting to SQLite with DatabaseConfigurationError.
       - Rejects unconfigured or whitespace-only URLs.
    2. If `environment` is not 'production' (development/test):
       - If `url` is omitted or empty, defaults to local SQLite at DEFAULT_DB_PATH.
    3. Normalizes PostgreSQL schemes:
       - 'postgres://' -> 'postgresql+psycopg2://'
       - 'postgresql://' -> 'postgresql+psycopg2://'
    4. Validates supported schemes:
       - Must be 'sqlite://' or 'postgresql+psycopg2://' (or 'postgresql://').
       - Rejects unsupported schemes (e.g., 'mysql://', 'mongodb://').
    """
    raw_url = url if url is not None else os.getenv(DATABASE_URL_ENV)
    env = (environment if environment is not None else os.getenv(ENVIRONMENT_ENV, "development")).strip().lower()
    is_production = env == "production"

    clean_url = raw_url.strip() if raw_url else ""

    if not clean_url:
        if is_production:
            raise DatabaseConfigurationError(
                "DATABASE_URL environment variable is required in production mode. "
                "Silently defaulting to local SQLite in production is disabled for data safety."
            )
        return _default_sqlite_url()

    # Normalize deprecated postgres:// and standard postgresql:// schemes to explicit psycopg2 driver
    if clean_url.startswith("postgres://"):
        clean_url = clean_url.replace("postgres://", "postgresql+psycopg2://", 1)
    elif clean_url.startswith("postgresql://") and not clean_url.startswith("postgresql+psycopg2://"):
        clean_url = clean_url.replace("postgresql://", "postgresql+psycopg2://", 1)

    # Validate scheme
    if not (
        clean_url.startswith("sqlite:///")
        or clean_url.startswith("sqlite://")
        or clean_url.startswith("postgresql+psycopg2://")
        or clean_url.startswith("postgresql://")
    ):
        scheme = clean_url.split("://")[0] if "://" in clean_url else "unknown"
        raise DatabaseConfigurationError(
            f"Unsupported database scheme '{scheme}://'. "
            f"CareerLens supports 'sqlite:///' for local development and 'postgresql+psycopg2://' for production."
        )

    return clean_url


def _configure_sqlite_pragmas(eng: Engine) -> None:
    """Attach connection event listener for SQLite WAL mode, foreign keys, and busy timeout."""
    @event.listens_for(eng, "connect")
    def set_sqlite_pragmas(dbapi_connection, connection_record):
        # Double check dialect name or cursor presence
        if hasattr(dbapi_connection, "cursor"):
            cursor = dbapi_connection.cursor()
            try:
                # Enable foreign key constraint enforcement
                cursor.execute("PRAGMA foreign_keys = ON;")
                # Enable WAL mode for high concurrency
                cursor.execute("PRAGMA journal_mode = WAL;")
                # Wait up to 5000ms if database is locked by concurrent transaction
                cursor.execute("PRAGMA busy_timeout = 5000;")
            except Exception as e:
                logger.warning("Failed to configure SQLite pragmas: %s", e)
            finally:
                cursor.close()


def create_app_engine(
    database_url: Optional[str] = None,
    environment: Optional[str] = None,
) -> Engine:
    """
    Factory function creating an appropriate SQLAlchemy Engine for the target database.

    Configures:
    - SQLite: check_same_thread=False, PRAGMA listener for foreign keys and WAL.
    - PostgreSQL: pool_pre_ping=True, connection pooling (pool_size, max_overflow, pool_recycle),
      NO SQLite pragmas attached.
    """
    resolved_url = resolve_database_url(database_url, environment)

    if resolved_url.startswith("sqlite"):
        eng = create_engine(
            resolved_url,
            connect_args={"check_same_thread": False},
            pool_pre_ping=True,
        )
        _configure_sqlite_pragmas(eng)
        logger.info("Initialized SQLite engine: %s", resolved_url.split("?")[0])
        return eng

    elif resolved_url.startswith("postgresql"):
        pool_size = int(os.getenv("DB_POOL_SIZE", "10"))
        max_overflow = int(os.getenv("DB_MAX_OVERFLOW", "20"))
        pool_recycle = int(os.getenv("DB_POOL_RECYCLE", "1800"))

        eng = create_engine(
            resolved_url,
            pool_pre_ping=True,
            pool_size=pool_size,
            max_overflow=max_overflow,
            pool_recycle=pool_recycle,
        )
        # Redact credentials in logs
        sanitized_url = resolved_url.split("@")[-1] if "@" in resolved_url else "postgresql://[redacted]"
        logger.info("Initialized PostgreSQL engine with pool_size=%d, max_overflow=%d (@%s)", pool_size, max_overflow, sanitized_url)
        return eng

    else:
        raise DatabaseConfigurationError(f"Cannot initialize engine for URL: {resolved_url}")


def verify_database_connectivity(target_engine: Optional[Engine] = None, timeout_seconds: float = 3.0) -> bool:
    """
    Execute a lightweight ping query ('SELECT 1') to verify database reachability.

    Raises:
        DatabaseConnectionError: If connection cannot be established.
    """
    eng = target_engine or engine
    try:
        with eng.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as exc:
        env = os.getenv(ENVIRONMENT_ENV, "development").strip().lower()
        err_msg = f"Failed to connect to database ({eng.url.drivername}): {type(exc).__name__} - {str(exc)}"
        logger.error(err_msg)
        if env == "production":
            raise DatabaseConnectionError(err_msg) from exc
        raise DatabaseConnectionError(err_msg) from exc


def get_database_url() -> str:
    """Return the configured database URL from environment or default to SQLite."""
    return resolve_database_url()


# Default singleton module engine and sessionmaker
DATABASE_URL = get_database_url()

# Configure SQLite connect args for backward compatibility
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args["check_same_thread"] = False

engine = create_app_engine(DATABASE_URL)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False,
)

Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency yielding a managed SQLAlchemy database session.
    Guarantees session cleanup upon request completion.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
