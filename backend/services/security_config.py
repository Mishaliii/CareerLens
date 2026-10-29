"""
Production security configuration for CareerLens AI.

Centralizes and validates:
- Environment mode (development, test, production).
- JWT secret key, algorithms, and token expiration.
- Dynamic CORS origins (with strict validation against wildcard origins in production).
- Maximum upload size limits.
- Rate limiting thresholds.
"""

import logging
import os
import re
from typing import List, Optional, Set

logger = logging.getLogger(__name__)

# Development fallback secret (32+ bytes); strictly rejected in production
DEFAULT_DEV_SECRET = "careerlens-development-secret-key-min-32-bytes-long!"

# Known insecure/placeholder secrets rejected in production
KNOWN_INSECURE_SECRETS: Set[str] = {
    DEFAULT_DEV_SECRET.lower(),
    "your_jwt_secret_key_here",
    "change-me-in-production-min-32-bytes-long!",
    "secret",
    "supersecret",
    "password",
    "admin",
    "12345678901234567890123456789012",
}

DEFAULT_ALGORITHM = "HS256"
DEFAULT_ISSUER = "careerlens-auth"
DEFAULT_AUDIENCE = "careerlens-api"
DEFAULT_EXPIRE_MINUTES = 60
DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_DEV_CORS_ORIGINS = ["http://localhost:5173", "http://localhost:3000"]


class SecurityConfigurationError(ValueError):
    """Raised when security configuration is missing, weak, or invalid."""
    pass


def get_environment() -> str:
    """Return normalized environment mode ('development', 'test', 'production')."""
    return os.getenv("ENVIRONMENT", "development").strip().lower()


def is_production() -> bool:
    """Return True if running in production mode."""
    return get_environment() == "production"


def validate_jwt_secret(secret: Optional[str] = None, environment: Optional[str] = None) -> str:
    """
    Retrieve and strictly validate the JWT signing secret.

    Rules:
    - Production mode:
      - Secret is strictly required (raises SecurityConfigurationError if missing/empty).
      - Minimum length is 32 characters / bytes.
      - Reject known insecure placeholders (including DEFAULT_DEV_SECRET).
    - Non-production mode:
      - If missing or empty, safely falls back to DEFAULT_DEV_SECRET with a warning.
      - If provided but < 32 characters, logs a warning.
    """
    raw = secret if secret is not None else os.getenv("JWT_SECRET_KEY")
    env = (environment if environment is not None else get_environment()).strip().lower()

    clean_secret = raw.strip() if raw else ""

    if not clean_secret:
        if env == "production":
            raise SecurityConfigurationError(
                "CRITICAL: JWT_SECRET_KEY environment variable is required in production mode. "
                "Defaulting to insecure development secrets is forbidden."
            )
        logger.warning("JWT_SECRET_KEY not configured; using local development secret.")
        return DEFAULT_DEV_SECRET

    if env == "production":
        if clean_secret.lower() in KNOWN_INSECURE_SECRETS:
            raise SecurityConfigurationError(
                "CRITICAL: JWT_SECRET_KEY is using a known insecure or default placeholder value. "
                "A unique, cryptographically random secret of at least 32 characters is required."
            )
        if len(clean_secret) < 32:
            raise SecurityConfigurationError(
                f"CRITICAL: JWT_SECRET_KEY must be at least 32 characters long in production mode "
                f"(current length: {len(clean_secret)})."
            )

    return clean_secret


def get_token_expire_minutes() -> int:
    """
    Retrieve access token expiration in minutes.
    Checks JWT_ACCESS_TOKEN_EXPIRE_MINUTES, then ACCESS_TOKEN_EXPIRE_MINUTES.
    """
    val = os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES") or os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES")
    if not val:
        return DEFAULT_EXPIRE_MINUTES
    try:
        minutes = int(val.strip())
        if minutes < 1:
            raise ValueError()
        return minutes
    except ValueError:
        logger.warning("Invalid token expiration '%s'; using default %d minutes.", val, DEFAULT_EXPIRE_MINUTES)
        return DEFAULT_EXPIRE_MINUTES


def get_max_upload_size_bytes() -> int:
    """Retrieve maximum allowed upload size in bytes (defaults to 10 MiB)."""
    val = os.getenv("MAX_UPLOAD_SIZE_BYTES")
    if not val:
        return DEFAULT_MAX_UPLOAD_BYTES
    try:
        size = int(val.strip())
        if size < 1024:
            raise ValueError()
        return size
    except ValueError:
        logger.warning("Invalid MAX_UPLOAD_SIZE_BYTES '%s'; using default %d bytes.", val, DEFAULT_MAX_UPLOAD_BYTES)
        return DEFAULT_MAX_UPLOAD_BYTES


def parse_cors_origins(raw_origins: Optional[str] = None, environment: Optional[str] = None) -> List[str]:
    """
    Parse and validate allowed CORS origins from comma-separated string.

    Rules:
    - Production mode:
      - CORS_ORIGINS must be explicitly configured and non-empty.
      - Wildcard '*' is strictly forbidden when credentialed requests are enabled.
      - Each origin must be a valid http:// or https:// URL without path or trailing slash.
    - Development mode:
      - If omitted or empty, defaults to ['http://localhost:5173', 'http://localhost:3000'].
    """
    raw = raw_origins if raw_origins is not None else os.getenv("CORS_ORIGINS")
    env = (environment if environment is not None else get_environment()).strip().lower()

    if not raw or not raw.strip():
        if env == "production":
            raise SecurityConfigurationError(
                "CRITICAL: CORS_ORIGINS environment variable is required in production mode. "
                "Specify a comma-separated list of allowed domains (e.g. 'https://careerlens.pages.dev')."
            )
        return list(DEFAULT_DEV_CORS_ORIGINS)

    parts = [p.strip().rstrip("/") for p in raw.split(",") if p.strip()]
    if not parts:
        if env == "production":
            raise SecurityConfigurationError("CORS_ORIGINS cannot be empty in production mode.")
        return list(DEFAULT_DEV_CORS_ORIGINS)

    if "*" in parts:
        if env == "production":
            raise SecurityConfigurationError(
                "CRITICAL: Wildcard '*' CORS origin is forbidden in production when credentialed requests "
                "are enabled (RFC 6454 / Fetch security standard)."
            )

    origin_pattern = re.compile(r"^https?://[a-zA-Z0-9.\-_]+(:[0-9]+)?$")
    for origin in parts:
        if not origin_pattern.match(origin):
            raise SecurityConfigurationError(
                f"Invalid CORS origin format: '{origin}'. "
                f"Origins must be scheme + host (e.g. 'https://app.careerlens.ai' or 'http://localhost:5173')."
            )

    return parts


def validate_production_security() -> None:
    """
    Comprehensive startup validator. Fails fast if any security setting is unsafe in production.
    """
    if is_production():
        logger.info("Executing production security pre-flight checks...")
        validate_jwt_secret(environment="production")
        parse_cors_origins(environment="production")
        logger.info("Production security pre-flight checks passed successfully.")
