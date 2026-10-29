"""
Authentication and security service for CareerLens.

Provides:
- Secure password hashing and constant-time verification using RFC 7914 Scrypt.
- Stateless JWT issuance, signing, and verification using PyJWT.
- FastAPI dependency for extracting and authenticating the current user.
- Defensive handling of malformed tokens, expired sessions, and inactive accounts.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import hmac
import logging
import os
import secrets
from typing import Any, Dict, Optional
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User
from backend.services.security_config import (
    DEFAULT_DEV_SECRET,
    DEFAULT_ALGORITHM,
    DEFAULT_ISSUER,
    DEFAULT_AUDIENCE,
    DEFAULT_EXPIRE_MINUTES,
    SecurityConfigurationError,
    validate_jwt_secret,
    get_token_expire_minutes,
)

logger = logging.getLogger(__name__)

# Security configuration environment variables
JWT_SECRET_KEY_ENV = "JWT_SECRET_KEY"
JWT_ALGORITHM_ENV = "JWT_ALGORITHM"
JWT_ISSUER_ENV = "JWT_ISSUER"
JWT_AUDIENCE_ENV = "JWT_AUDIENCE"
ACCESS_TOKEN_EXPIRE_MINUTES_ENV = "ACCESS_TOKEN_EXPIRE_MINUTES"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES_ENV = "JWT_ACCESS_TOKEN_EXPIRE_MINUTES"

# Scrypt parameters (NIST / RFC 7914 recommended)
SCRYPT_N = 16384  # CPU/memory cost
SCRYPT_R = 8      # Block size
SCRYPT_P = 1      # Parallelization
SCRYPT_DKLEN = 64 # 512-bit key length

# HTTP Bearer scheme with automatic 401 handling disabled for custom error messaging
http_bearer_scheme = HTTPBearer(auto_error=False)


def get_jwt_secret() -> str:
    """
    Retrieve and strictly validate JWT signing secret from environment.

    Raises SecurityConfigurationError if in production mode and secret is missing, weak, or placeholder.
    """
    return validate_jwt_secret()


def hash_password(plain_password: str) -> str:
    """
    Hash a plain password using RFC 7914 Scrypt with a cryptographically secure random salt.

    Format: scrypt$16384$8$1$<salt_hex>$<hash_hex>
    """
    if not plain_password:
        raise ValueError("Password must not be empty")

    salt = secrets.token_bytes(16)
    derived = hashlib.scrypt(
        plain_password.encode("utf-8"),
        salt=salt,
        n=SCRYPT_N,
        r=SCRYPT_R,
        p=SCRYPT_P,
        maxmem=0,
        dklen=SCRYPT_DKLEN,
    )
    return f"scrypt${SCRYPT_N}${SCRYPT_R}${SCRYPT_P}${salt.hex()}${derived.hex()}"


def verify_password(plain_password: str, stored_hash: str) -> bool:
    """
    Verify a password against a stored Scrypt hash in constant time.

    Safely catches and rejects malformed hashes, returning False without raising exceptions.
    """
    if not plain_password or not stored_hash:
        return False

    try:
        parts = stored_hash.split("$")
        if len(parts) != 6 or parts[0] != "scrypt":
            logger.warning("Attempted verification against malformed or unsupported password hash format")
            return False

        n = int(parts[1])
        r = int(parts[2])
        p = int(parts[3])
        salt = bytes.fromhex(parts[4])
        expected_hash = bytes.fromhex(parts[5])

        derived = hashlib.scrypt(
            plain_password.encode("utf-8"),
            salt=salt,
            n=n,
            r=r,
            p=p,
            maxmem=0,
            dklen=len(expected_hash),
        )
        # Constant-time comparison to prevent timing attacks
        return hmac.compare_digest(derived, expected_hash)
    except Exception as e:
        logger.warning("Password verification failed safely due to invalid hash encoding: %s", type(e).__name__)
        return False


def create_access_token(
    user_id: str,
    email: str,
    expires_delta: Optional[timedelta] = None,
    issuer: Optional[str] = None,
    audience: Optional[str] = None,
) -> str:
    """
    Create a signed JWT access token containing standard claims.
    """
    secret_key = get_jwt_secret()
    algorithm = os.getenv(JWT_ALGORITHM_ENV, DEFAULT_ALGORITHM)
    iss = issuer or os.getenv(JWT_ISSUER_ENV, DEFAULT_ISSUER)
    aud = audience or os.getenv(JWT_AUDIENCE_ENV, DEFAULT_AUDIENCE)

    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        minutes = int(os.getenv(ACCESS_TOKEN_EXPIRE_MINUTES_ENV, str(DEFAULT_EXPIRE_MINUTES)))
        expire = datetime.now(timezone.utc) + timedelta(minutes=minutes)

    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user_id),
        "email": str(email),
        "iss": iss,
        "aud": aud,
        "iat": int(now.timestamp()),
        "exp": int(expire.timestamp()),
    }

    return jwt.encode(payload, secret_key, algorithm=algorithm)


def decode_access_token(
    token: str,
    secret_key: Optional[str] = None,
    expected_issuer: Optional[str] = None,
    expected_audience: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Decode and validate a JWT access token.

    Enforces signature, expiration, algorithm, issuer, and audience.

    Raises:
        HTTPException: On expired, malformed, or untrusted tokens.
    """
    secret = secret_key or get_jwt_secret()
    algorithm = os.getenv(JWT_ALGORITHM_ENV, DEFAULT_ALGORITHM)
    iss = expected_issuer or os.getenv(JWT_ISSUER_ENV, DEFAULT_ISSUER)
    aud = expected_audience or os.getenv(JWT_AUDIENCE_ENV, DEFAULT_AUDIENCE)

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[algorithm],
            issuer=iss,
            audience=aud,
            options={
                "require": ["sub", "exp", "iat", "iss", "aud"],
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
        if not payload.get("sub"):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token subject (user ID) is missing",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return payload

    except jwt.ExpiredSignatureError:
        logger.info("Rejected expired JWT access token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        logger.warning("Rejected token with invalid issuer")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token issuer is invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        logger.warning("Rejected token with invalid audience")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token audience is invalid",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except (jwt.InvalidSignatureError, jwt.DecodeError) as e:
        logger.warning("Rejected token with invalid signature or decoding error: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.PyJWTError as e:
        logger.warning("JWT validation failed: %s", type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    auth: Optional[HTTPAuthorizationCredentials] = Depends(http_bearer_scheme),
    db: Session = Depends(get_db),
) -> User:
    """
    FastAPI dependency yielding the authenticated User model.

    1. Extracts Bearer token from the Authorization header.
    2. Decodes and verifies token claims.
    3. Fetches user record from SQLite.
    4. Rejects inactive or missing user accounts.
    """
    if not auth or not auth.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials were not provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(auth.credentials)
    user_id = payload.get("sub")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        logger.warning("Token references non-existent user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User account does not exist or has been deleted",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        logger.warning("Authentication rejected for inactive user_id=%s", user_id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive",
        )

    return user
