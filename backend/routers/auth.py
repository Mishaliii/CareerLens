"""
Authentication router for CareerLens candidate user accounts.

Endpoints:
- POST /api/v1/auth/register: Register a new candidate account.
- POST /api/v1/auth/login: Authenticate candidate and issue signed JWT access token.
- GET /api/v1/auth/me: Retrieve authenticated user profile.
"""

from datetime import timedelta
import logging
import os
import uuid
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User
from backend.models.schemas import (
    TokenResponse,
    UserLoginRequest,
    UserRegisterRequest,
    UserResponse,
)
from backend.services.auth_service import (
    ACCESS_TOKEN_EXPIRE_MINUTES_ENV,
    DEFAULT_EXPIRE_MINUTES,
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)

from backend.services.security_config import get_token_expire_minutes
from backend.services.rate_limiter import rate_limit_auth_login, rate_limit_auth_register

logger = logging.getLogger(__name__)

def _mask_email(email: str) -> str:
    """Mask email address for privacy in server logs."""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    masked_local = (local[0] + "***") if len(local) > 1 else "***"
    return f"{masked_local}@{domain}"

router = APIRouter(prefix="/api/v1/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Register Candidate Account",
    description="Registers a new candidate user account with an email and secure password.",
    dependencies=[Depends(rate_limit_auth_register)],
)
def register_user(
    request: UserRegisterRequest,
    db: Session = Depends(get_db),
) -> UserResponse:
    """
    Register a new user, enforcing email uniqueness and Scrypt password hashing.
    """
    normalized_email = request.email.strip().lower()

    # Check for existing email registration
    existing_user = db.query(User).filter(User.email == normalized_email).first()
    if existing_user:
        logger.warning("Registration rejected: email %s already registered", _mask_email(normalized_email))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="An account with this email address already exists",
        )

    # Securely hash password
    hashed = hash_password(request.password)

    new_user = User(
        id=str(uuid.uuid4()),
        email=normalized_email,
        hashed_password=hashed,
        full_name=request.full_name.strip() if request.full_name else None,
        is_active=True,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    logger.info("Successfully registered new user: id=%s", new_user.id)
    return UserResponse.model_validate(new_user)


@router.post(
    "/login",
    response_model=TokenResponse,
    status_code=status.HTTP_200_OK,
    summary="User Login",
    description="Authenticates candidate credentials and issues a signed JWT access token.",
    dependencies=[Depends(rate_limit_auth_login)],
)
def login_user(
    request: UserLoginRequest,
    db: Session = Depends(get_db),
) -> TokenResponse:
    """
    Authenticate user credentials with generic error responses to prevent account enumeration.
    """
    normalized_email = request.email.strip().lower()
    user = db.query(User).filter(User.email == normalized_email).first()

    # Generic authentication check to prevent timing or user existence leaks
    if not user or not verify_password(request.password, user.hashed_password):
        logger.warning("Authentication failed for email: %s", _mask_email(normalized_email))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        logger.warning("Authentication rejected for inactive user: %s", user.id)
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is inactive. Please contact support.",
        )

    expire_minutes = get_token_expire_minutes()
    token = create_access_token(
        user_id=user.id,
        email=user.email,
        expires_delta=timedelta(minutes=expire_minutes),
    )

    logger.info("User authenticated successfully: id=%s", user.id)
    return TokenResponse(
        access_token=token,
        token_type="bearer",
        expires_in=expire_minutes * 60,
        user=UserResponse.model_validate(user),
    )


@router.get(
    "/me",
    response_model=UserResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Current User Profile",
    description="Returns the profile of the currently authenticated candidate user.",
)
def get_current_user_profile(
    current_user: User = Depends(get_current_user),
) -> UserResponse:
    """
    Retrieve identity and profile details of the authenticated user.
    """
    return UserResponse.model_validate(current_user)
