"""
CareerLens AI Backend — Application Entrypoint & Lifespan Management.

Configures:
- Production security pre-flight checks and fail-fast lifespan management.
- Dynamic CORS middleware based on validated environment configuration.
- Health and readiness probes (/health, /health/live, /health/ready).
- Sanitized exception handlers to prevent internal detail leaks in production.
- Versioned API routers (/api/v1/auth, /api/v1/cv, /api/v1/agent, /api/v1/jobs).
"""

from contextlib import asynccontextmanager
import logging
import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.exception_handlers import (
    http_exception_handler,
    request_validation_exception_handler,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.services.security_config import (
    get_environment,
    parse_cors_origins,
    validate_production_security,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager executing pre-flight validation on boot.

    In production mode:
    - Verifies JWT secret strength (rejects weak/default/missing secrets).
    - Verifies CORS configuration (rejects wildcards with credentials).
    - Verifies database connectivity.
    - Executes schema migrations.
    Fails fast with fatal error if any check fails.
    """
    logger.info("Starting CareerLens AI Backend")
    env = get_environment()
    try:
        # 1. Validate production security rules
        validate_production_security()

        # 2. Verify database connectivity
        from backend.db.database import verify_database_connectivity
        verify_database_connectivity()

        # 3. Apply database schema migrations
        from backend.db.migrations import run_migrations
        run_migrations()
    except Exception as e:
        if env == "production":
            logger.error("FATAL: Startup pre-flight checks failed in production mode: %s", e)
            raise RuntimeError(f"Startup pre-flight checks failed in production mode: {e}") from e
        logger.warning("Database schema check deferred in %s: %s", env, e)
    yield
    logger.info("Shutting down CareerLens AI Backend")


app = FastAPI(
    title="CareerLens AI",
    description="AI-powered career intelligence platform for freshers and job seekers",
    version="1.0.0",
    lifespan=lifespan,
)

# Dynamic CORS origins configuration
cors_origins = parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch-all exception handler to prevent implementation detail leakage in production.
    Preserves all standard HTTP exceptions (400, 401, 403, 404, 413, 429) and validation errors (422).
    """
    if isinstance(exc, (HTTPException, StarletteHTTPException)):
        return await http_exception_handler(request, exc)

    if isinstance(exc, RequestValidationError):
        return await request_validation_exception_handler(request, exc)

    logger.error("Unhandled server exception on %s %s: %s", request.method, request.url.path, type(exc).__name__, exc_info=True)
    if get_environment() == "production":
        return JSONResponse(
            status_code=500,
            content={"detail": "An internal server error occurred. Please try again later."},
        )
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal Server Error: {str(exc)}", "type": type(exc).__name__},
    )


# --- Root and Health/Readiness Probes ---

@app.get("/", tags=["General"])
async def root():
    return {"message": "CareerLens AI Backend is running"}


@app.get("/health", tags=["Health"])
async def health_check():
    """
    Standard health check endpoint ensuring backward compatibility.
    """
    return {"status": "healthy"}


@app.get("/health/live", tags=["Health"])
async def liveness_check():
    """
    Liveness probe confirming the application process is running and accepting HTTP requests.
    """
    return {
        "status": "alive",
        "version": "1.0.0",
        "environment": get_environment(),
    }


@app.get("/health/ready", tags=["Health"])
async def readiness_check():
    """
    Readiness probe verifying that required dependencies (database) are operational.

    Returns:
        HTTP 200 with {"status": "ready", "database": "connected"} when healthy.
        HTTP 503 with {"status": "not_ready", "database": "unavailable"} when unreachable.
    """
    from backend.db.database import verify_database_connectivity
    try:
        verify_database_connectivity(timeout_seconds=2.0)
        return {"status": "ready", "database": "connected"}
    except Exception as e:
        logger.warning("Readiness probe check failed: %s", e)
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "database": "unavailable"},
        )


# --- Versioned API Routers ---
from backend.routers.cv import router as cv_router
from backend.routers.agent import router as agent_router
from backend.routers.auth import router as auth_router
from backend.routers.jobs import router as jobs_router

app.include_router(auth_router)
app.include_router(cv_router)
app.include_router(agent_router)
app.include_router(jobs_router)
