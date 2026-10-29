import logging
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CareerLens AI Backend")
    env = os.getenv("ENVIRONMENT", "development").strip().lower()
    # Initialize database schema and apply pending migrations
    try:
        from backend.db.database import verify_database_connectivity
        from backend.db.migrations import run_migrations
        verify_database_connectivity()
        run_migrations()
    except Exception as e:
        if env == "production":
            logger.error("FATAL: Database initialization failed in production mode: %s", e)
            raise RuntimeError(f"Database initialization failed in production mode: {e}") from e
        logger.warning("Database schema check deferred in development: %s", e)
    yield
    logger.info("Shutting down CareerLens AI Backend")

app = FastAPI(
    title="CareerLens AI",
    description="AI-powered career intelligence platform for freshers and job seekers",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CareerLens AI Backend is running"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Routers
from backend.routers.cv import router as cv_router
from backend.routers.agent import router as agent_router
from backend.routers.auth import router as auth_router
from backend.routers.jobs import router as jobs_router

app.include_router(auth_router)
app.include_router(cv_router)
app.include_router(agent_router)
app.include_router(jobs_router)
