import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting CareerLens AI Backend")
    # Initialize BGE-M3, BGE-Reranker, and FAISS index here in future steps
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

# Routers will be included here in future steps
# from routers import cv, jobs, match, optimizer, builder, ats, roadmap, interview, copilot, tracker, apply, profile
