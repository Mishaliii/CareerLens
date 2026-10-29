import logging
import time
from fastapi import APIRouter, Depends, HTTPException, Path, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import CandidateProfileRecord, Resume, ResumeChunkRecord, User
from backend.models.schemas import (
    ResumeDeleteResponse,
    ResumeIndexStatusResponse,
    ResumeQueryRequest,
    ResumeQueryResponse,
)
from backend.services.auth_service import get_current_user
from backend.services.rate_limiter import rate_limit_agent_query
from backend.services.resume_agent import ResumeIntelligenceAgent, get_resume_agent
from backend.services.retriever import ResumeRetriever, get_retriever

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/agent/resume", tags=["Resume Intelligence Agent"])


def get_agent_service() -> ResumeIntelligenceAgent:
    """Dependency provider returning ResumeIntelligenceAgent."""
    return get_resume_agent()


def get_retriever_service() -> ResumeRetriever:
    """Dependency provider returning ResumeRetriever."""
    return get_retriever()


@router.post(
    "/query",
    response_model=ResumeQueryResponse,
    status_code=status.HTTP_200_OK,
    summary="Query Resume Intelligence Agent",
    description="Ask questions about an uploaded, indexed resume and receive grounded answers with citations.",
    dependencies=[Depends(rate_limit_agent_query)],
)
async def query_resume(
    request: ResumeQueryRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    agent: ResumeIntelligenceAgent = Depends(get_agent_service),
) -> ResumeQueryResponse:
    """
    Execute grounded RAG query over a specific resume belonging to the authenticated user.
    """
    # 1. Authoritative SQLite ownership check BEFORE embedding or LLM invocation
    resume = (
        db.query(Resume)
        .filter(Resume.id == request.resume_id, Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        # Uniform 404: never leak whether an unowned resume exists or belongs to another user
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found",
        )

    try:
        response = agent.answer_query(request)
        return response
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Unhandled error in resume agent query: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred while processing your resume query.",
        )


@router.get(
    "/{resume_id}/status",
    response_model=ResumeIndexStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Get Resume Index Status",
    description="Check whether a resume is currently indexed and inspect chunk counts for an authorized user.",
)
async def get_resume_status(
    resume_id: str = Path(..., description="Unique resume identifier", min_length=4, max_length=64),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    retriever: ResumeRetriever = Depends(get_retriever_service),
) -> ResumeIndexStatusResponse:
    """
    Retrieve indexing status and section breakdown for an authenticated user's resume.
    """
    # 1. Authoritative SQLite ownership check
    resume = (
        db.query(Resume)
        .filter(Resume.id == resume_id, Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found",
        )

    # 2. Get status from retriever / vector store
    status_info = retriever.get_index_status(resume_id)
    return ResumeIndexStatusResponse(
        resume_id=status_info.get("resume_id", resume_id),
        is_indexed=status_info.get("is_indexed", resume.indexing_status == "indexed"),
        chunk_count=status_info.get("chunk_count", resume.chunk_count or 0),
        sections_indexed=status_info.get("sections_indexed", []),
    )


@router.delete(
    "/{resume_id}",
    response_model=ResumeDeleteResponse,
    status_code=status.HTTP_200_OK,
    summary="Delete Resume and Vectors",
    description="Securely delete an uploaded resume, candidate profile, chunks, and FAISS vector index.",
)
async def delete_resume(
    resume_id: str = Path(..., description="Unique resume identifier", min_length=4, max_length=64),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    retriever: ResumeRetriever = Depends(get_retriever_service),
) -> ResumeDeleteResponse:
    """
    Securely delete resume, associated SQLite records, and FAISS vector index.
    """
    # 1. Authoritative ownership check
    resume = (
        db.query(Resume)
        .filter(Resume.id == resume_id, Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found",
        )

    # 2. Delete FAISS vector index and memory cache
    try:
        retriever.delete_resume_index(resume_id)
    except Exception as exc:
        logger.warning("Error deleting vector index for %s: %s", resume_id, type(exc).__name__)
        # Note: Do not leak filesystem paths or exception details to client.
        # Proceed with database deletion to revoke authorization immediately.

    # 3. Delete Resume from SQLite (guaranteeing cascading of candidate profile and chunks)
    try:
        db.query(CandidateProfileRecord).filter(CandidateProfileRecord.resume_id == resume_id).delete()
        db.query(ResumeChunkRecord).filter(ResumeChunkRecord.resume_id == resume_id).delete()
        db.delete(resume)
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Failed to delete resume record %s from database: %s", resume_id, type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while deleting the resume.",
        )

    return ResumeDeleteResponse(
        success=True,
        message="Resume and associated vector index deleted successfully",
        resume_id=resume_id,
    )
