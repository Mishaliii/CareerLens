"""
Job Intelligence and Career Gap Analysis Router for CareerLens AI.

Provides versioned endpoints:
- POST /api/v1/jobs/match: Evaluates candidate alignment against a supplied job description.

Security:
- Requires mandatory JWT Bearer authentication.
- Verifies resume ownership strictly in SQLite before parsing or matching.
- Rejects unowned or non-existent resumes with uniform 404 Not Found.
- Never accepts request-body user_id as an authorization source.
"""

import logging
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import CandidateProfileRecord, Resume, ResumeChunkRecord, User
from backend.models.schemas import (
    CandidateProfile,
    CandidateProfileMetadata,
    CandidateProfilePersonal,
    CandidateProfileStatistics,
    JobMatchRequest,
    JobMatchResponse,
    ResumeChunk,
)
from backend.services.auth_service import get_current_user
from backend.services.job_matcher import JobMatcherService, get_job_matcher
from backend.services.job_parser import JobParserService, get_job_parser
from backend.services.rate_limiter import rate_limit_job_match

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/jobs", tags=["Job Intelligence"])


def get_job_parser_service() -> JobParserService:
    """Dependency provider returning JobParserService."""
    return get_job_parser()


def get_job_matcher_service() -> JobMatcherService:
    """Dependency provider returning JobMatcherService."""
    return get_job_matcher()


@router.post(
    "/match",
    response_model=JobMatchResponse,
    status_code=status.HTTP_200_OK,
    summary="Match Resume against Job Description",
    description="Evaluates a candidate resume against a job description, computing category and overall match scores, identified gaps, and actionable recommendations.",
    dependencies=[Depends(rate_limit_job_match)],
)
async def match_job_description(
    request: JobMatchRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
    job_parser: JobParserService = Depends(get_job_parser_service),
    job_matcher: JobMatcherService = Depends(get_job_matcher_service),
) -> JobMatchResponse:
    """
    Execute grounded job alignment assessment for an authenticated user's resume.
    """
    # 1. Authoritative SQLite ownership check
    resume = (
        db.query(Resume)
        .filter(Resume.id == request.resume_id, Resume.user_id == current_user.id)
        .first()
    )
    if not resume:
        # Uniform 404: never leak whether resume exists for another user
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Resume not found",
        )

    # 2. Retrieve CandidateProfile
    profile_record = (
        db.query(CandidateProfileRecord)
        .filter(CandidateProfileRecord.resume_id == request.resume_id)
        .first()
    )

    if profile_record and profile_record.profile_json:
        try:
            raw_p = profile_record.profile_json
            if isinstance(raw_p, str):
                candidate_profile = CandidateProfile.model_validate_json(raw_p)
            else:
                candidate_profile = CandidateProfile.model_validate(raw_p)
        except Exception as e:
            logger.warning("Could not deserialize stored profile_json for %s: %s", request.resume_id, e)
            candidate_profile = _build_minimal_profile(resume)
    else:
        candidate_profile = _build_minimal_profile(resume)

    # 3. Retrieve Resume Chunks for supporting evidence
    chunk_records = (
        db.query(ResumeChunkRecord)
        .filter(ResumeChunkRecord.resume_id == request.resume_id)
        .order_by(ResumeChunkRecord.chunk_index)
        .all()
    )
    resume_chunks = [
        ResumeChunk(
            chunk_id=r.id,
            resume_id=r.resume_id,
            section=r.section,
            content=r.content,
            item_title=r.item_title,
            page_number=r.page_number,
            metadata=r.metadata_json or {},
        )
        for r in chunk_records
    ]

    # 4. Parse the Job Description
    try:
        parsed_job = job_parser.parse(
            raw_text=request.job_description,
            job_title_hint=request.job_title,
            company_hint=request.company,
        )
    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(ve),
        )
    except Exception as exc:
        logger.error("Job parsing unexpected error: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while parsing the job description.",
        )

    # 5. Execute Matching Engine
    try:
        match_result = job_matcher.match(
            parsed_job=parsed_job,
            candidate_profile=candidate_profile,
            resume_chunks=resume_chunks,
        )
    except Exception as exc:
        logger.error("Job matching unexpected error: %s", type(exc).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating job match.",
        )

    return JobMatchResponse(
        success=True,
        resume_id=request.resume_id,
        job_title=parsed_job.job_title,
        company=parsed_job.company,
        overall_score=match_result["overall_score"],
        has_sufficient_evidence=match_result["has_sufficient_evidence"],
        category_scores=match_result["category_scores"],
        matched_requirements=match_result["matched_requirements"],
        partially_demonstrated_requirements=match_result["partially_demonstrated_requirements"],
        missing_requirements=match_result["missing_requirements"],
        critical_gaps=match_result.get("critical_gaps", []),
        contextual_evidence=match_result.get("contextual_evidence", []),
        uncertain_requirements=match_result.get("uncertain_requirements", []),
        recommendations=match_result["recommendations"],
        scoring_explanation=match_result["scoring_explanation"],
        limitations=match_result["limitations"],
        parsed_job=parsed_job,
    )


def _build_minimal_profile(resume: Resume) -> CandidateProfile:
    """Construct a minimal fallback profile if candidate profile is missing."""
    return CandidateProfile(
        metadata=CandidateProfileMetadata(
            schema_version="1.0.0",
            parser_version="1.0.0",
            word_count=len((resume.raw_text or "").split()),
            page_count=1,
            language="en",
        ),
        personal=CandidateProfilePersonal(),
        summary="",
        education="",
        experience="",
        projects="",
        skills=resume.raw_text or "",
        certifications="",
        achievements="",
        languages_spoken="",
        resume_statistics=CandidateProfileStatistics(),
        structured_data=None,
    )
