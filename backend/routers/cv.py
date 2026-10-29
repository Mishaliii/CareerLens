import logging
import time
from typing import Optional

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, status
from sqlalchemy.orm import Session

from backend.db.database import get_db
from backend.db.models import User
from backend.models.schemas import (
    CandidateProfile,
    CVProcessResponse,
    DocumentAnalysisResponse,
)
from backend.services.auth_service import get_current_user
from backend.services.cv_pipeline import CVPipelineService
from backend.services.rate_limiter import rate_limit_cv_process
from backend.services.security_config import get_max_upload_size_bytes

logger = logging.getLogger(__name__)

# Enforce default 10 MiB upload limit (10 * 1024 * 1024 bytes)
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024
READ_CHUNK_SIZE_BYTES = 64 * 1024  # 64 KiB chunks

router = APIRouter(prefix="/api/v1/cv", tags=["CV"])


def get_cv_pipeline() -> CVPipelineService:
    """Dependency provider for CVPipelineService."""
    return CVPipelineService()


@router.post(
    "/process",
    response_model=CVProcessResponse,
    status_code=status.HTTP_200_OK,
    summary="Process CV PDF",
    description="Extracts, analyzes, normalizes, and structures candidate CV data from an uploaded PDF.",
    dependencies=[Depends(rate_limit_cv_process)],
)
async def process_cv(
    request: Request,
    file: UploadFile = File(..., description="PDF document of the candidate CV (max 10 MiB)."),
    pipeline: CVPipelineService = Depends(get_cv_pipeline),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
) -> CVProcessResponse:
    """
    Handle CV upload, validate PDF structure and size, execute pipeline, and return CandidateProfile.
    """
    start_time = time.perf_counter()
    configured_limit = get_max_upload_size_bytes()
    max_size_bytes = MAX_FILE_SIZE_BYTES if MAX_FILE_SIZE_BYTES != 10 * 1024 * 1024 else configured_limit
    limit_mib = max(1, max_size_bytes // (1024 * 1024))
    limit_detail = f"File size exceeds maximum allowed limit of {limit_mib} MiB"

    try:
        # 1. Early Content-Length check if provided by client headers
        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
                if content_length > max_size_bytes:
                    logger.warning("Upload rejected early: Content-Length %d exceeds limit", content_length)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=limit_detail,
                    )
            except ValueError:
                pass

        # 2. Bounded stream reading to detect oversized files without unbounded memory allocation
        file_bytes = bytearray()
        while True:
            chunk = await file.read(READ_CHUNK_SIZE_BYTES)
            if not chunk:
                break
            file_bytes.extend(chunk)
            if len(file_bytes) > max_size_bytes:
                logger.warning("Upload rejected: stream exceeded limit (%d bytes read)", len(file_bytes))
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=limit_detail,
                )

        # 3. Empty file check
        if len(file_bytes) == 0:
            logger.warning("Upload rejected: empty file received")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty",
            )

        # 4. MIME type supporting check: reject clear non-PDF types
        if file.content_type:
            mime = file.content_type.strip().lower()
            if mime not in ["application/pdf", "application/x-pdf", "application/octet-stream"]:
                logger.warning("Upload rejected: invalid client MIME type '%s'", mime)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file must be a valid PDF document",
                )

        # 5. PDF signature validation
        if not file_bytes.startswith(b"%PDF"):
            logger.warning("Upload rejected: missing or invalid PDF signature")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must be a valid PDF document",
            )

        # 6. Execute CV pipeline with authenticated user context
        try:
            result = pipeline.process_cv(
                bytes(file_bytes),
                user_id=current_user.id,
                filename=file.filename or f"resume_{current_user.id}.pdf",
                db=db,
            )
        except ValueError as ve:
            err_msg = str(ve).lower()
            logger.warning("Pipeline rejected unprocessable PDF: %s", type(ve).__name__)
            if "empty" in err_msg or "too short" in err_msg:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Uploaded file must be a valid PDF document",
                )
            # Map corrupted PDFs, missing pages, or unreadable text layers to 422
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="PDF contains no readable text or is corrupted",
            )
        except HTTPException:
            raise
        except Exception as exc:
            # Conceal internal traceback and sensitive details
            logger.error("Unexpected error in CV processing pipeline: %s", type(exc).__name__)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal error occurred while processing the CV",
            )

        elapsed_ms = max(0.0, round((time.perf_counter() - start_time) * 1000.0, 2))

        return CVProcessResponse(
            success=True,
            message="CV processed successfully",
            profile=result.profile,
            document_analysis=DocumentAnalysisResponse(
                page_count=result.document_analysis.page_count,
                language=result.document_analysis.language,
                parser_confidence=result.document_analysis.parser_confidence,
                total_characters=result.document_analysis.total_characters,
                total_words=result.document_analysis.total_words,
                pages_with_text=result.document_analysis.pages_with_text,
                pages_sparse_or_empty=result.document_analysis.pages_sparse_or_empty,
                character_density_per_page=result.document_analysis.character_density_per_page,
            ),
            processing_time_ms=elapsed_ms,
            resume_id=result.resume_id,
            indexed=result.indexed,
            chunk_count=result.chunk_count,
        )

    finally:
        # Guarantee resource cleanup
        await file.close()
