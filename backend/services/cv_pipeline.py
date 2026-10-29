import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from backend.db.database import Base, SessionLocal, engine
from backend.db.models import CandidateProfileRecord, Resume
from backend.models.schemas import CandidateProfile, StructuredCV
from backend.services.candidate_profile import CandidateProfileBuilder
from backend.services.chunker import ResumeChunker
from backend.services.cv_parser import CVParser
from backend.services.document_analyzer import DocumentAnalyzer, DocumentAnalysisResult
from backend.services.gemini import GeminiService
from backend.services.normalizer import DataNormalizer
from backend.services.retriever import ResumeRetriever, get_retriever
from backend.services.skill_ontology import SkillOntology

logger = logging.getLogger(__name__)


@dataclass
class CVPipelineResult:
    """Encapsulates the final result of CV pipeline processing and RAG indexing."""
    profile: CandidateProfile
    document_analysis: DocumentAnalysisResult
    resume_id: Optional[str] = None
    indexed: bool = False
    chunk_count: Optional[int] = None


class CVPipelineService:
    """
    Orchestration service coordinating CV parsing, document analysis,
    structured extraction, data normalization, skill ontology canonicalization,
    candidate profile building, and vector indexing for RAG retrieval.
    """

    def __init__(
        self,
        parser: Optional[CVParser] = None,
        analyzer: Optional[DocumentAnalyzer] = None,
        gemini_service: Optional[GeminiService] = None,
        normalizer: Optional[DataNormalizer] = None,
        skill_ontology: Optional[SkillOntology] = None,
        profile_builder: Optional[CandidateProfileBuilder] = None,
        chunker: Optional[ResumeChunker] = None,
        retriever: Optional[ResumeRetriever] = None,
    ):
        self.parser = parser or CVParser()
        self.analyzer = analyzer or DocumentAnalyzer()
        self.gemini_service = gemini_service or GeminiService()
        self.normalizer = normalizer or DataNormalizer()
        self.skill_ontology = skill_ontology or SkillOntology()
        self.profile_builder = profile_builder or CandidateProfileBuilder()
        self.chunker = chunker or ResumeChunker()
        self.retriever = retriever or get_retriever()

    def process_cv(
        self,
        file_bytes: bytes,
        user_id: Optional[str] = None,
        filename: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> CVPipelineResult:
        """
        Execute end-to-end CV processing pipeline on PDF bytes.

        Workflow:
        1. Parse uploaded PDF bytes using CVParser.parse_pdf() -> raw_text
        2. Extract sections using CVParser.extract_sections() -> parsed_sections
        3. Analyze document using DocumentAnalyzer.analyze() -> doc_analysis
        4. Extract structured data using GeminiService.extract_structured_cv() -> structured_cv
        5. Normalize structured data using DataNormalizer.normalize() -> normalized_cv
        6. Normalize skills and project technologies using SkillOntology.normalize_cv_skills() -> canonical_cv
        7. Build final profile using CandidateProfileBuilder.build() -> CandidateProfile
        8. Derive user-scoped deterministic resume ID (avoids cross-user collisions)
        9. Persist Resume and CandidateProfileRecord in SQLite
        10. Chunk and index into FAISS vector store

        Args:
            file_bytes: Raw bytes of the uploaded PDF file.
            user_id: Optional authenticated owner user ID.
            filename: Optional original upload filename.
            db: Optional active SQLAlchemy database session.

        Returns:
            CVPipelineResult containing CandidateProfile and DocumentAnalysisResult.

        Raises:
            ValueError: If PDF is empty, invalid, corrupted, or has no readable text.
        """
        byte_length = len(file_bytes) if file_bytes is not None else 0
        logger.info("Starting CV pipeline processing (%d bytes, user_id=%s)", byte_length, user_id)

        # 1. Parse PDF text
        raw_text = self.parser.parse_pdf(file_bytes)

        # 2. Extract sections
        sections = self.parser.extract_sections(raw_text)

        # 3. Analyze document using extracted text to avoid re-extraction
        doc_analysis = self.analyzer.analyze(file_bytes, extracted_text=raw_text)

        # 4. Extract structured data (Gemini with automated fallback)
        structured_cv = self.gemini_service.extract_structured_cv(raw_text)

        # 5. Normalize dates, qualifications, and grades
        normalized_cv = self.normalizer.normalize(structured_cv)

        # 6. Normalize skills and project technologies to canonical taxonomy
        canonical_cv = self.skill_ontology.normalize_cv_skills(normalized_cv)

        # 7. Build canonical candidate profile
        profile = self.profile_builder.build(
            parsed_sections=sections,
            document_analysis=doc_analysis,
            structured_cv=canonical_cv,
        )

        # 8. Scope deterministic hash to user_id to prevent cross-user collision.
        # If user_id is provided, identical text uploaded by different users produces distinct IDs.
        seed = f"{user_id}:{raw_text}" if user_id else raw_text
        resume_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()[:16]

        # 9. Persist/update Resume and CandidateProfileRecord in SQLite
        session = db if db is not None else SessionLocal()
        should_close = db is None
        try:
            bind = getattr(session, "bind", None) or engine
            Base.metadata.create_all(bind=bind)

            resume = session.query(Resume).filter(Resume.id == resume_id).first()
            safe_filename = filename or f"resume_{resume_id}.pdf"
            if not resume:
                resume = Resume(
                    id=resume_id,
                    user_id=user_id,
                    filename=safe_filename,
                    raw_text=raw_text,
                    processing_status="completed",
                    indexing_status="indexing",
                    chunk_count=0,
                )
                session.add(resume)
            else:
                if user_id and not resume.user_id:
                    resume.user_id = user_id
                resume.filename = safe_filename
                resume.raw_text = raw_text
                resume.processing_status = "completed"
                resume.indexing_status = "indexing"

            session.flush()

            if hasattr(profile, "model_dump_json"):
                profile_dict = json.loads(profile.model_dump_json())
            else:
                profile_dict = json.loads(profile.json())

            raw_doc_dict = doc_analysis.to_dict() if hasattr(doc_analysis, "to_dict") else doc_analysis.__dict__
            doc_dict = json.loads(json.dumps(raw_doc_dict, default=str))

            profile_rec = session.query(CandidateProfileRecord).filter(CandidateProfileRecord.resume_id == resume_id).first()
            if not profile_rec:
                profile_rec = CandidateProfileRecord(
                    resume_id=resume_id,
                    user_id=user_id,
                    profile_json=profile_dict,
                    doc_analysis_json=doc_dict,
                )
                session.add(profile_rec)
            else:
                profile_rec.user_id = user_id
                profile_rec.profile_json = profile_dict
                profile_rec.doc_analysis_json = doc_dict

            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("Failed to persist Resume/CandidateProfileRecord for %s: %s", resume_id, type(e).__name__)
            raise
        finally:
            if should_close:
                session.close()

        # 10. Chunk and index into FAISS vector store
        try:
            page_texts = self.parser.parse_pdf_pages(file_bytes)
        except Exception as e:
            logger.warning("Could not extract page-by-page text for resume %s: %s", resume_id, type(e).__name__)
            page_texts = None

        chunks = self.chunker.chunk(
            resume_id=resume_id,
            raw_text=raw_text,
            parsed_sections=sections,
            structured_cv=canonical_cv,
            page_texts=page_texts,
        )
        indexed_count = self.retriever.index_resume(resume_id=resume_id, chunks=chunks, user_id=user_id)

        logger.info(
            "CV pipeline completed: pages=%d, lang=%s, confidence=%s, skills=%d, edu=%d, exp=%d, indexed_chunks=%d",
            doc_analysis.page_count,
            doc_analysis.language,
            f"{doc_analysis.parser_confidence:.2f}" if doc_analysis.parser_confidence is not None else "None",
            len(canonical_cv.skills),
            len(canonical_cv.education),
            len(canonical_cv.experience),
            indexed_count,
        )

        return CVPipelineResult(
            profile=profile,
            document_analysis=doc_analysis,
            resume_id=resume_id,
            indexed=indexed_count > 0,
            chunk_count=len(chunks),
        )
