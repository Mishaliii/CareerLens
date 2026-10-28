import logging
from dataclasses import dataclass
from typing import Optional

from backend.models.schemas import CandidateProfile, StructuredCV
from backend.services.candidate_profile import CandidateProfileBuilder
from backend.services.cv_parser import CVParser
from backend.services.document_analyzer import DocumentAnalyzer, DocumentAnalysisResult
from backend.services.gemini import GeminiService
from backend.services.normalizer import DataNormalizer
from backend.services.skill_ontology import SkillOntology

logger = logging.getLogger(__name__)


@dataclass
class CVPipelineResult:
    """Encapsulates the final result of CV pipeline processing."""
    profile: CandidateProfile
    document_analysis: DocumentAnalysisResult


class CVPipelineService:
    """
    Orchestration service coordinating CV parsing, document analysis,
    structured extraction, data normalization, skill ontology canonicalization,
    and candidate profile building.
    """

    def __init__(
        self,
        parser: Optional[CVParser] = None,
        analyzer: Optional[DocumentAnalyzer] = None,
        gemini_service: Optional[GeminiService] = None,
        normalizer: Optional[DataNormalizer] = None,
        skill_ontology: Optional[SkillOntology] = None,
        profile_builder: Optional[CandidateProfileBuilder] = None,
    ):
        self.parser = parser or CVParser()
        self.analyzer = analyzer or DocumentAnalyzer()
        self.gemini_service = gemini_service or GeminiService()
        self.normalizer = normalizer or DataNormalizer()
        self.skill_ontology = skill_ontology or SkillOntology()
        self.profile_builder = profile_builder or CandidateProfileBuilder()

    def process_cv(self, file_bytes: bytes) -> CVPipelineResult:
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

        Args:
            file_bytes: Raw bytes of the uploaded PDF file.

        Returns:
            CVPipelineResult containing CandidateProfile and DocumentAnalysisResult.

        Raises:
            ValueError: If PDF is empty, invalid, corrupted, or has no readable text.
        """
        byte_length = len(file_bytes) if file_bytes is not None else 0
        logger.info("Starting CV pipeline processing (%d bytes)", byte_length)

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

        logger.info(
            "CV pipeline completed: pages=%d, lang=%s, confidence=%s, skills=%d, edu=%d, exp=%d",
            doc_analysis.page_count,
            doc_analysis.language,
            f"{doc_analysis.parser_confidence:.2f}" if doc_analysis.parser_confidence is not None else "None",
            len(canonical_cv.skills),
            len(canonical_cv.education),
            len(canonical_cv.experience),
        )

        return CVPipelineResult(
            profile=profile,
            document_analysis=doc_analysis,
        )
