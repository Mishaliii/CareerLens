from pydantic import BaseModel, EmailStr, Field, field_validator
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum


class ExperienceLevel(str, Enum):
    ENTRY = "entry"
    MID = "mid"
    SENIOR = "senior"
    EXECUTIVE = "executive"


class JobSource(str, Enum):
    ADZUNA = "adzuna"
    JSEARCH = "jsearch"
    REMOTEOK = "remoteok"
    ARBEITNOW = "arbeitnow"


class TrackerStatus(str, Enum):
    SAVED = "saved"
    APPLIED = "applied"
    INTERVIEW = "interview"
    OFFER = "offer"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


class WorkPreference(str, Enum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    ANY = "any"


# CV Schemas
class CVCreate(BaseModel):
    user_id: str
    raw_text: str
    structured_json: Optional[Dict[str, Any]] = None
    file_url: str
    version: int = 1


class CVResponse(BaseModel):
    id: str
    user_id: str
    raw_text: str
    structured_json: Optional[Dict[str, Any]] = None
    file_url: str
    version: int
    created_at: datetime

    class Config:
        from_attributes = True


# Job Schemas
class JobCreate(BaseModel):
    title: str
    company: str
    location: str
    description: str
    source: JobSource
    experience_level: ExperienceLevel
    redirect_url: str
    date_posted: Optional[str] = None
    embedding: Optional[List[float]] = None


class JobResponse(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    source: JobSource
    experience_level: ExperienceLevel
    redirect_url: str
    date_posted: Optional[str] = None
    date_fetched: datetime
    embedding: Optional[List[float]] = None

    class Config:
        from_attributes = True


# Session Schemas (for matching/optimization results)
class SessionCreate(BaseModel):
    user_id: str
    job_id: str
    match_score: float = Field(ge=0, le=100)
    ats_score: float = Field(ge=0, le=100)
    skill_gap: Optional[List[str]] = None
    rewritten_cv_url: Optional[str] = None


class SessionResponse(BaseModel):
    id: str
    user_id: str
    job_id: str
    match_score: float
    ats_score: float
    skill_gap: Optional[List[str]] = None
    rewritten_cv_url: Optional[str] = None
    created_at: datetime

    class Config:
        from_attributes = True


# Interview Session Schemas
class InterviewSessionCreate(BaseModel):
    user_id: str
    job_id: str
    questions_answers: List[Dict[str, str]]
    dimension_scores: Dict[str, float]
    overall_score: float = Field(ge=0, le=100)
    difficulty: str


class InterviewSessionResponse(BaseModel):
    id: str
    user_id: str
    job_id: str
    questions_answers: List[Dict[str, str]]
    dimension_scores: Dict[str, float]
    overall_score: float
    difficulty: str
    created_at: datetime

    class Config:
        from_attributes = True


# Tracker Schemas
class TrackerCreate(BaseModel):
    user_id: str
    job_id: str
    status: TrackerStatus
    notes: Optional[str] = None


class TrackerUpdate(BaseModel):
    status: Optional[TrackerStatus] = None
    notes: Optional[str] = None


class TrackerResponse(BaseModel):
    id: str
    user_id: str
    job_id: str
    status: TrackerStatus
    notes: Optional[str] = None
    updated_at: datetime

    class Config:
        from_attributes = True


# Roadmap Schemas
class RoadmapCreate(BaseModel):
    user_id: str
    target_role: str
    milestones: List[Dict[str, Any]]
    completion: float = Field(ge=0, le=100, default=0)


class RoadmapResponse(BaseModel):
    id: str
    user_id: str
    target_role: str
    milestones: List[Dict[str, Any]]
    completion: float
    created_at: datetime

    class Config:
        from_attributes = True


class GradeScale(str, Enum):
    CGPA = "cgpa"
    GPA = "gpa"
    PERCENTAGE = "percentage"
    UNKNOWN = "unknown"


class AcademicGrade(BaseModel):
    value: float
    raw_value: Optional[str] = None
    scale: GradeScale = GradeScale.UNKNOWN
    max_scale: Optional[float] = None


class DatePrecision(str, Enum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"


class NormalizedDate(BaseModel):
    raw: str
    formatted: Optional[str] = None
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    precision: Optional[DatePrecision] = None
    is_current: bool = False


# Profile Schemas
class EducationItem(BaseModel):
    institution: str
    degree: str
    field: str
    start_year: Optional[int] = None
    end_year: Optional[int] = None
    gpa: Optional[float] = None
    raw_degree: Optional[str] = None
    normalized_degree: Optional[str] = None
    grade: Optional[AcademicGrade] = None


class ExperienceItem(BaseModel):
    company: str
    title: str
    start_date: str
    end_date: Optional[str] = None
    description: Optional[str] = None
    start_date_normalized: Optional[NormalizedDate] = None
    end_date_normalized: Optional[NormalizedDate] = None
    is_current: bool = False


class CertificationItem(BaseModel):
    name: str
    issuer: str
    date_obtained: str
    expiry_date: Optional[str] = None


class ProjectItem(BaseModel):
    name: str
    description: Optional[str] = None
    technologies: List[str] = Field(default_factory=list)
    link: Optional[str] = None


class StructuredCV(BaseModel):
    full_name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin_url: str = ""
    github_url: str = ""
    portfolio_url: str = ""
    summary: str = ""
    education: List[EducationItem] = Field(default_factory=list)
    experience: List[ExperienceItem] = Field(default_factory=list)
    projects: List[ProjectItem] = Field(default_factory=list)
    skills: List[str] = Field(default_factory=list)
    certifications: List[CertificationItem] = Field(default_factory=list)
    extraction_source: str = "fallback"


class ProfileCreate(BaseModel):
    user_id: str
    full_name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    target_title: Optional[str] = None
    years_experience: Optional[int] = Field(ge=0, default=0)
    expected_salary: Optional[int] = None
    work_preference: WorkPreference = WorkPreference.ANY
    education: Optional[List[EducationItem]] = None
    experience: Optional[List[ExperienceItem]] = None
    skills: Optional[List[str]] = None
    certifications: Optional[List[CertificationItem]] = None
    cover_letter_templates: Optional[Dict[str, str]] = None


class ProfileUpdate(BaseModel):
    full_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    target_title: Optional[str] = None
    years_experience: Optional[int] = Field(ge=0, default=None)
    expected_salary: Optional[int] = None
    work_preference: Optional[WorkPreference] = None
    education: Optional[List[EducationItem]] = None
    experience: Optional[List[ExperienceItem]] = None
    skills: Optional[List[str]] = None
    certifications: Optional[List[CertificationItem]] = None
    cover_letter_templates: Optional[Dict[str, str]] = None


class ProfileResponse(BaseModel):
    id: str
    user_id: str
    full_name: str
    email: str
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin_url: Optional[str] = None
    github_url: Optional[str] = None
    portfolio_url: Optional[str] = None
    target_title: Optional[str] = None
    years_experience: Optional[int] = None
    expected_salary: Optional[int] = None
    work_preference: WorkPreference
    education: Optional[List[EducationItem]] = None
    experience: Optional[List[ExperienceItem]] = None
    skills: Optional[List[str]] = None
    certifications: Optional[List[CertificationItem]] = None
    cover_letter_templates: Optional[Dict[str, str]] = None

    class Config:
        from_attributes = True


# Request/Response schemas for specific endpoints
class CVUploadResponse(BaseModel):
    cv_id: str
    message: str


class MatchRequest(BaseModel):
    user_id: str
    cv_id: str
    limit: int = Field(ge=1, le=50, default=10)


class MatchResponse(BaseModel):
    job_id: str
    title: str
    company: str
    location: str
    match_score: float
    ats_score: Optional[float] = None
    redirect_url: str


class OptimizationRequest(BaseModel):
    user_id: str
    cv_id: str
    job_id: str


class OptimizationResponse(BaseModel):
    rewritten_cv_url: str
    skill_gap: List[str]
    improvements_made: List[str]


class ATSCheckRequest(BaseModel):
    cv_text: str
    job_description: str


class ATSCheckResponse(BaseModel):
    overall_score: float
    keyword_score: float
    section_score: float
    parsability_score: float
    chronology_score: float
    quantification_score: float
    missing_keywords: List[str]
    missing_sections: List[str]
    suggestions: List[str]


class RoadmapRequest(BaseModel):
    user_id: str
    target_role: str
    current_skills: List[str]


class InterviewRequest(BaseModel):
    user_id: str
    job_id: str
    difficulty: str = "medium"


class InterviewResponse(BaseModel):
    questions: List[Dict[str, str]]
    difficulty: str
    session_id: str


class InterviewEvaluationRequest(BaseModel):
    session_id: str
    answers: List[Dict[str, str]]


class InterviewEvaluationResponse(BaseModel):
    dimension_scores: Dict[str, float]
    overall_score: float
    feedback: Dict[str, str]


class CoPilotRequest(BaseModel):
    user_id: str
    query: str


class CoPilotResponse(BaseModel):
    answer: str
    sources: List[str]


# Candidate Profile Schemas
class CandidateProfileMetadata(BaseModel):
    profile_schema_version: str = "1.0.0"
    parser_version: str = "1.0.0"
    parser_confidence: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    word_count: int = 0
    page_count: Optional[int] = None
    language: Optional[str] = None


class CandidateProfilePersonal(BaseModel):
    name: str = ""
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    portfolio: str = ""


class CandidateProfileStatistics(BaseModel):
    word_count: int = 0
    bullet_count: int = 0
    experience_count: int = 0
    project_count: int = 0
    certification_count: int = 0
    skills_count: int = 0


class CandidateProfile(BaseModel):
    metadata: CandidateProfileMetadata = Field(default_factory=CandidateProfileMetadata)
    personal: CandidateProfilePersonal = Field(default_factory=CandidateProfilePersonal)
    summary: str = ""
    education: str = ""
    experience: str = ""
    projects: str = ""
    skills: str = ""
    certifications: str = ""
    achievements: str = ""
    languages_spoken: str = ""
    resume_statistics: CandidateProfileStatistics = Field(default_factory=CandidateProfileStatistics)
    structured_data: Optional[StructuredCV] = None


# CV Processing Pipeline Schemas
class DocumentAnalysisResponse(BaseModel):
    page_count: int
    language: Optional[str] = None
    parser_confidence: Optional[float] = None
    total_characters: int
    total_words: int
    pages_with_text: int
    pages_sparse_or_empty: int
    character_density_per_page: float


class CVProcessResponse(BaseModel):
    success: bool = True
    message: str = "CV processed successfully"
    profile: CandidateProfile
    document_analysis: DocumentAnalysisResponse
    processing_time_ms: float
    resume_id: Optional[str] = None
    indexed: bool = False
    chunk_count: Optional[int] = None


# RAG & Resume Intelligence Schemas
class ResumeChunk(BaseModel):
    chunk_id: str
    resume_id: str
    section: str
    content: str
    item_title: Optional[str] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CitationSource(BaseModel):
    citation_id: int
    section: str
    item_title: Optional[str] = None
    page_number: Optional[int] = None
    snippet: str


class ResumeQueryRequest(BaseModel):
    resume_id: str
    query: str = Field(..., min_length=2, max_length=500, description="Question about the resume")
    top_k: int = Field(default=4, ge=1, le=10, description="Maximum number of evidence chunks to retrieve")
    user_id: Optional[str] = Field(default=None, description="Optional user ID for tracking; not used for auth")


class ResumeQueryResponse(BaseModel):
    success: bool = True
    answer: str
    citations: List[CitationSource] = Field(default_factory=list)
    has_sufficient_evidence: bool = True
    resume_id: str
    query: str
    processing_time_ms: float


class ResumeIndexStatusResponse(BaseModel):
    resume_id: str
    is_indexed: bool
    chunk_count: int = 0
    sections_indexed: List[str] = Field(default_factory=list)


class ResumeDeleteResponse(BaseModel):
    success: bool = True
    message: str
    resume_id: str


# Authentication & User Management Schemas
class UserRegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128, description="User password (8-128 chars)")
    full_name: Optional[str] = Field(default=None, max_length=255, description="Candidate's full name")

    @field_validator("password")
    @classmethod
    def validate_password_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Password cannot be blank or whitespace only")
        return v

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserLoginRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=1, max_length=128)

    @field_validator("email")
    @classmethod
    def normalize_email(cls, v: str) -> str:
        return v.strip().lower()


class UserResponse(BaseModel):
    id: str
    email: str
    full_name: Optional[str] = None
    is_active: bool = True
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse


# Job Intelligence & Career Gap Analysis Schemas
class RequirementOutcome(str, Enum):
    DIRECT_MATCH = "direct_match"
    RECOGNIZED_ALIAS = "recognized_alias"
    RELATED_SKILL = "related_skill"
    NOT_FOUND_IN_RESUME = "not_found_in_resume"
    DEMONSTRATED = "demonstrated"
    PARTIALLY_DEMONSTRATED = "partially_demonstrated"
    NOT_APPLICABLE = "not_applicable"


class ParsedJobDescription(BaseModel):
    job_title: Optional[str] = Field(default=None, description="Extracted job title if identifiable")
    company: Optional[str] = Field(default=None, description="Extracted company/employer name if identifiable")
    required_skills: List[str] = Field(default_factory=list, description="Mandatory technical skills extracted from job description")
    preferred_skills: List[str] = Field(default_factory=list, description="Preferred/nice-to-have technical skills")
    required_qualifications: List[str] = Field(default_factory=list, description="Mandatory education and credential requirements")
    preferred_qualifications: List[str] = Field(default_factory=list, description="Preferred education and credentials")
    minimum_experience_years: Optional[float] = Field(default=None, description="Minimum years of experience required")
    responsibilities: List[str] = Field(default_factory=list, description="Core job duties and responsibilities")
    other_requirements: List[str] = Field(default_factory=list, description="Other explicit job requirements (e.g. location, clearance)")
    raw_text: str = Field(..., description="Original job description text")
    extraction_source: str = Field(default="deterministic", description="Extraction engine: deterministic or gemini")


class RequirementMatch(BaseModel):
    requirement: str
    category: str = Field(..., description="Category: required_skills, preferred_skills, experience, education, other")
    outcome: str = Field(..., description="demonstrated, partially_demonstrated, not_found_in_resume, not_applicable")
    score: float = Field(..., ge=0.0, le=1.0, description="Normalized match score for this requirement")
    resume_evidence: Optional[str] = Field(default=None, description="Direct quote or proof from candidate profile/resume")
    source_section: Optional[str] = Field(default=None, description="Section where evidence was found (skills, experience, etc.)")
    explanation: str = Field(..., description="Detailed explanation of the match or gap")


class CategoryScore(BaseModel):
    category: str
    score: float = Field(..., ge=0.0, le=100.0, description="Raw percentage score for this category")
    weight: float = Field(..., ge=0.0, le=1.0, description="Relative category weight (0.0 to 1.0)")
    weighted_score: float = Field(..., ge=0.0, le=100.0, description="Category score scaled by normalized weight")
    total_items: int = Field(default=0)
    demonstrated_items: int = Field(default=0)
    partially_demonstrated_items: int = Field(default=0)
    missing_items: int = Field(default=0)
    explanation: str = Field(default="")


class CareerGapRecommendation(BaseModel):
    requirement: str
    gap_type: str = Field(..., description="'missing_evidence_in_resume' or 'skill_gap'")
    evidence_gap: str = Field(..., description="Description of what is missing in candidate evidence")
    suggested_action: str = Field(..., description="Actionable recommendation without fabricated claims or employment guarantees")
    action_type: str = Field(default="learning_and_practice", description="learning_and_practice, resume_clarification, or project_building")


class JobMatchRequest(BaseModel):
    resume_id: str = Field(..., min_length=4, max_length=64, description="Resume identifier belonging to the authenticated user")
    job_description: str = Field(..., min_length=20, max_length=50000, description="Job description text to evaluate")
    job_title: Optional[str] = Field(default=None, max_length=255, description="Optional override/hint for job title")
    company: Optional[str] = Field(default=None, max_length=255, description="Optional override/hint for company name")
    user_id: Optional[str] = Field(default=None, description="Optional user ID for client tracking; not used for auth")

    @field_validator("job_description")
    @classmethod
    def validate_job_description_not_blank(cls, v: str) -> str:
        if not v or not v.strip() or len(v.strip()) < 20:
            raise ValueError("Job description cannot be blank or shorter than 20 characters")
        return v.strip()


class JobMatchResponse(BaseModel):
    success: bool = True
    resume_id: str
    job_title: Optional[str] = None
    company: Optional[str] = None
    overall_score: float = Field(..., ge=0.0, le=100.0, description="Overall match score (0-100)")
    has_sufficient_evidence: bool = Field(default=True, description="False if resume or job description has insufficient extractable evidence")
    category_scores: Dict[str, CategoryScore] = Field(default_factory=dict)
    matched_requirements: List[RequirementMatch] = Field(default_factory=list)
    partially_demonstrated_requirements: List[RequirementMatch] = Field(default_factory=list)
    missing_requirements: List[RequirementMatch] = Field(default_factory=list)
    critical_gaps: List[RequirementMatch] = Field(default_factory=list, description="Mandatory job requirements not demonstrated in resume")
    contextual_evidence: List[RequirementMatch] = Field(default_factory=list, description="Related skills or background observed without awarding substitution credit")
    uncertain_requirements: List[str] = Field(default_factory=list, description="Requirements or employment dates that could not be reliably verified from text")
    recommendations: List[CareerGapRecommendation] = Field(default_factory=list)
    scoring_explanation: str
    limitations: List[str] = Field(default_factory=list)
    parsed_job: Optional[ParsedJobDescription] = None
