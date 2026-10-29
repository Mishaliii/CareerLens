"""
Job matching engine and career gap analysis service for CareerLens AI.

Compares structured job requirements against stored candidate profiles and resume evidence:
- Direct skill matches, recognized aliases, documented related skills, and contextual evidence.
- Transparent, deterministic scoring with documented category weights.
- Distinguishes missing resume evidence from confirmed skill deficiencies.
- Actionable career gap recommendations without fabricated claims.
- Insufficient evidence detection.
- Non-overlapping professional experience interval calculations.
- Prominent visibility for mandatory/critical requirement gaps.
"""

from dataclasses import dataclass
from datetime import datetime
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.models.schemas import (
    CandidateProfile,
    CategoryScore,
    CareerGapRecommendation,
    ParsedJobDescription,
    RequirementMatch,
    RequirementOutcome,
    ResumeChunk,
    StructuredCV,
)
from backend.services.skill_ontology import SkillOntology

logger = logging.getLogger(__name__)

# Category Weights (Documented and Fixed)
DEFAULT_CATEGORY_WEIGHTS = {
    "required_skills": 0.45,
    "experience": 0.25,
    "education": 0.15,
    "preferred_skills": 0.15,
}


@dataclass(frozen=True)
class SkillRelationshipRule:
    """Documented relationship rule between a target requirement and candidate skill."""
    target_skill: str           # Target skill required in job description (canonical)
    candidate_skill: str        # Skill demonstrated by candidate (canonical)
    relation_type: str          # e.g., "framework_transferability", "cloud_hyperscaler", "database_paradigm"
    transferability_score: float# 0.5 for proven transferability, 0.0 for contextual evidence only
    rationale: str              # Transparent explainable reasoning


# Documented Skill Relationships (Explicit, Non-Arbitrary)
# Rules must be based on verified architectural/language commonality rather than broad category co-membership.
DOCUMENTED_SKILL_RELATIONSHIPS: Dict[Tuple[str, str], SkillRelationshipRule] = {
    # 1. Python Web Framework Transferability (0.5 credit)
    ("FastAPI", "Django"): SkillRelationshipRule(
        target_skill="FastAPI",
        candidate_skill="Django",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates backend architecture in Django. Core Python language, routing, ORM, and REST concepts transfer directly to FastAPI with brief ramp-up.",
    ),
    ("Django", "FastAPI"): SkillRelationshipRule(
        target_skill="Django",
        candidate_skill="FastAPI",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates backend architecture in FastAPI. Core Python language, routing, HTTP handling, and async patterns transfer to Django.",
    ),
    ("FastAPI", "Flask"): SkillRelationshipRule(
        target_skill="FastAPI",
        candidate_skill="Flask",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates microframework development in Flask. Lightweight REST routing and WSGI/ASGI paradigms transfer directly to FastAPI.",
    ),
    ("Flask", "FastAPI"): SkillRelationshipRule(
        target_skill="Flask",
        candidate_skill="FastAPI",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates FastAPI development. Routing, request validation, and HTTP API design transfer directly to Flask.",
    ),
    ("Django", "Flask"): SkillRelationshipRule(
        target_skill="Django",
        candidate_skill="Flask",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates Python web development in Flask. Core Python backend patterns transfer to Django.",
    ),
    ("Flask", "Django"): SkillRelationshipRule(
        target_skill="Flask",
        candidate_skill="Django",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented framework transferability: Candidate demonstrates Django development. Web endpoints and request handling transfer directly to Flask.",
    ),
    ("Express.js", "Node.js"): SkillRelationshipRule(
        target_skill="Express.js",
        candidate_skill="Node.js",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented runtime-framework transferability: Candidate demonstrates Node.js runtime proficiency; HTTP server fundamentals transfer to Express.js.",
    ),
    ("Node.js", "Express.js"): SkillRelationshipRule(
        target_skill="Node.js",
        candidate_skill="Express.js",
        relation_type="framework_transferability",
        transferability_score=0.5,
        rationale="Documented runtime-framework transferability: Candidate demonstrates Express.js framework; server-side JavaScript fundamentals transfer to Node.js core.",
    ),

    # 2. Cloud Hyperscaler Transferability (0.5 credit)
    ("Google Cloud", "AWS"): SkillRelationshipRule(
        target_skill="Google Cloud",
        candidate_skill="AWS",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on AWS. Compute (EC2/GCE), storage (S3/GCS), networking (VPC), and IAM primitives transfer directly to Google Cloud.",
    ),
    ("AWS", "Google Cloud"): SkillRelationshipRule(
        target_skill="AWS",
        candidate_skill="Google Cloud",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on Google Cloud. Compute, storage, networking, and IAM primitives transfer directly to AWS.",
    ),
    ("Microsoft Azure", "AWS"): SkillRelationshipRule(
        target_skill="Microsoft Azure",
        candidate_skill="AWS",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on AWS. Compute, storage, networking, and IAM primitives transfer directly to Microsoft Azure.",
    ),
    ("AWS", "Microsoft Azure"): SkillRelationshipRule(
        target_skill="AWS",
        candidate_skill="Microsoft Azure",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on Microsoft Azure. Core cloud primitives transfer directly to AWS.",
    ),
    ("Google Cloud", "Microsoft Azure"): SkillRelationshipRule(
        target_skill="Google Cloud",
        candidate_skill="Microsoft Azure",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on Microsoft Azure. Core cloud primitives transfer directly to Google Cloud.",
    ),
    ("Microsoft Azure", "Google Cloud"): SkillRelationshipRule(
        target_skill="Microsoft Azure",
        candidate_skill="Google Cloud",
        relation_type="cloud_hyperscaler",
        transferability_score=0.5,
        rationale="Documented cloud provider transferability: Candidate demonstrates cloud architecture on Google Cloud. Core cloud primitives transfer directly to Microsoft Azure.",
    ),

    # 3. Relational RDBMS Transferability (0.5 credit)
    ("PostgreSQL", "MySQL"): SkillRelationshipRule(
        target_skill="PostgreSQL",
        candidate_skill="MySQL",
        relation_type="relational_rdbms",
        transferability_score=0.5,
        rationale="Documented relational database transferability: Candidate demonstrates relational RDBMS in MySQL. Standard SQL, relational schemas, indexing, and ACID transactions transfer to PostgreSQL.",
    ),
    ("MySQL", "PostgreSQL"): SkillRelationshipRule(
        target_skill="MySQL",
        candidate_skill="PostgreSQL",
        relation_type="relational_rdbms",
        transferability_score=0.5,
        rationale="Documented relational database transferability: Candidate demonstrates relational RDBMS in PostgreSQL. Standard SQL, relational schemas, indexing, and ACID transactions transfer to MySQL.",
    ),
    ("PostgreSQL", "SQLite"): SkillRelationshipRule(
        target_skill="PostgreSQL",
        candidate_skill="SQLite",
        relation_type="relational_rdbms",
        transferability_score=0.5,
        rationale="Documented relational database transferability: Candidate demonstrates SQL relational databases in SQLite. Relational querying and schema concepts transfer to PostgreSQL.",
    ),
    ("SQLite", "PostgreSQL"): SkillRelationshipRule(
        target_skill="SQLite",
        candidate_skill="PostgreSQL",
        relation_type="relational_rdbms",
        transferability_score=0.5,
        rationale="Documented relational database transferability: Candidate demonstrates client-server SQL in PostgreSQL. Relational querying concepts transfer to SQLite.",
    ),

    # 4. Deep Learning Framework Transferability (0.5 credit)
    ("PyTorch", "TensorFlow"): SkillRelationshipRule(
        target_skill="PyTorch",
        candidate_skill="TensorFlow",
        relation_type="deep_learning",
        transferability_score=0.5,
        rationale="Documented deep learning framework transferability: Candidate demonstrates deep learning in TensorFlow. Tensor computations, autodiff, and layer architectures transfer to PyTorch.",
    ),
    ("TensorFlow", "PyTorch"): SkillRelationshipRule(
        target_skill="TensorFlow",
        candidate_skill="PyTorch",
        relation_type="deep_learning",
        transferability_score=0.5,
        rationale="Documented deep learning framework transferability: Candidate demonstrates deep learning in PyTorch. Tensor computations, autodiff, and neural architectures transfer to TensorFlow.",
    ),

    # 5. Frontend Component Framework Transferability (0.5 credit)
    ("React", "Vue.js"): SkillRelationshipRule(
        target_skill="React",
        candidate_skill="Vue.js",
        relation_type="frontend_framework",
        transferability_score=0.5,
        rationale="Documented frontend framework transferability: Candidate demonstrates component-based UI in Vue.js. Reactive state management, virtual DOM, and component lifecycles transfer to React.",
    ),
    ("Vue.js", "React"): SkillRelationshipRule(
        target_skill="Vue.js",
        candidate_skill="React",
        relation_type="frontend_framework",
        transferability_score=0.5,
        rationale="Documented frontend framework transferability: Candidate demonstrates component-based UI in React. State management, props, and virtual DOM concepts transfer to Vue.js.",
    ),

    # 6. Contextual Evidence Only (0.0 Substitution Credit)
    # Different paradigms or tools within the same broad ecosystem that do NOT substitute for each other.
    ("Kubernetes", "Docker"): SkillRelationshipRule(
        target_skill="Kubernetes",
        candidate_skill="Docker",
        relation_type="container_ecosystem",
        transferability_score=0.0,
        rationale="Contextual evidence: Candidate demonstrates Docker containerization. While containerization is foundational for container orchestration, Kubernetes requires distinct cluster management, pod networking, ingress, and manifest configuration; no substitution credit awarded.",
    ),
    ("Docker", "Kubernetes"): SkillRelationshipRule(
        target_skill="Docker",
        candidate_skill="Kubernetes",
        relation_type="container_ecosystem",
        transferability_score=0.5,
        rationale="Documented container ecosystem transferability: Candidate demonstrates Kubernetes orchestration, which directly incorporates Docker container concepts; awarded 0.5 partial credit.",
    ),
    ("Redis", "PostgreSQL"): SkillRelationshipRule(
        target_skill="Redis",
        candidate_skill="PostgreSQL",
        relation_type="database_paradigm",
        transferability_score=0.0,
        rationale="Contextual evidence: Candidate demonstrates relational database proficiency in PostgreSQL. Redis operates under an in-memory key-value caching paradigm; no substitution credit awarded.",
    ),
    ("PostgreSQL", "Redis"): SkillRelationshipRule(
        target_skill="PostgreSQL",
        candidate_skill="Redis",
        relation_type="database_paradigm",
        transferability_score=0.0,
        rationale="Contextual evidence: Candidate demonstrates in-memory caching in Redis. PostgreSQL operates under a relational SQL/ACID schema paradigm; no substitution credit awarded.",
    ),
    ("PostgreSQL", "MongoDB"): SkillRelationshipRule(
        target_skill="PostgreSQL",
        candidate_skill="MongoDB",
        relation_type="database_paradigm",
        transferability_score=0.0,
        rationale="Contextual evidence: Candidate demonstrates document-based NoSQL in MongoDB. Relational SQL queries, schema normalization, and ACID transactions represent a distinct database paradigm; no substitution credit awarded.",
    ),
    ("MongoDB", "PostgreSQL"): SkillRelationshipRule(
        target_skill="MongoDB",
        candidate_skill="PostgreSQL",
        relation_type="database_paradigm",
        transferability_score=0.0,
        rationale="Contextual evidence: Candidate demonstrates relational database in PostgreSQL. Document-oriented NoSQL represents a distinct database paradigm; no substitution credit awarded.",
    ),
}


@dataclass
class CandidateExperienceAnalysis:
    """Detailed non-overlapping experience calculation results."""
    professional_years: float
    internship_years: float
    project_count: int
    is_duration_reliable: bool
    roles_count: int
    internships_count: int
    unparseable_roles_count: int
    uncertainty_reasons: List[str]


class JobMatcherService:
    """
    Deterministic matching engine assessing candidate alignment against job requirements.
    """

    def __init__(
        self,
        skill_ontology: Optional[SkillOntology] = None,
        category_weights: Optional[Dict[str, float]] = None,
    ):
        self.skill_ontology = skill_ontology or SkillOntology()
        self.weights = category_weights or dict(DEFAULT_CATEGORY_WEIGHTS)
        logger.info("JobMatcherService initialized with weights: %s", self.weights)

    def match(
        self,
        parsed_job: ParsedJobDescription,
        candidate_profile: CandidateProfile,
        resume_chunks: Optional[List[ResumeChunk]] = None,
    ) -> Dict[str, Any]:
        """
        Execute end-to-end matching evaluation.
        """
        chunks = resume_chunks or []

        # 1. Pre-computation: Gather candidate's evidence indices
        candidate_skills, candidate_aliases, skill_sources, candidate_raw_terms = self._index_candidate_skills(candidate_profile, chunks)
        exp_analysis = self._calculate_candidate_experience(candidate_profile)
        candidate_education = self._index_candidate_education(candidate_profile)

        # 2. Check for sufficient evidence
        has_sufficient_evidence, sufficiency_reason = self._check_sufficiency(
            parsed_job, candidate_profile, candidate_skills
        )

        if not has_sufficient_evidence:
            logger.info("Insufficient evidence detected: %s", sufficiency_reason)
            return {
                "overall_score": 0.0,
                "has_sufficient_evidence": False,
                "category_scores": {},
                "matched_requirements": [],
                "partially_demonstrated_requirements": [],
                "missing_requirements": [],
                "critical_gaps": [],
                "contextual_evidence": [],
                "uncertain_requirements": [sufficiency_reason],
                "recommendations": [],
                "scoring_explanation": f"Insufficient evidence: {sufficiency_reason}",
                "limitations": [
                    "Cannot compute a reliable match score because the candidate profile or job description lacks sufficient extractable information.",
                    sufficiency_reason,
                    "This insufficient-evidence status is distinct from a genuine zero alignment score.",
                ],
            }

        # 3. Match categories
        req_skills_matches = self._match_skills(
            skills=parsed_job.required_skills,
            category_name="required_skills",
            candidate_skills=candidate_skills,
            candidate_aliases=candidate_aliases,
            skill_sources=skill_sources,
            candidate_raw_terms=candidate_raw_terms,
            chunks=chunks,
        )

        pref_skills_matches = self._match_skills(
            skills=parsed_job.preferred_skills,
            category_name="preferred_skills",
            candidate_skills=candidate_skills,
            candidate_aliases=candidate_aliases,
            skill_sources=skill_sources,
            candidate_raw_terms=candidate_raw_terms,
            chunks=chunks,
        )

        exp_matches, exp_uncertainties = self._match_experience(
            required_years=parsed_job.minimum_experience_years,
            exp_analysis=exp_analysis,
            candidate_profile=candidate_profile,
        )

        edu_matches = self._match_education(
            required_quals=parsed_job.required_qualifications,
            candidate_education=candidate_education,
            candidate_profile=candidate_profile,
        )

        # 4. Aggregate outcomes into explicit lists
        all_matches = req_skills_matches + pref_skills_matches + exp_matches + edu_matches

        # Demonstrated: score == 1.0 (direct_match, recognized_alias, or full credit)
        matched_items = [m for m in all_matches if m.score >= 1.0]

        # Partially Demonstrated: 0.0 < score < 1.0 (related skills with credit or partial experience/education)
        partial_items = [m for m in all_matches if 0.0 < m.score < 1.0]

        # Missing Requirements: score == 0.0
        missing_items = [m for m in all_matches if m.score == 0.0]

        # Contextual Evidence: related skills recognized with 0.0 substitution credit
        contextual_evidence = [
            m for m in all_matches
            if m.outcome == RequirementOutcome.RELATED_SKILL and m.score == 0.0
        ]

        # Critical Gaps: Unmet mandatory requirements clearly required in JD (required_skills, experience, education)
        critical_gaps = [
            m for m in all_matches
            if m.category in ("required_skills", "experience", "education") and m.score < 0.5
        ]

        # Uncertain requirements
        uncertain_requirements = list(exp_analysis.uncertainty_reasons) + exp_uncertainties

        # 5. Compute category-level scores
        category_scores, overall_score, scoring_explanation = self._calculate_scores(
            req_skills_matches=req_skills_matches,
            pref_skills_matches=pref_skills_matches,
            exp_matches=exp_matches,
            edu_matches=edu_matches,
        )

        # 6. Generate career gap recommendations
        recommendations = self._generate_recommendations(
            partial_items=partial_items,
            missing_items=missing_items,
            candidate_skills=candidate_skills,
        )

        # 7. Document explicit limitations
        limitations = [
            "Scores represent textual and semantic alignment between the supplied job description and the submitted resume.",
            "This match score reflects resume keyword and structured alignment; it does not predict hiring success or interview outcomes.",
            "Absence of a skill in the resume indicates missing evidence, not necessarily a lack of candidate capability.",
            "Technical skill depth, code quality, and soft skills cannot be fully evaluated from resume text alone.",
        ]
        if not parsed_job.minimum_experience_years:
            limitations.append("The job description did not specify a minimum years of experience requirement; experience was scored neutrally.")
        if not parsed_job.required_qualifications:
            limitations.append("The job description did not specify explicit degree requirements; education was scored neutrally.")
        if critical_gaps:
            limitations.append(f"{len(critical_gaps)} mandatory requirement(s) were not fully demonstrated. Review Critical Gaps independently of the overall score.")

        return {
            "overall_score": overall_score,
            "has_sufficient_evidence": True,
            "category_scores": category_scores,
            "matched_requirements": matched_items,
            "partially_demonstrated_requirements": partial_items,
            "missing_requirements": missing_items,
            "critical_gaps": critical_gaps,
            "contextual_evidence": contextual_evidence,
            "uncertain_requirements": uncertain_requirements,
            "recommendations": recommendations,
            "scoring_explanation": scoring_explanation,
            "limitations": limitations,
        }

    # =========================================================================
    # Candidate Evidence Indexing Helpers
    # =========================================================================

    def _index_candidate_skills(
        self,
        profile: CandidateProfile,
        chunks: List[ResumeChunk],
    ) -> Tuple[Set[str], Dict[str, str], Dict[str, List[str]], Set[str]]:
        """
        Extract and canonicalize all candidate skills from structured data and text sections.
        """
        canonical_skills: Set[str] = set()
        alias_to_canonical: Dict[str, str] = {}
        skill_sources: Dict[str, List[str]] = {}
        candidate_raw_terms: Set[str] = set()

        def record_skill(skill_str: str, source: str, snippet: Optional[str] = None):
            if not skill_str or not skill_str.strip():
                return
            clean_raw = skill_str.lower().strip()
            candidate_raw_terms.add(clean_raw)
            match = self.skill_ontology.normalize_skill(skill_str)
            canonical = match.canonical
            canonical_skills.add(canonical)
            alias_to_canonical[clean_raw] = canonical
            alias_to_canonical[canonical.lower()] = canonical

            if canonical not in skill_sources:
                skill_sources[canonical] = []
            info = f"[{source}] {snippet or skill_str}"
            if info not in skill_sources[canonical]:
                skill_sources[canonical].append(info)

        # 1. From StructuredCV skills list
        if profile.structured_data and profile.structured_data.skills:
            for s in profile.structured_data.skills:
                record_skill(s, "Structured Skills", s)

        # 2. From Experience technologies and descriptions
        if profile.structured_data and profile.structured_data.experience:
            for exp in profile.structured_data.experience:
                role_title = getattr(exp, "title", None) or getattr(exp, "job_title", "Role")
                company = getattr(exp, "company", "")
                role_label = f"{role_title} at {company}" if company else role_title
                techs = getattr(exp, "technologies", [])
                for t in techs:
                    record_skill(t, "Experience Technology", f"{t} used in {role_label}")
                if getattr(exp, "description", None):
                    desc_lower = exp.description.lower()
                    for canon in SkillOntology.TAXONOMY.keys():
                        if re.search(r"\b" + re.escape(canon.lower()) + r"\b", desc_lower):
                            record_skill(canon, "Experience Description", f"{canon} in {role_label}")

        # 3. From Project technologies
        if profile.structured_data and profile.structured_data.projects:
            for p in profile.structured_data.projects:
                proj_label = getattr(p, "name", None) or getattr(p, "title", "Project")
                techs = getattr(p, "technologies", [])
                for t in techs:
                    record_skill(t, "Project Technology", f"{t} applied in {proj_label}")

        # 4. From raw skills text
        if profile.skills:
            raw_tokens = re.split(r"[,;\n•|]+", profile.skills)
            for tok in raw_tokens:
                clean_tok = tok.strip()
                if 2 <= len(clean_tok) <= 40:
                    record_skill(clean_tok, "Skills Section", clean_tok)

        # 5. From resume chunks
        for c in chunks:
            if c.section in ["skills", "experience", "projects"]:
                for canon in SkillOntology.TAXONOMY.keys():
                    if canon.lower() in c.content.lower():
                        record_skill(canon, f"Resume Chunk ({c.section})", c.item_title or c.section)

        return canonical_skills, alias_to_canonical, skill_sources, candidate_raw_terms

    # =========================================================================
    # Experience Calculation & Non-Overlapping Interval Union
    # =========================================================================

    def _calculate_candidate_experience(self, profile: CandidateProfile) -> CandidateExperienceAnalysis:
        """
        Calculate non-overlapping elapsed professional experience.

        Rules:
        - Distinguishes professional employment from internships and projects.
        - Does NOT count overlapping employment periods twice (union of date intervals).
        - Does NOT invent arbitrary durations when dates are missing.
        - Returns explicit reliability and uncertainty state when dates cannot be verified.
        """
        if not profile.structured_data or not profile.structured_data.experience:
            # Check if candidate has projects
            proj_count = len(profile.structured_data.projects) if (profile.structured_data and profile.structured_data.projects) else 0

            # Check if raw text describes experience
            exp_text = profile.experience or ""
            if not exp_text.strip():
                return CandidateExperienceAnalysis(
                    professional_years=0.0,
                    internship_years=0.0,
                    project_count=proj_count,
                    is_duration_reliable=True,
                    roles_count=0,
                    internships_count=0,
                    unparseable_roles_count=0,
                    uncertainty_reasons=[],
                )

            # Parse text if present without inventing arbitrary numbers
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*(?:years?|yrs?)", exp_text, re.IGNORECASE)
            if m:
                years = float(m.group(1))
                return CandidateExperienceAnalysis(
                    professional_years=round(years, 1),
                    internship_years=0.0,
                    project_count=proj_count,
                    is_duration_reliable=True,
                    roles_count=1,
                    internships_count=0,
                    unparseable_roles_count=0,
                    uncertainty_reasons=[],
                )
            else:
                return CandidateExperienceAnalysis(
                    professional_years=0.0,
                    internship_years=0.0,
                    project_count=proj_count,
                    is_duration_reliable=False,
                    roles_count=0,
                    internships_count=0,
                    unparseable_roles_count=0,
                    uncertainty_reasons=["Experience section text is present but contains no verifiable dates or duration."],
                )

        prof_intervals: List[Tuple[int, int]] = []
        intern_intervals: List[Tuple[int, int]] = []

        roles_count = 0
        internships_count = 0
        unparseable_count = 0
        uncertainty_reasons: List[str] = []

        now = datetime.now()
        current_month_idx = now.year * 12 + now.month

        for exp in profile.structured_data.experience:
            title = (getattr(exp, "title", None) or getattr(exp, "job_title", "")).lower()
            desc = (getattr(exp, "description", None) or "").lower()
            company = getattr(exp, "company", "Company")

            is_internship = bool(re.search(r"\b(intern|internship|trainee|apprentice|co-op)\b", f"{title} {desc}"))
            if is_internship:
                internships_count += 1
            else:
                roles_count += 1

            # Parse start date
            s_idx = self._extract_date_month_index(getattr(exp, "start_date", None), is_start=True)
            if s_idx is None:
                unparseable_count += 1
                role_label = getattr(exp, "title", None) or getattr(exp, "job_title", "Role")
                uncertainty_reasons.append(f"Employment dates could not be verified for role: {role_label} at {company}.")
                continue

            # Parse end date
            is_curr = getattr(exp, "is_current", False)
            raw_end = getattr(exp, "end_date", None)
            if is_curr or (raw_end and raw_end.strip().lower() in ["present", "current", "now", "ongoing"]):
                e_idx = current_month_idx
            else:
                e_idx = self._extract_date_month_index(raw_end, is_start=False)
                if e_idx is None:
                    # If end date is missing for non-current role, assume month after start or current
                    e_idx = s_idx + 1

            if e_idx < s_idx:
                e_idx = s_idx + 1  # Guard against inverted inputs

            interval = (s_idx, e_idx)
            if is_internship:
                intern_intervals.append(interval)
            else:
                prof_intervals.append(interval)

        # Merge overlapping intervals (Non-overlapping union)
        merged_prof = self._merge_intervals(prof_intervals)
        merged_intern = self._merge_intervals(intern_intervals)

        total_prof_months = sum((end - start + 1) for start, end in merged_prof)
        total_intern_months = sum((end - start + 1) for start, end in merged_intern)

        prof_years = round(total_prof_months / 12.0, 1)
        intern_years = round(total_intern_months / 12.0, 1)

        total_analyzed = roles_count + internships_count
        is_reliable = True
        if total_analyzed > 0 and unparseable_count == total_analyzed:
            is_reliable = False
            uncertainty_reasons.append("All documented roles in resume lack parseable employment dates.")

        proj_count = len(profile.structured_data.projects) if (profile.structured_data and profile.structured_data.projects) else 0

        return CandidateExperienceAnalysis(
            professional_years=prof_years,
            internship_years=intern_years,
            project_count=proj_count,
            is_duration_reliable=is_reliable,
            roles_count=roles_count,
            internships_count=internships_count,
            unparseable_roles_count=unparseable_count,
            uncertainty_reasons=uncertainty_reasons,
        )

    def _extract_date_month_index(self, d_str: Optional[str], is_start: bool = True) -> Optional[int]:
        """Convert varied resume date formats into an integer month index (year * 12 + month)."""
        if not d_str or not d_str.strip():
            return None
        d = d_str.strip().lower()
        if d in ["present", "current", "now"]:
            now = datetime.now()
            return now.year * 12 + now.month

        # 1. YYYY-MM
        m1 = re.search(r"\b(20\d\d|19\d\d)[-/](0?[1-9]|1[0-2])\b", d)
        if m1:
            return int(m1.group(1)) * 12 + int(m1.group(2))

        # 2. MM/YYYY
        m2 = re.search(r"\b(0?[1-9]|1[0-2])[-/](20\d\d|19\d\d)\b", d)
        if m2:
            return int(m2.group(2)) * 12 + int(m2.group(1))

        # 3. Month Name + Year (e.g. May 2021)
        month_names = {
            "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
            "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12
        }
        for name, m_int in month_names.items():
            if name in d:
                y_match = re.search(r"\b(20\d\d|19\d\d)\b", d)
                if y_match:
                    return int(y_match.group(1)) * 12 + m_int

        # 4. Year only (e.g. 2021)
        m4 = re.search(r"\b(20\d\d|19\d\d)\b", d)
        if m4:
            year = int(m4.group(1))
            return year * 12 + (1 if is_start else 12)

        return None

    def _merge_intervals(self, intervals: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Compute the union of date intervals to avoid counting overlapping periods twice."""
        if not intervals:
            return []
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged: List[List[int]] = []

        for start, end in sorted_intervals:
            if not merged:
                merged.append([start, end])
            else:
                prev_start, prev_end = merged[-1]
                if start <= prev_end:  # Overlapping or adjacent
                    merged[-1][1] = max(prev_end, end)
                else:
                    merged.append([start, end])

        return [(start, end) for start, end in merged]

    def _index_candidate_education(self, profile: CandidateProfile) -> List[Dict[str, str]]:
        """Index candidate's educational degrees and fields of study."""
        quals = []
        if profile.structured_data and profile.structured_data.education:
            for edu in profile.structured_data.education:
                deg = getattr(edu, "degree", "")
                field = getattr(edu, "field", None) or getattr(edu, "field_of_study", "")
                inst = getattr(edu, "institution", "")
                summary = f"{deg} in {field} from {inst}".strip() if (deg or field or inst) else ""
                quals.append({
                    "degree": deg,
                    "field": field,
                    "institution": inst,
                    "summary": summary,
                })
        elif profile.education:
            quals.append({"degree": "", "field": "", "institution": "", "summary": profile.education[:200]})
        return quals

    def _check_sufficiency(
        self,
        job: ParsedJobDescription,
        profile: CandidateProfile,
        candidate_skills: Set[str],
    ) -> Tuple[bool, str]:
        """Check whether minimum evidence exists to calculate an explainable match."""
        total_job_reqs = (
            len(job.required_skills)
            + len(job.preferred_skills)
            + len(job.required_qualifications)
            + (1 if job.minimum_experience_years else 0)
        )
        if total_job_reqs == 0:
            return False, "The job description does not contain any extractable skills, experience, or qualifications."

        has_text = bool(profile.summary or profile.experience or profile.skills or profile.education)
        has_skills = len(candidate_skills) > 0
        has_roles = bool(profile.structured_data and profile.structured_data.experience)

        if not has_text and not has_skills and not has_roles:
            return False, "The candidate resume appears empty or contains no extractable skills, experience, or education."

        return True, ""

    # =========================================================================
    # Matching Engines
    # =========================================================================

    def _match_skills(
        self,
        skills: List[str],
        category_name: str,
        candidate_skills: Set[str],
        candidate_aliases: Dict[str, str],
        skill_sources: Dict[str, List[str]],
        candidate_raw_terms: Set[str],
        chunks: List[ResumeChunk],
    ) -> List[RequirementMatch]:
        """
        Match a list of skills against candidate profile evidence using documented rules:
        - direct_match: Exact verbatim or canonical string match (score 1.0).
        - recognized_alias: Documented taxonomy alias match (score 1.0).
        - related_skill: Documented relationship rule (score 0.5 for transferability, 0.0 for contextual evidence).
        - not_found_in_resume: No documented relationship or mention (score 0.0).
        """
        matches: List[RequirementMatch] = []

        for skill in skills:
            norm = self.skill_ontology.normalize_skill(skill)
            canonical = norm.canonical
            skill_lower = skill.lower().strip()
            canon_lower = canonical.lower()

            is_direct = (
                (skill_lower in candidate_raw_terms)
                or (canonical.lower() in candidate_raw_terms and skill_lower == canon_lower)
            )

            # 1. Direct match (exact verbatim or canonical name explicitly listed)
            if is_direct and canonical in candidate_skills:
                sources = skill_sources.get(canonical, [])
                evidence = sources[0] if sources else f"Listed in candidate resume skills: {canonical}"
                matches.append(
                    RequirementMatch(
                        requirement=skill,
                        category=category_name,
                        outcome=RequirementOutcome.DIRECT_MATCH,
                        score=1.0,
                        resume_evidence=evidence,
                        source_section="skills",
                        explanation=f"Direct match: Candidate demonstrates verified proficiency in {canonical}.",
                    )
                )
                continue

            # 2. Recognized alias match (e.g. 'postgres' for 'PostgreSQL', 'py' for 'Python')
            if canonical in candidate_skills or skill_lower in candidate_aliases or canon_lower in candidate_aliases:
                sources = skill_sources.get(canonical, [])
                evidence = sources[0] if sources else f"Recognized alias listed in candidate resume: {skill}"
                matches.append(
                    RequirementMatch(
                        requirement=skill,
                        category=category_name,
                        outcome=RequirementOutcome.RECOGNIZED_ALIAS,
                        score=1.0,
                        resume_evidence=evidence,
                        source_section="skills",
                        explanation=f"Recognized alias: '{skill}' resolves to canonical '{canonical}' demonstrated on resume.",
                    )
                )
                continue

            # 3. Documented related skill rule check
            rule_match: Optional[SkillRelationshipRule] = None
            for c_skill in candidate_skills:
                key = (canonical, c_skill)
                if key in DOCUMENTED_SKILL_RELATIONSHIPS:
                    rule_match = DOCUMENTED_SKILL_RELATIONSHIPS[key]
                    break

            if rule_match:
                sources = skill_sources.get(rule_match.candidate_skill, [])
                c_evidence = sources[0] if sources else f"Candidate demonstrates {rule_match.candidate_skill}"
                matches.append(
                    RequirementMatch(
                        requirement=skill,
                        category=category_name,
                        outcome=RequirementOutcome.RELATED_SKILL,
                        score=rule_match.transferability_score,
                        resume_evidence=f"{c_evidence} (Related Technology)",
                        source_section="skills",
                        explanation=rule_match.rationale,
                    )
                )
                continue

            # 4. Not found in resume
            matches.append(
                RequirementMatch(
                    requirement=skill,
                    category=category_name,
                    outcome=RequirementOutcome.NOT_FOUND_IN_RESUME,
                    score=0.0,
                    resume_evidence=None,
                    source_section=None,
                    explanation=(
                        f"Not found in resume: {skill} is not documented in the candidate's profile. "
                        f"This indicates absent resume evidence, not proof of incapacity."
                    ),
                )
            )

        return matches

    def _match_experience(
        self,
        required_years: Optional[float],
        exp_analysis: CandidateExperienceAnalysis,
        candidate_profile: CandidateProfile,
    ) -> Tuple[List[RequirementMatch], List[str]]:
        """
        Evaluate minimum years of experience requirement against non-overlapping professional tenure.
        """
        if required_years is None:
            return [], []

        req_str = f"{required_years:g}+ years of professional experience"
        uncertainties: List[str] = []

        # 1. Handle unverified / missing dates
        if not exp_analysis.is_duration_reliable:
            uncertainty_msg = (
                f"Candidate lists {exp_analysis.roles_count} role(s) on resume, but employment dates "
                f"could not be verified from the text. Required tenure of {required_years:g} years cannot be confirmed."
            )
            uncertainties.append(uncertainty_msg)
            return [
                RequirementMatch(
                    requirement=req_str,
                    category="experience",
                    outcome=RequirementOutcome.NOT_FOUND_IN_RESUME,
                    score=0.0,
                    resume_evidence="Documented roles lack verified start/end dates.",
                    source_section="experience",
                    explanation=f"Experience duration uncertain: {uncertainty_msg}",
                )
            ], uncertainties

        # 2. Check professional experience meets requirement
        prof_years = exp_analysis.professional_years
        if prof_years >= required_years:
            return [
                RequirementMatch(
                    requirement=req_str,
                    category="experience",
                    outcome=RequirementOutcome.DEMONSTRATED,
                    score=1.0,
                    resume_evidence=f"Candidate has {prof_years:.1f} years of verified non-overlapping professional experience across {exp_analysis.roles_count} role(s).",
                    source_section="experience",
                    explanation=f"Demonstrated: Candidate meets professional experience requirement ({prof_years:.1f} yrs >= {required_years:g} yrs across non-overlapping employment).",
                )
            ], []

        # 3. Check partial professional experience (>= 50% threshold)
        if prof_years >= (0.5 * required_years):
            intern_note = (
                f" Candidate also documents {exp_analysis.internship_years:.1f} years of internship experience."
                if exp_analysis.internship_years > 0 else ""
            )
            return [
                RequirementMatch(
                    requirement=req_str,
                    category="experience",
                    outcome=RequirementOutcome.PARTIALLY_DEMONSTRATED,
                    score=0.5,
                    resume_evidence=f"Candidate documents {prof_years:.1f} years of professional experience.{intern_note}",
                    source_section="experience",
                    explanation=f"Partially demonstrated: Candidate documents {prof_years:.1f} years of verified professional experience against {required_years:g} required years.{intern_note}",
                )
            ], []

        # 4. Below 50% or 0 professional years
        if prof_years == 0.0 and (exp_analysis.internship_years > 0 or exp_analysis.project_count > 0):
            return [
                RequirementMatch(
                    requirement=req_str,
                    category="experience",
                    outcome=RequirementOutcome.NOT_FOUND_IN_RESUME,
                    score=0.0,
                    resume_evidence=(
                        f"Candidate documents {exp_analysis.internship_years:.1f} years of internship experience "
                        f"and {exp_analysis.project_count} project(s), with 0 years of full-time professional employment."
                    ),
                    source_section="experience",
                    explanation=(
                        f"Entry-level profile: Role requires {required_years:g} years of professional experience. "
                        f"Internships and academic projects demonstrate technical capability but do not fulfill "
                        f"full-time industry tenure requirements."
                    ),
                )
            ], []

        return [
            RequirementMatch(
                requirement=req_str,
                category="experience",
                outcome=RequirementOutcome.NOT_FOUND_IN_RESUME,
                score=0.0,
                resume_evidence=f"Candidate has {prof_years:.1f} years of documented professional experience.",
                source_section="experience",
                explanation=f"Experience gap: Required {required_years:g} years, but resume documents {prof_years:.1f} years of verified experience.",
            )
        ], []

    def _match_education(
        self,
        required_quals: List[str],
        candidate_education: List[Dict[str, str]],
        candidate_profile: CandidateProfile,
    ) -> List[RequirementMatch]:
        """Match degree requirements against candidate education."""
        if not required_quals:
            return []

        matches: List[RequirementMatch] = []
        for qual in required_quals:
            qual_lower = qual.lower()

            matched = False
            for edu in candidate_education:
                deg_lower = edu.get("degree", "").lower()
                summary_lower = edu.get("summary", "").lower()

                # Check degree match (e.g. bachelor, master, b.tech, bs)
                deg_matches = (
                    ("bachelor" in qual_lower and ("bachelor" in deg_lower or "b.tech" in deg_lower or "bs" in deg_lower or "b.e" in deg_lower or "bachelor" in summary_lower))
                    or ("master" in qual_lower and ("master" in deg_lower or "m.tech" in deg_lower or "ms" in deg_lower or "master" in summary_lower))
                    or ("phd" in qual_lower and ("phd" in deg_lower or "doctor" in deg_lower))
                )

                if deg_matches:
                    matches.append(
                        RequirementMatch(
                            requirement=qual,
                            category="education",
                            outcome=RequirementOutcome.DEMONSTRATED,
                            score=1.0,
                            resume_evidence=edu.get("summary") or "Documented university degree",
                            source_section="education",
                            explanation=f"Demonstrated: Candidate holds qualifying degree ({edu.get('summary')}).",
                        )
                    )
                    matched = True
                    break

            if not matched:
                if candidate_education:
                    matches.append(
                        RequirementMatch(
                            requirement=qual,
                            category="education",
                            outcome=RequirementOutcome.PARTIALLY_DEMONSTRATED,
                            score=0.5,
                            resume_evidence=candidate_education[0].get("summary", ""),
                            source_section="education",
                            explanation=f"Partially demonstrated: Candidate documents higher education, but exact match for '{qual}' requires review.",
                        )
                    )
                else:
                    matches.append(
                        RequirementMatch(
                            requirement=qual,
                            category="education",
                            outcome=RequirementOutcome.NOT_FOUND_IN_RESUME,
                            score=0.0,
                            resume_evidence=None,
                            source_section=None,
                            explanation=f"Not found in resume: Degree requirement '{qual}' is not evidenced in the education section.",
                        )
                    )

        return matches

    # =========================================================================
    # Scoring & Aggregation
    # =========================================================================

    def _calculate_scores(
        self,
        req_skills_matches: List[RequirementMatch],
        pref_skills_matches: List[RequirementMatch],
        exp_matches: List[RequirementMatch],
        edu_matches: List[RequirementMatch],
    ) -> Tuple[Dict[str, CategoryScore], float, str]:
        """
        Calculate category scores and weighted overall score.
        Dynamically redistributes weights if a category has no items.
        """
        category_data = [
            ("required_skills", req_skills_matches, self.weights["required_skills"]),
            ("experience", exp_matches, self.weights["experience"]),
            ("education", edu_matches, self.weights["education"]),
            ("preferred_skills", pref_skills_matches, self.weights["preferred_skills"]),
        ]

        active_categories: List[Tuple[str, List[RequirementMatch], float]] = []
        for name, items, w in category_data:
            if items:
                active_categories.append((name, items, w))

        total_active_weight = sum(w for _, _, w in active_categories) or 1.0

        category_scores: Dict[str, CategoryScore] = {}
        total_weighted_score = 0.0

        for name, items, base_weight in active_categories:
            norm_weight = base_weight / total_active_weight

            total_items = len(items)
            demonstrated = sum(1 for m in items if m.score >= 1.0)
            partial = sum(1 for m in items if 0.0 < m.score < 1.0)
            missing = sum(1 for m in items if m.score == 0.0)

            item_sum = sum(m.score for m in items)
            raw_score = (item_sum / total_items) * 100.0 if total_items > 0 else 100.0
            weighted_score = raw_score * norm_weight
            total_weighted_score += weighted_score

            category_scores[name] = CategoryScore(
                category=name,
                score=round(raw_score, 1),
                weight=round(norm_weight, 3),
                weighted_score=round(weighted_score, 1),
                total_items=total_items,
                demonstrated_items=demonstrated,
                partially_demonstrated_items=partial,
                missing_items=missing,
                explanation=f"{demonstrated}/{total_items} demonstrated, {partial} partial, {missing} missing.",
            )

        overall_score = round(max(0.0, min(100.0, total_weighted_score)), 1)

        explanation = (
            f"Overall score {overall_score}/100 calculated from {len(active_categories)} active categories: "
            + ", ".join([f"{name} ({cs.score:.1f}% @ {cs.weight * 100:.0f}% weight)" for name, cs in category_scores.items()])
            + ". Mandatory requirements carry highest impact."
        )

        return category_scores, overall_score, explanation

    # =========================================================================
    # Career Gap Recommendations
    # =========================================================================

    def _generate_recommendations(
        self,
        partial_items: List[RequirementMatch],
        missing_items: List[RequirementMatch],
        candidate_skills: Set[str],
    ) -> List[CareerGapRecommendation]:
        """
        Generate grounded, ethical career gap recommendations:
        - Never fabricates achievements, projects, or metrics.
        - Avoids claiming courses guarantee employment.
        - Distinguishes missing resume evidence from verified skill gaps.
        """
        recommendations: List[CareerGapRecommendation] = []

        # Process partially demonstrated items first (e.g. transferable skill or partial tenure)
        for item in partial_items:
            req = item.requirement
            if item.category in ["required_skills", "preferred_skills"]:
                recommendations.append(
                    CareerGapRecommendation(
                        requirement=req,
                        gap_type="missing_evidence_in_resume",
                        evidence_gap=(
                            f"You demonstrated related background ({item.resume_evidence}), "
                            f"but {req} is not explicitly stated in your work experience or skills."
                        ),
                        suggested_action=(
                            f"If you have applied {req} in your projects or coursework, add an explicit mention "
                            f"highlighting how you used it alongside your existing skills."
                        ),
                        action_type="resume_clarification",
                    )
                )
            elif item.category == "experience":
                recommendations.append(
                    CareerGapRecommendation(
                        requirement=req,
                        gap_type="missing_evidence_in_resume",
                        evidence_gap=item.explanation,
                        suggested_action=(
                            "Ensure all professional roles and internships are clearly listed with start and end dates "
                            "to reflect your complete experience duration."
                        ),
                        action_type="resume_clarification",
                    )
                )
            elif item.category == "education":
                recommendations.append(
                    CareerGapRecommendation(
                        requirement=req,
                        gap_type="missing_evidence_in_resume",
                        evidence_gap=item.explanation,
                        suggested_action=(
                            f"Ensure your Education section explicitly lists the degree title, major, "
                            f"and institution corresponding to '{req}' for clear alignment."
                        ),
                        action_type="resume_clarification",
                    )
                )

        # Process missing items (score == 0.0)
        for item in missing_items:
            req = item.requirement
            if item.category in ["required_skills", "preferred_skills"]:
                if item.outcome == RequirementOutcome.RELATED_SKILL:
                    # Contextual evidence observed but no substitution credit
                    recommendations.append(
                        CareerGapRecommendation(
                            requirement=req,
                            gap_type="skill_gap",
                            evidence_gap=item.explanation,
                            suggested_action=(
                                f"To build verified competency in {req}, consider building a targeted hands-on project "
                                f"integrating {req} to demonstrate practical capabilities. "
                                f"Note: Completing tutorials or certificates alone does not guarantee employment."
                            ),
                            action_type="project_building",
                        )
                    )
                else:
                    recommendations.append(
                        CareerGapRecommendation(
                            requirement=req,
                            gap_type="skill_gap",
                            evidence_gap=f"No evidence of {req} was identified in your resume.",
                            suggested_action=(
                                f"To build verified competency in {req}, consider building a targeted hands-on project "
                                f"integrating {req} or studying official documentation and community standards. "
                                f"Note: Completing tutorials or certificates alone does not guarantee employment."
                            ),
                            action_type="learning_and_practice",
                        )
                    )
            elif item.category == "experience":
                recommendations.append(
                    CareerGapRecommendation(
                        requirement=req,
                        gap_type="missing_evidence_in_resume",
                        evidence_gap=item.explanation,
                        suggested_action=(
                            "Ensure all professional roles, contracts, and internships include verified employment dates. "
                            "For roles requiring industry tenure, highlight senior project leadership and technical ownership."
                        ),
                        action_type="resume_clarification",
                    )
                )
            elif item.category == "education":
                recommendations.append(
                    CareerGapRecommendation(
                        requirement=req,
                        gap_type="missing_evidence_in_resume",
                        evidence_gap=f"Degree requirement '{req}' is not evidenced on your resume.",
                        suggested_action=(
                            "If you hold this degree or an equivalent credential, ensure your Education section "
                            "explicitly states your degree title, major, and graduation year."
                        ),
                        action_type="resume_clarification",
                    )
                )

        return recommendations


_GLOBAL_JOB_MATCHER: Optional[JobMatcherService] = None


def get_job_matcher() -> JobMatcherService:
    """Singleton provider for JobMatcherService."""
    global _GLOBAL_JOB_MATCHER
    if _GLOBAL_JOB_MATCHER is None:
        _GLOBAL_JOB_MATCHER = JobMatcherService()
    return _GLOBAL_JOB_MATCHER


def reset_job_matcher() -> None:
    """Reset singleton instance (useful for testing)."""
    global _GLOBAL_JOB_MATCHER
    _GLOBAL_JOB_MATCHER = None
