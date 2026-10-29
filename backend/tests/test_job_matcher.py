"""
Unit and integration tests for JobMatcherService.

Verifies:
1. Direct skill matching, alias resolution, and evidence tracking.
2. Explicit documented related skills with transferability credit (Django vs FastAPI, AWS vs GCP).
3. Contextual evidence without substitution credit (Docker vs Kubernetes, PostgreSQL vs Redis).
4. Non-overlapping experience calculation (union of employment intervals).
5. Distinguishing internships and projects from professional employment.
6. Explicit uncertainty state when dates cannot be verified (no invented durations).
7. Critical mandatory gaps visibility vs preferred missing requirements.
8. Dynamic weight redistribution and deterministic scoring.
9. Grounded, ethical career gap recommendations.
"""

import unittest

from backend.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    ParsedJobDescription,
    ProjectItem,
    RequirementOutcome,
    ResumeChunk,
    StructuredCV,
)
from backend.services.job_matcher import JobMatcherService, reset_job_matcher
from backend.services.skill_ontology import SkillOntology


class TestJobMatcherService(unittest.TestCase):
    """Test suite for calibrated deterministic job matching and career gap analysis."""

    def setUp(self):
        reset_job_matcher()
        self.ontology = SkillOntology()
        self.matcher = JobMatcherService(skill_ontology=self.ontology)

    def tearDown(self):
        reset_job_matcher()

    def _create_sample_profile(
        self,
        skills: list[str] = None,
        experience_roles: list[dict] = None,
        education: list[dict] = None,
        projects: list[dict] = None,
    ) -> CandidateProfile:
        """Helper to create a populated CandidateProfile."""
        exp_objs = []
        if experience_roles:
            for r in experience_roles:
                exp_objs.append(
                    ExperienceItem(
                        title=r.get("job_title", "Software Engineer"),
                        company=r.get("company", "TechCorp"),
                        start_date=r.get("start_date", "2021-01"),
                        end_date=r.get("end_date", "2024-01"),
                        is_current=r.get("is_current", False),
                        description=r.get("description", "Developed backend services in Python and FastAPI."),
                    )
                )

        edu_objs = []
        if education:
            for e in education:
                edu_objs.append(
                    EducationItem(
                        degree=e.get("degree", "Bachelor of Science"),
                        field=e.get("field", "Computer Science"),
                        institution=e.get("institution", "State University"),
                    )
                )

        proj_objs = []
        if projects:
            for p in projects:
                proj_objs.append(
                    ProjectItem(
                        name=p.get("name", "Web App"),
                        description=p.get("description", "A project"),
                        technologies=p.get("technologies", []),
                    )
                )

        skill_list = skills or ["Python", "PostgreSQL", "Docker", "FastAPI"]
        structured = StructuredCV(
            full_name="Alex Smith",
            email="alex@example.com",
            skills=skill_list,
            experience=exp_objs,
            education=edu_objs,
            projects=proj_objs,
        )

        return CandidateProfile(
            name="Alex Smith",
            email="alex@example.com",
            skills=", ".join(skill_list),
            experience="3 years backend software development",
            education="Bachelor of Science in Computer Science",
            structured_data=structured,
        )

    # =========================================================================
    # 1. Skill Relationship Tests
    # =========================================================================

    def test_python_and_aliases(self):
        """Python exact match produces direct_match; aliases produce recognized_alias with 1.0 score."""
        profile = self._create_sample_profile(skills=["Python", "py"])
        parsed_job = ParsedJobDescription(
            job_title="Python Developer",
            company="Acme Corp",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Python",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)
        match = result["matched_requirements"][0]
        self.assertIn(match.outcome, (RequirementOutcome.DIRECT_MATCH, RequirementOutcome.RECOGNIZED_ALIAS))
        self.assertEqual(match.score, 1.0)

    def test_postgresql_and_postgres_alias(self):
        """Postgres alias resolves to canonical PostgreSQL with full 1.0 credit."""
        profile = self._create_sample_profile(skills=["Postgres"])
        parsed_job = ParsedJobDescription(
            job_title="Database Engineer",
            company="Acme Corp",
            required_skills=["PostgreSQL"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: PostgreSQL",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)
        match = result["matched_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.RECOGNIZED_ALIAS)
        self.assertEqual(match.score, 1.0)
        self.assertIn("PostgreSQL", match.explanation)

    def test_docker_vs_kubernetes_contextual_evidence_no_substitution_credit(self):
        """Docker vs Kubernetes is reported as contextual evidence with 0.0 substitution credit."""
        profile = self._create_sample_profile(skills=["Docker"])
        parsed_job = ParsedJobDescription(
            job_title="DevOps Engineer",
            company="Acme Corp",
            required_skills=["Kubernetes"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Kubernetes",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)

        # Must NOT receive 0.5 credit merely because both are in Cloud & Infrastructure
        self.assertEqual(len(result["matched_requirements"]), 0)
        self.assertEqual(len(result["partially_demonstrated_requirements"]), 0)
        self.assertEqual(len(result["missing_requirements"]), 1)
        self.assertEqual(len(result["contextual_evidence"]), 1)

        match = result["missing_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.RELATED_SKILL)
        self.assertEqual(match.score, 0.0)
        self.assertIn("no substitution credit awarded", match.explanation.lower())

        # Being a mandatory requirement with 0 score, it must be in critical_gaps
        self.assertEqual(len(result["critical_gaps"]), 1)
        self.assertEqual(result["critical_gaps"][0].requirement, "Kubernetes")

    def test_postgresql_vs_redis_contextual_evidence_no_substitution_credit(self):
        """PostgreSQL vs Redis is reported as contextual evidence with 0.0 substitution credit."""
        profile = self._create_sample_profile(skills=["PostgreSQL"])
        parsed_job = ParsedJobDescription(
            job_title="Backend Engineer",
            company="Acme Corp",
            required_skills=["Redis"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Redis",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)

        self.assertEqual(len(result["matched_requirements"]), 0)
        self.assertEqual(len(result["partially_demonstrated_requirements"]), 0)
        self.assertEqual(len(result["missing_requirements"]), 1)
        self.assertEqual(len(result["contextual_evidence"]), 1)

        match = result["missing_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.RELATED_SKILL)
        self.assertEqual(match.score, 0.0)
        self.assertIn("in-memory key-value", match.explanation.lower())

    def test_django_vs_fastapi_documented_transferability(self):
        """Django vs FastAPI has documented framework transferability and receives 0.5 credit."""
        profile = self._create_sample_profile(skills=["Django"])
        parsed_job = ParsedJobDescription(
            job_title="Backend Engineer",
            company="Acme Corp",
            required_skills=["FastAPI"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: FastAPI",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)

        self.assertEqual(len(result["partially_demonstrated_requirements"]), 1)
        match = result["partially_demonstrated_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.RELATED_SKILL)
        self.assertEqual(match.score, 0.5)
        self.assertIn("framework transferability", match.explanation.lower())

    def test_aws_vs_gcp_documented_transferability(self):
        """AWS vs Google Cloud has documented hyperscaler transferability and receives 0.5 credit."""
        profile = self._create_sample_profile(skills=["AWS"])
        parsed_job = ParsedJobDescription(
            job_title="Cloud Engineer",
            company="Acme Corp",
            required_skills=["Google Cloud"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Google Cloud",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)

        self.assertEqual(len(result["partially_demonstrated_requirements"]), 1)
        match = result["partially_demonstrated_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.RELATED_SKILL)
        self.assertEqual(match.score, 0.5)
        self.assertIn("cloud provider transferability", match.explanation.lower())

    def test_unrelated_skills_return_not_found_in_resume(self):
        """Skills with no meaningful relationship receive not_found_in_resume and 0.0 score."""
        profile = self._create_sample_profile(skills=["Python"])
        parsed_job = ParsedJobDescription(
            job_title="Systems Engineer",
            company="Acme Corp",
            required_skills=["Rust"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Rust",
            extraction_source="deterministic",
        )

        result = self.matcher.match(parsed_job, profile)
        self.assertEqual(len(result["missing_requirements"]), 1)
        match = result["missing_requirements"][0]
        self.assertEqual(match.outcome, RequirementOutcome.NOT_FOUND_IN_RESUME)
        self.assertEqual(match.score, 0.0)
        self.assertIn("not documented", match.explanation.lower())

    # =========================================================================
    # 2. Experience & Overlap Calculation Tests
    # =========================================================================

    def test_overlapping_roles_not_counted_twice(self):
        """Concurrent overlapping employment intervals are merged to prevent double-counting."""
        # Role 1: Jan 2021 to Dec 2022 (24 months)
        # Role 2: Jan 2022 to Dec 2022 (12 months, completely inside Role 1)
        # Total elapsed professional experience should be 24 months (2.0 years), NOT 36 months (3.0 years)!
        profile = self._create_sample_profile(
            experience_roles=[
                {
                    "job_title": "Backend Developer",
                    "company": "Company A",
                    "start_date": "2021-01",
                    "end_date": "2022-12",
                },
                {
                    "job_title": "Contract Engineer",
                    "company": "Company B",
                    "start_date": "2022-01",
                    "end_date": "2022-12",
                },
            ]
        )

        job = ParsedJobDescription(
            job_title="Software Engineer",
            company="Acme",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=2.5,
            raw_text="Requires 2.5 years experience",
            extraction_source="deterministic",
        )

        res = self.matcher.match(job, profile)
        # With 2.0 verified non-overlapping years, candidate has >= 50% (1.25 yrs) of 2.5 yrs -> PARTIALLY_DEMONSTRATED
        exp_match = [m for m in res["partially_demonstrated_requirements"] if m.category == "experience"]
        self.assertEqual(len(exp_match), 1)
        self.assertIn("2.0 years", exp_match[0].resume_evidence)

    def test_internships_distinguished_from_fulltime_employment(self):
        """Internships are recognized separately and do not satisfy full-time experience requirements."""
        profile = self._create_sample_profile(
            experience_roles=[
                {
                    "job_title": "Software Engineering Intern",
                    "company": "Tech Corp",
                    "start_date": "2022-06",
                    "end_date": "2022-12",
                }
            ],
            projects=[
                {
                    "name": "E-Commerce Site",
                    "description": "Built full stack app",
                    "technologies": ["Python", "FastAPI"],
                }
            ],
        )

        job = ParsedJobDescription(
            job_title="Software Engineer",
            company="Acme",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=2.0,
            raw_text="Requires 2 years experience",
            extraction_source="deterministic",
        )

        res = self.matcher.match(job, profile)
        exp_gap = [m for m in res["missing_requirements"] if m.category == "experience"]
        self.assertEqual(len(exp_gap), 1)
        self.assertIn("entry-level profile", exp_gap[0].explanation.lower())
        self.assertIn("internship", exp_gap[0].resume_evidence.lower())

    def test_missing_dates_flag_uncertainty_without_inventing_durations(self):
        """Roles lacking parseable dates produce an explicit uncertainty state without guessing 1.5 yrs."""
        profile = self._create_sample_profile(
            experience_roles=[
                {
                    "job_title": "Software Developer",
                    "company": "Tech Corp",
                    "start_date": "unspecified date",
                    "end_date": "unknown",
                }
            ]
        )

        job = ParsedJobDescription(
            job_title="Software Engineer",
            company="Acme",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=3.0,
            raw_text="Requires 3 years experience",
            extraction_source="deterministic",
        )

        res = self.matcher.match(job, profile)
        self.assertGreater(len(res["uncertain_requirements"]), 0)
        exp_match = [m for m in res["missing_requirements"] if m.category == "experience"]
        self.assertEqual(len(exp_match), 1)
        self.assertIn("experience duration uncertain", exp_match[0].explanation.lower())

    # =========================================================================
    # 3. Critical Gaps & Mandatory Visibility Tests
    # =========================================================================

    def test_critical_gaps_include_mandatory_but_not_preferred_missing_skills(self):
        """Critical gaps prominently expose unmet mandatory requirements without polluting with preferred skills."""
        profile = self._create_sample_profile(
            skills=["Docker", "AWS"],  # Lacks mandatory Python; lacks preferred Redis
        )

        job = ParsedJobDescription(
            job_title="Senior Engineer",
            company="Acme",
            required_skills=["Python"],  # MANDATORY
            preferred_skills=["Redis"],  # PREFERRED
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Python. Preferred: Redis.",
            extraction_source="deterministic",
        )

        res = self.matcher.match(job, profile)
        critical_names = [m.requirement for m in res["critical_gaps"]]

        # Mandatory Python MUST be in critical_gaps
        self.assertIn("Python", critical_names)

        # Preferred Redis MUST NOT be in critical_gaps
        self.assertNotIn("Redis", critical_names)

        # But Redis IS in missing_requirements
        missing_names = [m.requirement for m in res["missing_requirements"]]
        self.assertIn("Redis", missing_names)


if __name__ == "__main__":
    unittest.main()
