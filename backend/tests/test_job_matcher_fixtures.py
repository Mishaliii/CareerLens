"""
Realistic deterministic validation fixtures for CareerLens Job Intelligence & Matching Calibration.

Covers 10 distinct, grounded evaluation scenarios:
1. Strong technical match with direct evidence.
2. Candidate with related but non-equivalent technologies (contextual evidence).
3. Fresher applying to a role requesting professional experience.
4. Candidate with internship and project experience but limited employment history.
5. Resume with missing or unparseable employment dates.
6. Job description with no clearly stated mandatory skills.
7. Job description with ambiguous or non-technical requirements.
8. Candidate matching many preferred skills but lacking an explicitly mandatory requirement.
9. Resume omitting evidence without proving lack of candidate capacity.
10. Job description with unconventional section headers and irregular bullet formatting.
"""

import unittest

from backend.models.schemas import (
    CandidateProfile,
    EducationItem,
    ExperienceItem,
    ParsedJobDescription,
    ProjectItem,
    RequirementOutcome,
    StructuredCV,
)
from backend.services.job_matcher import JobMatcherService, reset_job_matcher
from backend.services.job_parser import JobParserService, reset_job_parser
from backend.services.skill_ontology import SkillOntology


class TestJobMatcherRealisticFixtures(unittest.TestCase):
    """Validation test suite executing 10 deterministic real-world job matching scenarios."""

    def setUp(self):
        reset_job_matcher()
        reset_job_parser()
        self.ontology = SkillOntology()
        self.matcher = JobMatcherService(skill_ontology=self.ontology)
        self.parser = JobParserService(skill_ontology=self.ontology)

    def tearDown(self):
        reset_job_matcher()
        reset_job_parser()

    # =========================================================================
    # Fixture 1: Strong Technical Match with Direct Evidence
    # =========================================================================
    def test_fixture_1_strong_technical_match(self):
        """
        Scenario 1: Candidate has direct, verbatim evidence for all required technologies,
        meets education and experience tenure. Expect high score (>= 90%) and zero critical gaps.
        """
        candidate = CandidateProfile(
            name="Sarah Jenkins",
            skills="Python, FastAPI, PostgreSQL, Docker",
            education="Bachelor of Science in Computer Science from State University",
            structured_data=StructuredCV(
                full_name="Sarah Jenkins",
                skills=["Python", "FastAPI", "PostgreSQL", "Docker"],
                experience=[
                    ExperienceItem(
                        title="Senior Software Engineer",
                        company="FinCloud Inc",
                        start_date="2020-01",
                        end_date="2024-01",
                        description="Developed RESTful microservices in Python and FastAPI with PostgreSQL databases and Docker.",
                    )
                ],
                education=[
                    EducationItem(
                        degree="Bachelor of Science",
                        field="Computer Science",
                        institution="State University",
                    )
                ],
            ),
        )

        jd_text = """
Job Title: Senior Backend Developer
Company: Apex Cloud
Requirements:
• 3+ years of professional software engineering experience.
• Hands-on proficiency in Python, FastAPI, and PostgreSQL.
• Bachelor's degree in Computer Science or equivalent.
Preferred Qualifications:
• Experience with Docker.
"""
        parsed_job = self.parser.parse(jd_text)
        result = self.matcher.match(parsed_job, candidate)

        self.assertTrue(result["has_sufficient_evidence"])
        self.assertGreaterEqual(result["overall_score"], 95.0)
        self.assertEqual(len(result["critical_gaps"]), 0)
        self.assertEqual(len(result["missing_requirements"]), 0)

        # Confirm exact outcomes
        matched_reqs = {m.requirement: m.outcome for m in result["matched_requirements"]}
        self.assertIn("Python", matched_reqs)
        self.assertIn("FastAPI", matched_reqs)
        self.assertIn("PostgreSQL", matched_reqs)
        self.assertIn(matched_reqs["Python"], (RequirementOutcome.DIRECT_MATCH, RequirementOutcome.DEMONSTRATED))

    # =========================================================================
    # Fixture 2: Candidate with Related but Non-Equivalent Technologies
    # =========================================================================
    def test_fixture_2_related_non_equivalent_technologies(self):
        """
        Scenario 2: Candidate has Docker and PostgreSQL; role requires Kubernetes and Redis.
        Expect contextual evidence reporting without substitution credit (0.0 score).
        Both missing requirements must be flagged as critical gaps.
        """
        candidate = CandidateProfile(
            name="Jordan Lee",
            skills="Docker, PostgreSQL, Python",
            structured_data=StructuredCV(
                full_name="Jordan Lee",
                skills=["Docker", "PostgreSQL", "Python"],
                experience=[
                    ExperienceItem(
                        title="Backend Developer",
                        company="AppScale",
                        start_date="2021-01",
                        end_date="2023-01",
                        description="Maintained SQL schemas in PostgreSQL and ran containers in Docker.",
                    )
                ],
            ),
        )

        job = ParsedJobDescription(
            job_title="DevOps & Cache Specialist",
            company="CacheCloud",
            required_skills=["Kubernetes", "Redis"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Kubernetes, Redis",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)

        # Required skills score must be 0.0 because neither Docker nor PostgreSQL substitute for K8s/Redis
        self.assertEqual(result["category_scores"]["required_skills"].score, 0.0)
        self.assertEqual(result["overall_score"], 0.0)

        # Contextual evidence recognized
        contextual_skills = [m.requirement for m in result["contextual_evidence"]]
        self.assertIn("Kubernetes", contextual_skills)
        self.assertIn("Redis", contextual_skills)

        # Critical gaps prominently flagged
        critical_skills = [m.requirement for m in result["critical_gaps"]]
        self.assertIn("Kubernetes", critical_skills)
        self.assertIn("Redis", critical_skills)

    # =========================================================================
    # Fixture 3: Fresher Applying to a Role Requesting Professional Experience
    # =========================================================================
    def test_fixture_3_fresher_applying_for_experienced_role(self):
        """
        Scenario 3: Fresher with CS degree and Python/FastAPI projects, but 0 professional experience,
        applying to a 3-year role. Technical skills match, but experience is flagged as a critical gap.
        """
        candidate = CandidateProfile(
            name="Rahul Sharma",
            skills="Python, FastAPI, SQL",
            education="Bachelor of Technology in Computer Science",
            structured_data=StructuredCV(
                full_name="Rahul Sharma",
                skills=["Python", "FastAPI", "SQL"],
                experience=[],  # No professional employment
                education=[
                    EducationItem(
                        degree="Bachelor of Technology",
                        field="Computer Science",
                        institution="National Institute of Technology",
                    )
                ],
                projects=[
                    ProjectItem(
                        name="Campus Placement Portal",
                        description="Built full-stack backend with Python and FastAPI",
                        technologies=["Python", "FastAPI"],
                    )
                ],
            ),
        )

        job = ParsedJobDescription(
            job_title="Mid-Level Backend Engineer",
            company="Enterprise Corp",
            required_skills=["Python", "FastAPI"],
            preferred_skills=[],
            required_qualifications=["Bachelor's degree in Computer Science"],
            preferred_qualifications=[],
            minimum_experience_years=3.0,
            raw_text="Requires 3+ years experience, Python, FastAPI, Bachelor's degree",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)

        # Technical skills and education match
        self.assertEqual(result["category_scores"]["required_skills"].score, 100.0)
        self.assertEqual(result["category_scores"]["education"].score, 100.0)

        # Experience score is 0.0 and isolated in critical_gaps
        self.assertEqual(result["category_scores"]["experience"].score, 0.0)
        crit_categories = [m.category for m in result["critical_gaps"]]
        self.assertIn("experience", crit_categories)

        # Explanation highlights entry-level status
        exp_gap = next(m for m in result["critical_gaps"] if m.category == "experience")
        self.assertIn("entry-level profile", exp_gap.explanation.lower())

    # =========================================================================
    # Fixture 4: Internship and Projects vs Full-Time Employment
    # =========================================================================
    def test_fixture_4_internship_and_projects_distinguished(self):
        """
        Scenario 4: Candidate has 6 months as 'Software Engineering Intern' and 2 projects.
        Job requires 2 years of professional experience. Internships do not grant 1:1 tenure credit.
        """
        candidate = CandidateProfile(
            name="Emily Chen",
            skills="Python, Docker",
            structured_data=StructuredCV(
                full_name="Emily Chen",
                skills=["Python", "Docker"],
                experience=[
                    ExperienceItem(
                        title="Software Engineering Intern",
                        company="Startup Labs",
                        start_date="2023-01",
                        end_date="2023-06",
                        description="Developed internal tool prototypes.",
                    )
                ],
                projects=[
                    ProjectItem(name="Project Alpha", technologies=["Python"]),
                    ProjectItem(name="Project Beta", technologies=["Docker"]),
                ],
            ),
        )

        job = ParsedJobDescription(
            job_title="Junior Software Engineer",
            company="GrowthCorp",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=2.0,
            raw_text="Requires Python and 2+ years of professional experience.",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)
        exp_match = [m for m in result["missing_requirements"] if m.category == "experience"]
        self.assertEqual(len(exp_match), 1)
        self.assertIn("internship", exp_match[0].resume_evidence.lower())
        self.assertIn("critical_gaps", result)
        self.assertTrue(any(m.category == "experience" for m in result["critical_gaps"]))

    # =========================================================================
    # Fixture 5: Resume with Missing Dates
    # =========================================================================
    def test_fixture_5_resume_with_missing_dates(self):
        """
        Scenario 5: Candidate lists past roles, but employment dates cannot be parsed.
        Expect explicit uncertainty flag without guessing or fabricating 18-month tenure.
        """
        candidate = CandidateProfile(
            name="David Miller",
            skills="Python, SQL",
            structured_data=StructuredCV(
                full_name="David Miller",
                skills=["Python", "SQL"],
                experience=[
                    ExperienceItem(
                        title="Backend Developer",
                        company="OldCorp",
                        start_date="Undisclosed",
                        end_date="Past",
                        description="Worked on APIs.",
                    )
                ],
            ),
        )

        job = ParsedJobDescription(
            job_title="Developer",
            company="TechGroup",
            required_skills=["Python"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=2.0,
            raw_text="Requires Python and 2 years experience.",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)
        self.assertGreater(len(result["uncertain_requirements"]), 0)
        self.assertTrue(any("could not be verified" in msg.lower() for msg in result["uncertain_requirements"]))
        self.assertEqual(result["category_scores"]["experience"].score, 0.0)

    # =========================================================================
    # Fixture 6: Job Description with No Clearly Stated Mandatory Skills
    # =========================================================================
    def test_fixture_6_job_without_mandatory_skills_section(self):
        """
        Scenario 6: Job description has general duties and overview but no explicit 'Requirements' section.
        Weights dynamically redistribute across active categories without raising errors or dividing by zero.
        """
        jd_text = """
Job Title: Engineering Team Member
Company: Global Systems

About the Team:
Join our collaborative platform engineering team building next-generation systems.

Key Responsibilities:
• Attend sprint standups and communicate architecture decisions.
• Collaborate with product managers on backlog refinement.
• Conduct code reviews and ensure testing standards.

Requirements:
• Bachelor's degree in Computer Science.
• 2+ years of experience.
"""
        parsed_job = self.parser.parse(jd_text)
        candidate = CandidateProfile(
            name="Jane Doe",
            education="Bachelor of Science in Computer Science",
            structured_data=StructuredCV(
                full_name="Jane Doe",
                education=[
                    EducationItem(
                        degree="Bachelor of Science",
                        field="Computer Science",
                        institution="State Tech",
                    )
                ],
                experience=[
                    ExperienceItem(
                        title="Software Engineer",
                        company="Legacy Systems",
                        start_date="2021-01",
                        end_date="2023-06",
                    )
                ],
            ),
        )

        result = self.matcher.match(parsed_job, candidate)
        self.assertTrue(result["has_sufficient_evidence"])

        # When required_skills is empty, active categories (experience + education) weights sum to 1.0
        cat_scores = result["category_scores"]
        self.assertNotIn("required_skills", cat_scores)
        self.assertIn("experience", cat_scores)
        self.assertIn("education", cat_scores)
        total_weight = sum(cs.weight for cs in cat_scores.values())
        self.assertAlmostEqual(total_weight, 1.0, places=2)
        self.assertEqual(result["overall_score"], 100.0)

    # =========================================================================
    # Fixture 7: Job Description with Ambiguous Requirements
    # =========================================================================
    def test_fixture_7_ambiguous_job_description(self):
        """
        Scenario 7: Vague job posting ("rockstar, ninja, positive vibes") lacks extractable
        technical requirements. Expect has_sufficient_evidence = False and overall_score = 0.0.
        """
        jd_text = """
We are looking for a rockstar coder and passionate team player to join our high-energy vibe!
Must be a self-starter with a growth mindset and great communication skills.
No corporate red tape, just pure creativity and passion!
"""
        parsed_job = self.parser.parse(jd_text)
        candidate = CandidateProfile(
            name="Alex Smith",
            skills="Python, SQL",
            structured_data=StructuredCV(
                full_name="Alex Smith",
                skills=["Python", "SQL"],
            ),
        )

        result = self.matcher.match(parsed_job, candidate)
        self.assertFalse(result["has_sufficient_evidence"])
        self.assertEqual(result["overall_score"], 0.0)
        self.assertTrue(any("does not contain any extractable" in msg for msg in result["limitations"]))

    # =========================================================================
    # Fixture 8: Matches Many Preferred Skills but Lacks Mandatory Requirement
    # =========================================================================
    def test_fixture_8_matches_preferred_but_lacks_mandatory(self):
        """
        Scenario 8: Candidate demonstrates all preferred technologies (AWS, Docker, Redis),
        but lacks the single mandatory requirement (Python).
        Verify Python is prominently isolated in critical_gaps despite a high preferred score.
        """
        candidate = CandidateProfile(
            name="Marcus Vance",
            skills="AWS, Docker, Redis",
            structured_data=StructuredCV(
                full_name="Marcus Vance",
                skills=["AWS", "Docker", "Redis"],
            ),
        )

        job = ParsedJobDescription(
            job_title="Python Cloud Developer",
            company="DataCloud",
            required_skills=["Python"],  # MANDATORY
            preferred_skills=["AWS", "Docker", "Redis"],  # PREFERRED
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Python. Preferred: AWS, Docker, Redis",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)

        # Preferred skills score is 100%
        self.assertEqual(result["category_scores"]["preferred_skills"].score, 100.0)
        # Required skills score is 0.0%
        self.assertEqual(result["category_scores"]["required_skills"].score, 0.0)

        # Mandatory Python MUST be in critical_gaps
        critical_names = [m.requirement for m in result["critical_gaps"]]
        self.assertIn("Python", critical_names)
        self.assertEqual(len(critical_names), 1)

    # =========================================================================
    # Fixture 9: Resume Omits Evidence Without Proving Lack of Skill
    # =========================================================================
    def test_fixture_9_omitted_evidence_phrasing(self):
        """
        Scenario 9: Candidate has Python and SQL, but resume text omits Git.
        Verify non-punitive phrasing: states absent resume evidence, not proof of incapacity.
        """
        candidate = CandidateProfile(
            name="Clara Oswald",
            skills="Python, SQL",
            structured_data=StructuredCV(
                full_name="Clara Oswald",
                skills=["Python", "SQL"],
            ),
        )

        job = ParsedJobDescription(
            job_title="Software Developer",
            company="Acme",
            required_skills=["Git"],
            preferred_skills=[],
            required_qualifications=[],
            preferred_qualifications=[],
            minimum_experience_years=None,
            raw_text="Required: Git",
            extraction_source="deterministic",
        )

        result = self.matcher.match(job, candidate)
        missing_match = result["missing_requirements"][0]

        self.assertEqual(missing_match.outcome, RequirementOutcome.NOT_FOUND_IN_RESUME)
        self.assertIn("absent resume evidence, not proof of incapacity", missing_match.explanation)
        self.assertNotIn("incapable", missing_match.explanation.lower())
        self.assertNotIn("unqualified", missing_match.explanation.lower())

    # =========================================================================
    # Fixture 10: Formatting Challenges Deterministic Section Extraction
    # =========================================================================
    def test_fixture_10_challenging_section_formatting(self):
        """
        Scenario 10: Job description uses irregular headers ('### WHAT WE EXPECT:', '### NICE PERKS:')
        and non-standard bullet points ('->', '*').
        Verify deterministic parser correctly captures required skills, preferred skills, and experience.
        """
        jd_text = """
JOB TITLE: Full Stack Engineer
COMPANY: InnovateTech

### ABOUT US:
InnovateTech is redefining data platforms.

### YOUR DAY-TO-DAY RESPONSIBILITIES:
-> Architect scalable REST APIs using Python and FastAPI.
-> Build reliable database migrations in PostgreSQL.

### WHAT WE EXPECT OF YOU:
* Minimum 3 years of software engineering experience.
* Bachelor's degree in Computer Science or related.
* Strong proficiency in Python, FastAPI, and PostgreSQL.

### NICE PERKS TO HAVE:
* Experience with Docker and Redis.
"""
        parsed = self.parser.parse(jd_text)

        self.assertEqual(parsed.job_title, "Full Stack Engineer")
        self.assertEqual(parsed.company, "InnovateTech")
        self.assertEqual(parsed.minimum_experience_years, 3.0)
        self.assertIn("Python", parsed.required_skills)
        self.assertIn("FastAPI", parsed.required_skills)
        self.assertIn("PostgreSQL", parsed.required_skills)
        self.assertIn("Docker", parsed.preferred_skills)
        self.assertIn("Redis", parsed.preferred_skills)


if __name__ == "__main__":
    unittest.main()
