"""
Unit and integration tests for JobParserService.

Verifies:
1. Extraction of required vs. preferred skills using SkillOntology.
2. Experience years parsing across varied regex patterns.
3. Educational qualification extraction (degrees, fields).
4. Responsibilities extraction.
5. Deterministic baseline guarantees (no hallucinated requirements).
6. Gemini structured parsing and fallback on failure.
7. Empty and ambiguous input handling.
"""

import unittest
from unittest.mock import MagicMock, patch

from backend.services.gemini import GeminiService
from backend.services.job_parser import JobParserService, reset_job_parser
from backend.services.skill_ontology import SkillOntology


class TestJobParserService(unittest.TestCase):
    """Test suite for JobParserService deterministic parsing and fallback."""

    def setUp(self):
        reset_job_parser()
        self.ontology = SkillOntology()
        # Initialize unconfigured GeminiService for deterministic testing
        unconfigured_gemini = GeminiService.__new__(GeminiService)
        unconfigured_gemini.api_key = None
        unconfigured_gemini._is_configured = False
        self.parser = JobParserService(
            skill_ontology=self.ontology,
            gemini_service=unconfigured_gemini,
        )

    def tearDown(self):
        reset_job_parser()

    def test_deterministic_parsing_extracts_title_company_and_sections(self):
        """Test extraction of title, employer, required skills, preferred skills, and experience."""
        jd_text = """
Job Title: Senior Backend Engineer
Company: CloudScale Systems

About Us:
CloudScale Systems is building next-generation cloud infrastructure.

Responsibilities:
• Design and maintain scalable microservice architectures using Python and FastAPI.
• Build reliable distributed systems and event-driven data pipelines.
• Collaborate with cross-functional teams to improve platform performance.

Requirements:
• Minimum 4 years of software engineering experience.
• Bachelor's degree in Computer Science or equivalent.
• Strong hands-on proficiency in Python, FastAPI, and PostgreSQL.
• Experience with Docker and RESTful APIs.

Preferred Qualifications:
• Master's degree in a technical field.
• Experience with Kubernetes, Redis, and AWS.
• Familiarity with CI/CD automation.
"""
        parsed = self.parser.parse(jd_text)

        self.assertEqual(parsed.job_title, "Senior Backend Engineer")
        self.assertEqual(parsed.company, "CloudScale Systems")
        self.assertEqual(parsed.minimum_experience_years, 4.0)
        self.assertEqual(parsed.extraction_source, "deterministic")

        # Required skills
        self.assertIn("Python", parsed.required_skills)
        self.assertIn("FastAPI", parsed.required_skills)
        self.assertIn("PostgreSQL", parsed.required_skills)
        self.assertIn("Docker", parsed.required_skills)

        # Preferred skills
        self.assertIn("Kubernetes", parsed.preferred_skills)
        self.assertIn("Redis", parsed.preferred_skills)
        self.assertIn("AWS", parsed.preferred_skills)

        # Ensure preferred skills do NOT leak into required skills
        self.assertNotIn("Kubernetes", parsed.required_skills)
        self.assertNotIn("Redis", parsed.required_skills)

        # Qualifications
        self.assertTrue(any("bachelor" in q.lower() for q in parsed.required_qualifications))
        self.assertTrue(any("master" in q.lower() for q in parsed.preferred_qualifications))

        # Responsibilities
        self.assertGreaterEqual(len(parsed.responsibilities), 2)
        self.assertTrue(any("microservice" in r.lower() for r in parsed.responsibilities))

    def test_experience_years_extraction_varied_patterns(self):
        """Test experience extraction across common phrasing variants."""
        # Pattern 1: "3+ years of experience"
        p1 = self.parser.parse("Software Developer\nRequirements:\nMust have 3+ years of experience in Python.")
        self.assertEqual(p1.minimum_experience_years, 3.0)

        # Pattern 2: "minimum 5 years"
        p2 = self.parser.parse("DevOps Engineer\nRequirements:\nMinimum 5 years of industry experience with AWS.")
        self.assertEqual(p2.minimum_experience_years, 5.0)

        # Pattern 3: "2-4 years of professional experience"
        p3 = self.parser.parse("Frontend Engineer\nRequirements:\n2-4 years of professional experience with React.")
        self.assertEqual(p3.minimum_experience_years, 2.0)

        # Pattern 4: No experience stated
        p4 = self.parser.parse("Junior Web Developer\nRequirements:\nProficiency in HTML, CSS, JavaScript.")
        self.assertIsNone(p4.minimum_experience_years)

    def test_overrides_for_title_and_company_hints(self):
        """User-supplied hints take precedence over ambiguous parsed text."""
        jd_text = "Looking for a rockstar coder to join our high-growth startup.\nRequirements: Python, SQL."
        parsed = self.parser.parse(
            jd_text,
            job_title_hint="Lead Software Engineer",
            company_hint="Acme Inc",
        )
        self.assertEqual(parsed.job_title, "Lead Software Engineer")
        self.assertEqual(parsed.company, "Acme Inc")

    def test_no_hallucinated_requirements(self):
        """Parser must never invent skills that do not appear in the text."""
        jd_text = """
Job Title: Python Developer
Company: DataTech

Requirements:
- 2+ years of experience in Python and SQL.
- Good communication skills.
"""
        parsed = self.parser.parse(jd_text)
        self.assertIn("Python", parsed.required_skills)
        self.assertIn("SQL", parsed.required_skills)

        # Verify unrelated skills are not hallucinated
        self.assertNotIn("Rust", parsed.required_skills)
        self.assertNotIn("Kubernetes", parsed.required_skills)
        self.assertNotIn("Java", parsed.required_skills)
        self.assertNotIn("React", parsed.required_skills)

    def test_empty_or_whitespace_job_description_raises_value_error(self):
        """Parser rejects empty or whitespace-only inputs with ValueError."""
        with self.assertRaises(ValueError):
            self.parser.parse("")

        with self.assertRaises(ValueError):
            self.parser.parse("   \n\t   ")

    def test_gemini_fallback_on_failure(self):
        """When Gemini is configured but throws an error, fallback succeeds deterministically."""
        mock_gemini = MagicMock()
        mock_gemini.is_configured = True
        mock_gemini.model.generate_content.side_effect = TimeoutError("Gemini API timed out")

        parser_with_failing_gemini = JobParserService(
            skill_ontology=self.ontology,
            gemini_service=mock_gemini,
        )

        jd_text = "Role: ML Engineer\nCompany: AI Dynamics\nRequirements: Python, PyTorch, 3+ years experience."
        parsed = parser_with_failing_gemini.parse(jd_text)

        # Fallback executed and extracted correctly
        self.assertEqual(parsed.extraction_source, "deterministic")
        self.assertEqual(parsed.job_title, "ML Engineer")
        self.assertEqual(parsed.company, "AI Dynamics")
        self.assertEqual(parsed.minimum_experience_years, 3.0)
        self.assertIn("Python", parsed.required_skills)

    def test_gemini_parsing_when_successful(self):
        """When Gemini returns valid structured JSON, it is parsed and validated."""
        mock_gemini = MagicMock()
        mock_gemini.is_configured = True
        mock_resp = MagicMock()
        mock_resp.text = """```json
{
  "job_title": "AI Research Scientist",
  "company": "DeepMind Partner",
  "required_skills": ["Python", "PyTorch"],
  "preferred_skills": ["Kubernetes"],
  "required_qualifications": ["PhD in Computer Science"],
  "preferred_qualifications": ["Publications at NeurIPS"],
  "minimum_experience_years": 2.0,
  "responsibilities": ["Train foundation models", "Write research papers"],
  "other_requirements": ["Remote work option"]
}
```"""
        mock_gemini.model.generate_content.return_value = mock_resp

        parser_with_gemini = JobParserService(
            skill_ontology=self.ontology,
            gemini_service=mock_gemini,
        )

        jd_text = "DeepMind Partner seeks AI Research Scientist with PhD."
        parsed = parser_with_gemini.parse(jd_text)

        self.assertEqual(parsed.extraction_source, "gemini")
        self.assertEqual(parsed.job_title, "AI Research Scientist")
        self.assertEqual(parsed.company, "DeepMind Partner")
        self.assertEqual(parsed.required_skills, ["Python", "PyTorch"])
        self.assertEqual(parsed.preferred_skills, ["Kubernetes"])
        self.assertEqual(parsed.minimum_experience_years, 2.0)


if __name__ == "__main__":
    unittest.main()
