import json
import unittest
from unittest.mock import MagicMock, patch

from backend.models.schemas import (
    StructuredCV,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    CertificationItem,
    CandidateProfile,
)
from backend.services.gemini import GeminiService
from backend.services.candidate_profile import CandidateProfileBuilder


class TestGeminiStructuredExtraction(unittest.TestCase):
    """Unit tests for GeminiService structured extraction and schema validation."""

    def setUp(self):
        """Set up test fixtures."""
        # Unconfigured service (using dummy/placeholder key to test defensive handling)
        self.service = GeminiService(api_key="your_gemini_api_key_here")
        self.profile_builder = CandidateProfileBuilder()

    def test_unconfigured_initialization(self):
        """Test service initializes safely without valid key and defaults to unconfigured."""
        service = GeminiService(api_key="")
        self.assertFalse(service.is_configured)

        placeholder_service = GeminiService(api_key="your_gemini_api_key_here")
        self.assertFalse(placeholder_service.is_configured)

    def test_clean_llm_json_response(self):
        """Test that markdown code fences and whitespace are stripped from LLM response."""
        raw_markdown = "```json\n{\"full_name\": \"John Doe\"}\n```"
        cleaned = self.service._clean_llm_json_response(raw_markdown)
        self.assertEqual(cleaned, '{"full_name": "John Doe"}')

        raw_fence_no_lang = "```\n{\"full_name\": \"Jane Doe\"}\n```"
        cleaned2 = self.service._clean_llm_json_response(raw_fence_no_lang)
        self.assertEqual(cleaned2, '{"full_name": "Jane Doe"}')

    def test_validate_and_sanitize_cv_dict_complete(self):
        """Test schema validation on a complete dictionary payload."""
        sample_dict = {
            "full_name": "Alice Johnson",
            "email": "alice@example.com",
            "phone": "+1-555-0199",
            "location": "Bengaluru, India",
            "linkedin_url": "https://linkedin.com/in/alice",
            "github_url": "https://github.com/alice",
            "portfolio_url": "https://alice.dev",
            "summary": "Experienced backend engineer.",
            "education": [
                {
                    "institution": "National Institute of Technology",
                    "degree": "B.Tech",
                    "field": "Computer Science",
                    "start_year": 2018,
                    "end_year": 2022,
                    "gpa": 8.7
                }
            ],
            "experience": [
                {
                    "company": "Tech Corp",
                    "title": "Software Engineer",
                    "start_date": "July 2022",
                    "end_date": "Present",
                    "description": "Built resilient payment APIs using FastAPI."
                }
            ],
            "projects": [
                {
                    "name": "Cloud Monitor",
                    "description": "Real-time metrics dashboard",
                    "technologies": ["Python", "FastAPI", "React"],
                    "link": "https://github.com/alice/cloud-monitor"
                }
            ],
            "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
            "certifications": [
                {
                    "name": "AWS Certified Developer",
                    "issuer": "Amazon Web Services",
                    "date_obtained": "2023",
                    "expiry_date": None
                }
            ]
        }

        cv = self.service.validate_and_sanitize_cv_dict(sample_dict)
        self.assertIsInstance(cv, StructuredCV)
        self.assertEqual(cv.full_name, "Alice Johnson")
        self.assertEqual(cv.email, "alice@example.com")
        self.assertEqual(len(cv.education), 1)
        self.assertEqual(cv.education[0].institution, "National Institute of Technology")
        self.assertEqual(cv.education[0].start_year, 2018)
        self.assertEqual(len(cv.experience), 1)
        self.assertEqual(cv.experience[0].company, "Tech Corp")
        self.assertEqual(len(cv.projects), 1)
        self.assertEqual(cv.projects[0].technologies, ["Python", "FastAPI", "React"])
        self.assertEqual(len(cv.skills), 4)
        self.assertEqual(len(cv.certifications), 1)

    def test_validate_and_sanitize_type_coercion(self):
        """Test that string years and loose types are safely coerced."""
        loose_dict = {
            "full_name": "Bob Smith",
            "education": [
                {
                    "institution": "University X",
                    "degree": "B.S.",
                    "field": "IT",
                    "start_year": "2019-08",
                    "end_year": "2023-05",
                    "gpa": "3.85"
                }
            ],
            "projects": [
                {
                    "name": "Project Alpha",
                    "technologies": "Python, SQL, AWS"  # comma-separated string instead of list
                }
            ],
            "skills": "Python, Docker, Linux"  # comma-separated string instead of list
        }

        cv = self.service.validate_and_sanitize_cv_dict(loose_dict)
        self.assertEqual(cv.education[0].start_year, 2019)
        self.assertEqual(cv.education[0].end_year, 2023)
        self.assertEqual(cv.education[0].gpa, 3.85)
        self.assertEqual(cv.projects[0].technologies, ["Python", "SQL", "AWS"])
        self.assertEqual(cv.skills, ["Python", "Docker", "Linux"])

    def test_empty_raw_text_handling(self):
        """Test that passing empty text returns empty StructuredCV without crashing."""
        cv = self.service.extract_structured_cv("")
        self.assertIsInstance(cv, StructuredCV)
        self.assertEqual(cv.full_name, "")
        self.assertEqual(len(cv.education), 0)
        self.assertEqual(len(cv.experience), 0)

    def test_fallback_rule_based_extraction(self):
        """Test rule-based fallback when Gemini is unconfigured."""
        resume_text = (
            "Robert Taylor\nrobert.taylor@example.com\n+1-555-888-9999\n"
            "linkedin.com/in/roberttaylor\ngithub.com/roberttaylor\n\n"
            "Summary\nExperienced developer.\n\n"
            "Skills: Python, Go, Docker, Kubernetes, SQL"
        )
        cv = self.service.extract_structured_cv(resume_text)
        self.assertIsInstance(cv, StructuredCV)
        self.assertEqual(cv.full_name, "Robert Taylor")
        self.assertEqual(cv.email, "robert.taylor@example.com")
        self.assertIn("555", cv.phone)
        self.assertEqual(cv.linkedin_url, "https://linkedin.com/in/roberttaylor")
        self.assertEqual(cv.github_url, "https://github.com/roberttaylor")
        self.assertIn("Python", cv.skills)
        self.assertIn("Docker", cv.skills)

    def test_mock_gemini_successful_extraction(self):
        """Test successful Gemini API execution with mock response."""
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()

        mock_payload = {
            "full_name": "Devin Miller",
            "email": "devin@example.com",
            "phone": "+1-555-4321",
            "location": "San Francisco, CA",
            "linkedin_url": "https://linkedin.com/in/devinm",
            "github_url": "https://github.com/devinm",
            "portfolio_url": "https://devin.io",
            "summary": "Full Stack Engineer with 3 years experience.",
            "education": [
                {
                    "institution": "Stanford University",
                    "degree": "B.S.",
                    "field": "Computer Science",
                    "start_year": 2017,
                    "end_year": 2021,
                    "gpa": 3.9
                }
            ],
            "experience": [
                {
                    "company": "NextGen AI",
                    "title": "Full Stack Developer",
                    "start_date": "2021",
                    "end_date": "Present",
                    "description": "Architected modern React and FastAPI interfaces."
                }
            ],
            "projects": [
                {
                    "name": "AI Search Engine",
                    "description": "Vector search engine using embeddings",
                    "technologies": ["Python", "FAISS", "React"],
                    "link": "https://github.com/devinm/ai-search"
                }
            ],
            "skills": ["Python", "FastAPI", "React", "TypeScript", "Docker"],
            "certifications": []
        }

        mock_response = MagicMock()
        mock_response.text = json.dumps(mock_payload)
        mock_model.generate_content.return_value = mock_response
        mock_service.model = mock_model

        cv = mock_service.extract_structured_cv("Raw resume sample text")
        self.assertEqual(cv.full_name, "Devin Miller")
        self.assertEqual(cv.email, "devin@example.com")
        self.assertEqual(len(cv.education), 1)
        self.assertEqual(cv.education[0].institution, "Stanford University")
        self.assertEqual(len(cv.experience), 1)
        self.assertEqual(cv.experience[0].company, "NextGen AI")
        self.assertEqual(len(cv.projects), 1)
        self.assertEqual(cv.projects[0].name, "AI Search Engine")

    def test_mock_gemini_failure_triggers_fallback(self):
        """Test that API exceptions gracefully degrade to fallback extraction without throwing."""
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()
        mock_model.generate_content.side_effect = RuntimeError("API quota exhausted or network error")
        mock_service.model = mock_model

        resume_text = "Carol Danvers\ncarol@example.com\nSkills: Python, Cloud, Security"
        # Must not raise an exception; should use fallback
        cv = mock_service.extract_structured_cv(resume_text)
        self.assertIsInstance(cv, StructuredCV)
        self.assertEqual(cv.full_name, "Carol Danvers")
        self.assertEqual(cv.email, "carol@example.com")
        self.assertIn("Python", cv.skills)

    def test_candidate_profile_builder_with_structured_cv(self):
        """Test CandidateProfileBuilder integration with StructuredCV."""
        structured_cv = StructuredCV(
            full_name="Samantha Ray",
            email="samantha@example.com",
            phone="+1-555-7777",
            location="Seattle, WA",
            linkedin_url="https://linkedin.com/in/samanthar",
            github_url="https://github.com/samanthar",
            portfolio_url="https://samantha.dev",
            summary="Senior Machine Learning Specialist.",
            education=[
                EducationItem(
                    institution="University of Washington",
                    degree="M.S.",
                    field="Computer Science",
                    start_year=2019,
                    end_year=2021,
                    gpa=3.95
                )
            ],
            experience=[
                ExperienceItem(
                    company="ML Systems Inc",
                    title="ML Engineer",
                    start_date="2021",
                    end_date="Present",
                    description="Trained foundation models."
                )
            ],
            projects=[
                ProjectItem(
                    name="Vision Pipeline",
                    description="Edge video inference",
                    technologies=["PyTorch", "C++"],
                    link="https://github.com/samanthar/vision"
                )
            ],
            skills=["Python", "PyTorch", "Kubernetes", "CUDA"],
            certifications=[
                CertificationItem(
                    name="NVIDIA Deep Learning Certified",
                    issuer="NVIDIA",
                    date_obtained="2022",
                    expiry_date=None
                )
            ]
        )

        parsed_sections = {
            "summary": "Summary text",
            "raw_text": "Sample raw text"
        }

        profile = self.profile_builder.build(parsed_sections, structured_cv=structured_cv)
        self.assertIsInstance(profile, CandidateProfile)

        # Name should now be populated from structured extraction
        self.assertEqual(profile.personal.name, "Samantha Ray")
        self.assertEqual(profile.personal.email, "samantha@example.com")
        self.assertEqual(profile.personal.location, "Seattle, WA")
        self.assertIsNotNone(profile.structured_data)
        self.assertEqual(profile.structured_data.full_name, "Samantha Ray")
        self.assertEqual(len(profile.structured_data.education), 1)
        self.assertEqual(len(profile.structured_data.experience), 1)
        self.assertEqual(len(profile.structured_data.projects), 1)
        self.assertEqual(len(profile.structured_data.skills), 4)

    def test_missing_education_start_year_remains_none(self):
        """Test that missing or unknown education start year is preserved as None (not fabricated to 2000)."""
        data = {
            "education": [
                {
                    "institution": "Tech University",
                    "degree": "B.Tech",
                    "field": "CS",
                    "end_year": 2024,
                    # start_year is omitted
                }
            ]
        }
        cv = self.service.validate_and_sanitize_cv_dict(data)
        self.assertEqual(len(cv.education), 1)
        self.assertIsNone(cv.education[0].start_year)
        self.assertEqual(cv.education[0].end_year, 2024)

    def test_title_prefixed_resume_fallback_does_not_set_title_as_name(self):
        """Test that conservative name fallback never assigns job titles as candidate names."""
        # 1. Headline before name
        text1 = "Full Stack Developer\nJohn Doe\njohn@example.com"
        cv1 = self.service.extract_structured_cv(text1)
        self.assertEqual(cv1.full_name, "John Doe")

        # 2. Headline with no candidate name
        text2 = "Software Engineer\nengineer@example.com\nSkills: Python, Go"
        cv2 = self.service.extract_structured_cv(text2)
        self.assertEqual(cv2.full_name, "")  # Remains empty rather than misidentifying title

        # 3. Data Scientist headline before name
        text3 = "Data Scientist\nJane Smith\njane@example.com"
        cv3 = self.service.extract_structured_cv(text3)
        self.assertEqual(cv3.full_name, "Jane Smith")

    def test_extraction_source_tagging(self):
        """Test that extraction_source distinguishes Gemini vs fallback."""
        # Fallback path
        fallback_cv = self.service.extract_structured_cv("Sample text with no Gemini key")
        self.assertEqual(fallback_cv.extraction_source, "fallback")

        # Successful Gemini mock path
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = json.dumps({"full_name": "Test User", "skills": ["Python"]})
        mock_model.generate_content.return_value = mock_response
        mock_service.model = mock_model

        gemini_cv = mock_service.extract_structured_cv("Sample text")
        self.assertEqual(gemini_cv.extraction_source, "gemini")

    def test_mock_gemini_timeout_triggers_fallback(self):
        """Test that an API timeout triggers fallback extraction gracefully."""
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()

        # Simulate timeout inside generate_content
        import concurrent.futures
        def mock_generate_content(*args, **kwargs):
            raise TimeoutError("Gemini API call timed out after 30 seconds")

        mock_model.generate_content.side_effect = mock_generate_content
        mock_service.model = mock_model

        resume_text = "Alice Cooper\nalice@example.com\nSkills: Python"
        cv = mock_service.extract_structured_cv(resume_text)
        self.assertIsInstance(cv, StructuredCV)
        self.assertEqual(cv.full_name, "Alice Cooper")
        self.assertEqual(cv.extraction_source, "fallback")

    def test_mock_gemini_empty_or_whitespace_response_triggers_fallback(self):
        """Test that empty or blank response from Gemini triggers fallback without crashing."""
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()

        # Empty response
        mock_response = MagicMock()
        mock_response.text = "   "
        mock_model.generate_content.return_value = mock_response
        mock_service.model = mock_model

        resume_text = "Brian Adams\nbrian@example.com\nSkills: Java"
        cv = mock_service.extract_structured_cv(resume_text)
        self.assertEqual(cv.full_name, "Brian Adams")
        self.assertEqual(cv.extraction_source, "fallback")

    def test_mock_gemini_malformed_json_triggers_fallback(self):
        """Test that malformed JSON response from Gemini triggers fallback without crashing."""
        mock_service = GeminiService()
        mock_service._is_configured = True
        mock_model = MagicMock()

        mock_response = MagicMock()
        mock_response.text = "{invalid json: 123,"
        mock_model.generate_content.return_value = mock_response
        mock_service.model = mock_model

        resume_text = "Diana Prince\ndiana@example.com\nSkills: Security"
        cv = mock_service.extract_structured_cv(resume_text)
        self.assertEqual(cv.full_name, "Diana Prince")
        self.assertEqual(cv.extraction_source, "fallback")

    def test_validate_and_sanitize_invalid_root_type(self):
        """Test that passing non-dictionary data raises ValidationError."""
        from pydantic import ValidationError
        with self.assertRaises(ValidationError):
            self.service.validate_and_sanitize_cv_dict("not a dictionary")

    def test_candidate_profile_builder_backward_compatibility_preserved(self):
        """Verify CandidateProfileBuilder without structured_cv leaves name empty as expected."""
        parsed_sections = {
            "summary": "Summary",
            "raw_text": "Sample text"
        }
        profile = self.profile_builder.build(parsed_sections)
        self.assertEqual(profile.personal.name, "")
        self.assertIsNone(profile.structured_data)


if __name__ == "__main__":
    unittest.main()

