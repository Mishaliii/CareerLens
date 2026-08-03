import unittest
from backend.services.candidate_profile import CandidateProfileBuilder
from backend.models.schemas import CandidateProfile


class TestCandidateProfileBuilder(unittest.TestCase):
    """Unit tests for Candidate Profile Builder."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.builder = CandidateProfileBuilder()
    
    def test_complete_resume(self):
        """Test building profile from complete resume."""
        parsed_sections = {
            "summary": "Experienced software engineer with 5 years in full-stack development.",
            "experience": "Senior Developer at Tech Corp (2020-Present)\nJunior Developer at Startup Inc (2018-2020)",
            "education": "B.Tech in Computer Science from ABC University (2018)",
            "projects": "E-commerce Platform\nReal-time Chat Application",
            "skills": "Python, JavaScript, React, Node.js, SQL",
            "certifications": "AWS Certified Developer\nGoogle Cloud Professional",
            "raw_text": "John Doe\njohn.doe@email.com\n+1-555-123-4567\nlinkedin.com/in/johndoe\ngithub.com/johndoe\n\nSummary\nExperienced software engineer with 5 years in full-stack development.\n\nExperience\nSenior Developer at Tech Corp (2020-Present)\nJunior Developer at Startup Inc (2018-2020)\n\nEducation\nB.Tech in Computer Science from ABC University (2018)\n\nProjects\nE-commerce Platform\nReal-time Chat Application\n\nSkills\nPython, JavaScript, React, Node.js, SQL\n\nCertifications\nAWS Certified Developer\nGoogle Cloud Professional",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify profile is created
        self.assertIsInstance(profile, CandidateProfile)
        
        # Verify sections are preserved
        self.assertEqual(profile.summary, parsed_sections["summary"])
        self.assertEqual(profile.experience, parsed_sections["experience"])
        self.assertEqual(profile.education, parsed_sections["education"])
        self.assertEqual(profile.projects, parsed_sections["projects"])
        self.assertEqual(profile.skills, parsed_sections["skills"])
        self.assertEqual(profile.certifications, parsed_sections["certifications"])
        
        # Verify personal info extraction
        self.assertEqual(profile.personal.name, "")  # Name heuristic removed
        self.assertEqual(profile.personal.email, "john.doe@email.com")
        self.assertIn("555", profile.personal.phone)  # Phone format may vary
        self.assertEqual(profile.personal.linkedin, "https://linkedin.com/in/johndoe")
        self.assertEqual(profile.personal.github, "https://github.com/johndoe")
        
        # Verify metadata
        self.assertEqual(profile.metadata.profile_schema_version, "1.0.0")
        self.assertEqual(profile.metadata.parser_version, "1.0.0")
        self.assertIsNone(profile.metadata.parser_confidence)
        self.assertGreater(profile.metadata.word_count, 0)
        self.assertIsNone(profile.metadata.page_count)
        self.assertIsNone(profile.metadata.language)
        
        # Verify statistics
        self.assertGreater(profile.resume_statistics.word_count, 0)
        self.assertGreater(profile.resume_statistics.skills_count, 0)
    
    def test_resume_with_missing_sections(self):
        """Test building profile from resume with missing sections."""
        parsed_sections = {
            "summary": "Entry-level developer looking for opportunities.",
            "experience": "",
            "education": "B.Sc in Computer Science from XYZ College (2023)",
            "projects": "",
            "skills": "Python, Java, HTML, CSS",
            "certifications": "",
            "raw_text": "Jane Smith\njane.smith@email.com\n\nSummary\nEntry-level developer looking for opportunities.\n\nEducation\nB.Sc in Computer Science from XYZ College (2023)\n\nSkills\nPython, Java, HTML, CSS",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify profile is created
        self.assertIsInstance(profile, CandidateProfile)
        
        # Verify present sections
        self.assertEqual(profile.summary, parsed_sections["summary"])
        self.assertEqual(profile.education, parsed_sections["education"])
        self.assertEqual(profile.skills, parsed_sections["skills"])
        
        # Verify missing sections are empty
        self.assertEqual(profile.experience, "")
        self.assertEqual(profile.projects, "")
        self.assertEqual(profile.certifications, "")
        self.assertEqual(profile.achievements, "")
        self.assertEqual(profile.languages_spoken, "")
    
    def test_empty_parser_output(self):
        """Test building profile from empty parser output."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify profile is created with all fields
        self.assertIsInstance(profile, CandidateProfile)
        
        # Verify all sections are empty
        self.assertEqual(profile.summary, "")
        self.assertEqual(profile.experience, "")
        self.assertEqual(profile.education, "")
        self.assertEqual(profile.projects, "")
        self.assertEqual(profile.skills, "")
        self.assertEqual(profile.certifications, "")
        self.assertEqual(profile.achievements, "")
        self.assertEqual(profile.languages_spoken, "")
        
        # Verify personal info is empty
        self.assertEqual(profile.personal.name, "")
        self.assertEqual(profile.personal.email, "")
        self.assertEqual(profile.personal.phone, "")
        self.assertEqual(profile.personal.location, "")
        self.assertEqual(profile.personal.linkedin, "")
        self.assertEqual(profile.personal.github, "")
        self.assertEqual(profile.personal.portfolio, "")
        
        # Verify statistics are zero
        self.assertEqual(profile.resume_statistics.word_count, 0)
        self.assertEqual(profile.resume_statistics.bullet_count, 0)
        self.assertEqual(profile.resume_statistics.experience_count, 0)
        self.assertEqual(profile.resume_statistics.project_count, 0)
        self.assertEqual(profile.resume_statistics.certification_count, 0)
        self.assertEqual(profile.resume_statistics.skills_count, 0)
    
    def test_statistics_calculation(self):
        """Test resume statistics calculation."""
        parsed_sections = {
            "summary": "Software engineer with experience.",
            "experience": "Senior Developer at Tech Corp\n- Led team of 5 engineers\n* Improved performance by 40%\n• Implemented CI/CD pipeline",
            "education": "B.Tech in Computer Science",
            "projects": "E-commerce Platform\n- Built using React\n* Integrated payment gateway",
            "skills": "Python, JavaScript, React, Node.js, SQL, Docker, Kubernetes",
            "certifications": "AWS Certified Developer\nGoogle Cloud Professional",
            "raw_text": "Test resume with bullets\n- Bullet point 1\n* Bullet point 2\n• Bullet point 3",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify word count
        self.assertGreater(profile.resume_statistics.word_count, 0)
        
        # Verify bullet count
        self.assertGreater(profile.resume_statistics.bullet_count, 0)
        
        # Verify skills count
        self.assertEqual(profile.resume_statistics.skills_count, 7)
        
        # Verify certification count
        self.assertGreater(profile.resume_statistics.certification_count, 0)
    
    def test_schema_integrity(self):
        """Test that all required keys exist in profile."""
        parsed_sections = {
            "summary": "Test summary",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "Test",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify all top-level fields exist
        self.assertTrue(hasattr(profile, "metadata"))
        self.assertTrue(hasattr(profile, "personal"))
        self.assertTrue(hasattr(profile, "summary"))
        self.assertTrue(hasattr(profile, "education"))
        self.assertTrue(hasattr(profile, "experience"))
        self.assertTrue(hasattr(profile, "projects"))
        self.assertTrue(hasattr(profile, "skills"))
        self.assertTrue(hasattr(profile, "certifications"))
        self.assertTrue(hasattr(profile, "achievements"))
        self.assertTrue(hasattr(profile, "languages_spoken"))
        self.assertTrue(hasattr(profile, "resume_statistics"))
        
        # Verify metadata fields exist
        self.assertTrue(hasattr(profile.metadata, "profile_schema_version"))
        self.assertTrue(hasattr(profile.metadata, "parser_version"))
        self.assertTrue(hasattr(profile.metadata, "parser_confidence"))
        self.assertTrue(hasattr(profile.metadata, "created_at"))
        self.assertTrue(hasattr(profile.metadata, "word_count"))
        self.assertTrue(hasattr(profile.metadata, "page_count"))
        self.assertTrue(hasattr(profile.metadata, "language"))
        
        # Verify personal fields exist
        self.assertTrue(hasattr(profile.personal, "name"))
        self.assertTrue(hasattr(profile.personal, "email"))
        self.assertTrue(hasattr(profile.personal, "phone"))
        self.assertTrue(hasattr(profile.personal, "location"))
        self.assertTrue(hasattr(profile.personal, "linkedin"))
        self.assertTrue(hasattr(profile.personal, "github"))
        self.assertTrue(hasattr(profile.personal, "portfolio"))
        
        # Verify statistics fields exist
        self.assertTrue(hasattr(profile.resume_statistics, "word_count"))
        self.assertTrue(hasattr(profile.resume_statistics, "bullet_count"))
        self.assertTrue(hasattr(profile.resume_statistics, "experience_count"))
        self.assertTrue(hasattr(profile.resume_statistics, "project_count"))
        self.assertTrue(hasattr(profile.resume_statistics, "certification_count"))
        self.assertTrue(hasattr(profile.resume_statistics, "skills_count"))
    
    def test_missing_fields_remain_empty(self):
        """Test that missing fields remain empty, not None."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify string fields are empty strings, not None
        self.assertEqual(profile.summary, "")
        self.assertEqual(profile.education, "")
        self.assertEqual(profile.experience, "")
        self.assertEqual(profile.projects, "")
        self.assertEqual(profile.skills, "")
        self.assertEqual(profile.certifications, "")
        self.assertEqual(profile.achievements, "")
        self.assertEqual(profile.languages_spoken, "")
        
        # Verify personal info fields are empty strings, not None
        self.assertEqual(profile.personal.name, "")
        self.assertEqual(profile.personal.email, "")
        self.assertEqual(profile.personal.phone, "")
        self.assertEqual(profile.personal.location, "")
        self.assertEqual(profile.personal.linkedin, "")
        self.assertEqual(profile.personal.github, "")
        self.assertEqual(profile.personal.portfolio, "")
    
    def test_email_extraction_variations(self):
        """Test email extraction with various formats."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "Contact: user.name@example.com or user+tag@domain.org",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Should extract at least one email
        self.assertTrue(
            profile.personal.email.endswith("@example.com") or 
            profile.personal.email.endswith("@domain.org")
        )
    
    def test_phone_extraction(self):
        """Test phone number extraction."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "Phone: +1-555-123-4567 or (555) 987-6543",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Should extract a phone number
        self.assertTrue(
            "555" in profile.personal.phone
        )
    
    def test_url_extraction(self):
        """Test URL extraction for LinkedIn, GitHub, and portfolio."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "LinkedIn: linkedin.com/in/username\nGitHub: github.com/devuser\nPortfolio: myportfolio.dev",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Verify LinkedIn
        self.assertIn("linkedin.com", profile.personal.linkedin)
        
        # Verify GitHub
        self.assertIn("github.com", profile.personal.github)
        
        # Verify portfolio
        self.assertIn("portfolio", profile.personal.portfolio.lower())
    
    def test_no_data_fabrication(self):
        """Test that builder does not fabricate missing information."""
        parsed_sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": "John Doe",
        }
        
        profile = self.builder.build(parsed_sections)
        
        # Should not fabricate email
        if not profile.personal.email:
            self.assertEqual(profile.personal.email, "")
        
        # Should not fabricate phone
        if not profile.personal.phone:
            self.assertEqual(profile.personal.phone, "")
        
        # Should not fabricate LinkedIn
        if not profile.personal.linkedin:
            self.assertEqual(profile.personal.linkedin, "")
        
        # Should not fabricate GitHub
        if not profile.personal.github:
            self.assertEqual(profile.personal.github, "")


if __name__ == "__main__":
    unittest.main()
