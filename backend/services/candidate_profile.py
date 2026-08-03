import logging
import re
from typing import Dict
from datetime import datetime
from backend.models.schemas import (
    CandidateProfile,
    CandidateProfileMetadata,
    CandidateProfilePersonal,
    CandidateProfileStatistics,
)

logger = logging.getLogger(__name__)

# Parser version - easily changeable later
PARSER_VERSION = "1.0.0"
PROFILE_VERSION = "1.0.0"


class CandidateProfileBuilder:
    """Builds canonical Candidate Profile from CVParser output."""
    
    def __init__(self):
        logger.info("Candidate Profile Builder initialized")
    
    def build(self, parsed_sections: Dict[str, str]) -> CandidateProfile:
        """
        Build a Candidate Profile from parsed CV sections.
        
        Args:
            parsed_sections: Output from CVParser.extract_sections()
            
        Returns:
            CandidateProfile with all required fields populated
        """
        logger.info("Profile building started")
        
        raw_text = parsed_sections.get("raw_text", "")
        
        # Build metadata
        metadata = self._build_metadata(raw_text)
        
        # Build personal information
        personal = self._extract_personal_info(raw_text)
        
        # Build statistics
        statistics = self._calculate_statistics(parsed_sections)
        
        # Create profile
        profile = CandidateProfile(
            metadata=metadata,
            personal=personal,
            summary=parsed_sections.get("summary", ""),
            education=parsed_sections.get("education", ""),
            experience=parsed_sections.get("experience", ""),
            projects=parsed_sections.get("projects", ""),
            skills=parsed_sections.get("skills", ""),
            certifications=parsed_sections.get("certifications", ""),
            achievements="",  # Will be populated by Document Analyzer in later sprint
            languages_spoken="",  # Will be populated by Document Analyzer in later sprint
            resume_statistics=statistics,
        )
        
        logger.info("Profile building completed")
        
        return profile
    
    def _build_metadata(self, raw_text: str) -> CandidateProfileMetadata:
        """
        Build profile metadata.
        
        Args:
            raw_text: Full resume text
            
        Returns:
            CandidateProfileMetadata
        """
        word_count = len(raw_text.split()) if raw_text else 0
        
        metadata = CandidateProfileMetadata(
            profile_schema_version=PROFILE_VERSION,
            parser_version=PARSER_VERSION,
            parser_confidence=None,  # Will be populated by Document Analyzer in Sprint 1.2.2
            created_at=datetime.utcnow(),
            word_count=word_count,
            page_count=None,  # Will be populated by Document Analyzer in Sprint 1.2.2
            language=None,  # Will be populated by Document Analyzer in Sprint 1.2.2
        )
        
        logger.info("Metadata generated: schema_version=%s, parser_version=%s, word_count=%d", PROFILE_VERSION, PARSER_VERSION, word_count)
        
        return metadata
    
    def _extract_personal_info(self, raw_text: str) -> CandidateProfilePersonal:
        """
        Extract personal information from raw text using basic regex.
        
        Note: This is basic extraction. Advanced extraction will be
        handled by Document Analyzer in Sprint 1.2.2.
        
        Args:
            raw_text: Full resume text
            
        Returns:
            CandidateProfilePersonal
        """
        personal = CandidateProfilePersonal()
        
        if not raw_text:
            logger.warning("Empty raw text provided for personal info extraction")
            return personal
        
        # Extract email
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        email_match = re.search(email_pattern, raw_text)
        if email_match:
            personal.email = email_match.group()
        
        # Extract phone (basic pattern)
        phone_pattern = r'\b(?:\+?(\d{1,3})?[-. ]?)?\(?(\d{3})\)?[-. ]?(\d{3})[-. ]?(\d{4})\b'
        phone_match = re.search(phone_pattern, raw_text)
        if phone_match:
            personal.phone = phone_match.group()
        
        # Extract LinkedIn URL
        linkedin_pattern = r'linkedin\.com/in/[A-Za-z0-9-]+'
        linkedin_match = re.search(linkedin_pattern, raw_text)
        if linkedin_match:
            personal.linkedin = f"https://{linkedin_match.group()}"
        
        # Extract GitHub URL
        github_pattern = r'github\.com/[A-Za-z0-9-]+'
        github_match = re.search(github_pattern, raw_text)
        if github_match:
            personal.github = f"https://{github_match.group()}"
        
        # Extract portfolio URL (basic pattern)
        portfolio_pattern = r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.(?:com|io|net|org|dev)(?:/[^\s]*)?\b'
        # Only match if not already matched as linkedin/github
        portfolio_matches = re.findall(portfolio_pattern, raw_text)
        for match in portfolio_matches:
            if 'linkedin' not in match.lower() and 'github' not in match.lower():
                personal.portfolio = match if match.startswith('http') else f"https://{match}"
                break
        
        
        logger.info(
            "Personal info extracted: name=%s, email=%s, phone=%s",
            personal.name if personal.name else "(none)",
            personal.email if personal.email else "(none)",
            personal.phone if personal.phone else "(none)",
        )
        
        return personal
    
    def _calculate_statistics(self, parsed_sections: Dict[str, str]) -> CandidateProfileStatistics:
        """
        Calculate resume statistics from parsed sections.
        
        Args:
            parsed_sections: Output from CVParser.extract_sections()
            
        Returns:
            CandidateProfileStatistics
        """
        raw_text = parsed_sections.get("raw_text", "")
        
        # Word count
        word_count = len(raw_text.split()) if raw_text else 0
        
        # Bullet count (lines starting with -, *, or •)
        bullet_count = 0
        for line in raw_text.split('\n'):
            stripped = line.strip()
            if stripped.startswith(('-', '*', '•', 'o')):
                bullet_count += 1
        
        # Experience count (basic heuristic - count company mentions)
        experience_text = parsed_sections.get("experience", "")
        experience_count = 0
        if experience_text:
            # Count lines that look like company entries
            for line in experience_text.split('\n'):
                if line.strip() and len(line.strip()) > 10:
                    experience_count += 1
            # Cap at reasonable number
            experience_count = min(experience_count, 20)
        
        # Project count (basic heuristic)
        projects_text = parsed_sections.get("projects", "")
        project_count = 0
        if projects_text:
            for line in projects_text.split('\n'):
                if line.strip() and len(line.strip()) > 5:
                    project_count += 1
            project_count = min(project_count, 20)
        
        # Certification count (basic heuristic)
        certifications_text = parsed_sections.get("certifications", "")
        certification_count = 0
        if certifications_text:
            for line in certifications_text.split('\n'):
                if line.strip() and len(line.strip()) > 5:
                    certification_count += 1
            certification_count = min(certification_count, 20)
        
        # Skills count (split by common delimiters)
        skills_text = parsed_sections.get("skills", "")
        skills_count = 0
        if skills_text:
            # Split by commas, newlines, or pipes
            skills = re.split(r'[,|\n|;]', skills_text)
            skills_count = len([s.strip() for s in skills if s.strip()])
            skills_count = min(skills_count, 100)
        
        statistics = CandidateProfileStatistics(
            word_count=word_count,
            bullet_count=bullet_count,
            experience_count=experience_count,
            project_count=project_count,
            certification_count=certification_count,
            skills_count=skills_count,
        )
        
        logger.info(
            "Statistics generated: words=%d, bullets=%d, exp=%d, projects=%d, certs=%d, skills=%d",
            word_count,
            bullet_count,
            experience_count,
            project_count,
            certification_count,
            skills_count,
        )
        
        return statistics
