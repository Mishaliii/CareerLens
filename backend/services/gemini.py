import concurrent.futures
import json
import logging
import os
import re
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv

import google.generativeai as genai
from pydantic import ValidationError

from backend.models.schemas import (
    StructuredCV,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    CertificationItem,
)

load_dotenv()

logger = logging.getLogger(__name__)

GEMINI_MODEL = "gemini-2.0-flash"
API_TIMEOUT_SECONDS = 30

_JOB_TITLE_KEYWORDS = {
    "developer", "engineer", "scientist", "manager", "architect", "lead",
    "specialist", "analyst", "consultant", "designer", "administrator", "intern",
    "director", "officer", "coordinator", "associate", "student", "candidate",
    "executive", "freelancer", "programmer", "tester", "researcher",
    "full stack", "fullstack", "front end", "frontend", "back end", "backend",
    "software", "web", "cloud", "devops", "data", "mobile", "ai", "ml",
    "machine learning", "product", "qa", "system", "systems", "security",
}

_NON_NAME_SECTION_KEYWORDS = {
    "resume", "cv", "curriculum", "vitae", "summary", "experience", "education",
    "skills", "projects", "certifications", "contact", "profile", "about",
    "objective", "work", "employment", "history", "languages", "portfolio",
    "technical", "personal", "overview", "background", "qualifications",
}


class GeminiService:
    """
    Service for AI-powered career intelligence operations using Gemini 2.0 Flash.
    Provides structured CV extraction, strict schema validation, and defensive fallbacks.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self._is_configured = False

        # Verify whether API key is valid (not None, empty, or placeholder)
        if self.api_key and not self.api_key.startswith("your_") and len(self.api_key.strip()) > 10:
            try:
                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(GEMINI_MODEL)
                self._is_configured = True
                logger.info("GeminiService initialized with configured model: %s", GEMINI_MODEL)
            except Exception as e:
                logger.warning("Failed to configure Google Generative AI: %s", type(e).__name__)
                self._is_configured = False
        else:
            logger.warning("GEMINI_API_KEY is not configured or is a placeholder. Fallback extraction will be used.")

    @property
    def is_configured(self) -> bool:
        return self._is_configured

    def extract_structured_cv(self, raw_text: str) -> StructuredCV:
        """
        Extract structured CV information from raw resume text.
        Attempts Gemini extraction first; if unconfigured or on failure, uses deterministic fallback.

        Args:
            raw_text: Raw resume text.

        Returns:
            StructuredCV validated instance.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty raw text provided for structured extraction")
            return StructuredCV(extraction_source="fallback")

        if self._is_configured:
            try:
                structured_data = self._call_gemini_structured_extraction(raw_text)
                if structured_data:
                    return structured_data
            except Exception as e:
                logger.warning("Gemini extraction failed (%s). Falling back to rule-based extraction.", type(e).__name__)

        # Fallback to rule-based extraction
        return self._fallback_rule_based_extraction(raw_text)

    def _call_gemini_structured_extraction(self, raw_text: str) -> Optional[StructuredCV]:
        """
        Invoke Gemini API to extract structured CV data with enforced timeout and validate with Pydantic.
        """
        prompt = self._build_extraction_prompt(raw_text)
        logger.info("Sending structured extraction request to Gemini (%d chars of text)", len(raw_text))

        try:
            # Enforce real 30-second timeout via ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    self.model.generate_content,
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.1,
                        candidate_count=1,
                    ),
                )
                try:
                    response = future.result(timeout=API_TIMEOUT_SECONDS)
                except concurrent.futures.TimeoutError as te:
                    logger.error("Gemini API call timed out after %d seconds", API_TIMEOUT_SECONDS)
                    raise TimeoutError(f"Gemini API call timed out after {API_TIMEOUT_SECONDS} seconds") from te

            response_text = response.text if response and hasattr(response, "text") and response.text else ""
            if not response_text or not response_text.strip():
                logger.warning("Gemini returned empty or blank response")
                raise ValueError("Gemini returned empty response")

            cleaned_json_str = self._clean_llm_json_response(response_text)
            parsed_dict = json.loads(cleaned_json_str)

            # Validate with Pydantic and tag source as gemini
            validated_cv = self.validate_and_sanitize_cv_dict(parsed_dict, source="gemini")
            logger.info(
                "Gemini structured extraction succeeded: edu=%d, exp=%d, proj=%d, skills=%d",
                len(validated_cv.education),
                len(validated_cv.experience),
                len(validated_cv.projects),
                len(validated_cv.skills),
            )
            return validated_cv

        except json.JSONDecodeError:
            logger.error("Gemini response was not valid JSON (decode failure)")
            raise
        except ValidationError as ve:
            error_fields = [str(err.get("loc", ["unknown"])[-1]) for err in ve.errors()]
            logger.error("Gemini JSON failed Pydantic schema validation on fields: %s", ", ".join(error_fields))
            raise
        except TimeoutError:
            raise
        except Exception as e:
            logger.error("Gemini API call failed with exception: %s", type(e).__name__)
            raise

    def _build_extraction_prompt(self, raw_text: str) -> str:
        """Construct the extraction prompt enforcing strict JSON and no fabrication."""
        schema_format = {
            "full_name": "Full name of candidate",
            "email": "Email address",
            "phone": "Phone number",
            "location": "City, State / Country",
            "linkedin_url": "Full LinkedIn URL",
            "github_url": "Full GitHub URL",
            "portfolio_url": "Portfolio or personal website URL",
            "summary": "Professional summary or objective statement",
            "education": [
                {
                    "institution": "University or School name",
                    "degree": "Degree name (e.g. B.Tech, M.S., B.Sc)",
                    "field": "Field of study / Major",
                    "start_year": 2018,
                    "end_year": 2022,
                    "gpa": 8.5
                }
            ],
            "experience": [
                {
                    "company": "Company or Organization name",
                    "title": "Job title",
                    "start_date": "Start date (e.g. 'Jan 2020' or '2020')",
                    "end_date": "End date (e.g. 'Present' or 'Dec 2023')",
                    "description": "Responsibilities and key achievements"
                }
            ],
            "projects": [
                {
                    "name": "Project name",
                    "description": "Overview of project",
                    "technologies": ["Python", "FastAPI"],
                    "link": "URL if present or null"
                }
            ],
            "skills": ["List of explicitly mentioned technical and domain skills"],
            "certifications": [
                {
                    "name": "Certification name",
                    "issuer": "Issuing organization",
                    "date_obtained": "Date or year obtained",
                    "expiry_date": "Expiry date or null"
                }
            ]
        }

        prompt = (
            f"You are the CareerLens AI Resume Extraction Engine.\n"
            f"Extract all career and candidate information from the following resume text into exact JSON adhering to this schema:\n\n"
            f"{json.dumps(schema_format, indent=2)}\n\n"
            f"CRITICAL RULES:\n"
            f"1. Never fabricate skills, roles, degrees, or achievements not present in the resume.\n"
            f"2. For missing information, use empty string \"\", empty list [], or null.\n"
            f"3. Do not infer skills that are not explicitly stated.\n"
            f"4. If education start_year or end_year is unknown, use null or omit.\n"
            f"5. Return JSON only. No markdown. No explanation. No code blocks.\n\n"
            f"RESUME TEXT:\n{raw_text}\n\n"
            f"Return JSON only. No markdown. No explanation. No code blocks."
        )
        return prompt

    def _clean_llm_json_response(self, response_text: str) -> str:
        """Strip markdown fences, leading/trailing commentary, and whitespace."""
        text = response_text.strip()
        # Remove markdown code blocks if present
        if text.startswith("```"):
            text = re.sub(r'^```(?:json)?\s*', '', text, flags=re.IGNORECASE)
            text = re.sub(r'\s*```$', '', text)
        return text.strip()

    def validate_and_sanitize_cv_dict(self, data: Dict[str, Any], source: str = "fallback") -> StructuredCV:
        """
        Validate and sanitize arbitrary dictionary data into a verified StructuredCV instance.
        Coerces numeric fields and cleans list items defensively.
        Preserves missing fields as None/empty without fabricating dates.
        """
        if not isinstance(data, dict):
            StructuredCV.model_validate(data)


        # Sanitize education items
        raw_education = data.get("education", [])
        clean_education: List[EducationItem] = []
        if isinstance(raw_education, list):
            for item in raw_education:
                if isinstance(item, dict):
                    try:
                        # Extract start_year as integer if present; otherwise keep None (never fabricate)
                        start_year = item.get("start_year")
                        if start_year is not None:
                            try:
                                start_year = int(str(start_year).strip()[:4])
                            except (ValueError, TypeError):
                                start_year = None
                        else:
                            start_year = None

                        end_year = item.get("end_year")
                        if end_year is not None:
                            try:
                                end_year = int(str(end_year).strip()[:4])
                            except (ValueError, TypeError):
                                end_year = None

                        gpa_val = item.get("gpa")
                        if gpa_val is not None:
                            try:
                                gpa_val = float(gpa_val)
                            except (ValueError, TypeError):
                                gpa_val = None

                        clean_education.append(
                            EducationItem(
                                institution=str(item.get("institution", "")).strip(),
                                degree=str(item.get("degree", "")).strip(),
                                field=str(item.get("field", "")).strip(),
                                start_year=start_year,
                                end_year=end_year,
                                gpa=gpa_val,
                            )
                        )
                    except Exception as err:
                        logger.debug("Skipping unparseable education item: %s", type(err).__name__)

        # Sanitize experience items
        raw_experience = data.get("experience", [])
        clean_experience: List[ExperienceItem] = []
        if isinstance(raw_experience, list):
            for item in raw_experience:
                if isinstance(item, dict) and (item.get("company") or item.get("title")):
                    clean_experience.append(
                        ExperienceItem(
                            company=str(item.get("company", "")).strip(),
                            title=str(item.get("title", "")).strip(),
                            start_date=str(item.get("start_date", "")).strip(),
                            end_date=str(item.get("end_date", "")).strip() if item.get("end_date") else None,
                            description=str(item.get("description", "")).strip() if item.get("description") else None,
                        )
                    )

        # Sanitize projects
        raw_projects = data.get("projects", [])
        clean_projects: List[ProjectItem] = []
        if isinstance(raw_projects, list):
            for item in raw_projects:
                if isinstance(item, dict) and item.get("name"):
                    techs = item.get("technologies", [])
                    if isinstance(techs, str):
                        techs = [t.strip() for t in techs.split(",") if t.strip()]
                    elif not isinstance(techs, list):
                        techs = []
                    clean_projects.append(
                        ProjectItem(
                            name=str(item.get("name", "")).strip(),
                            description=str(item.get("description", "")).strip() if item.get("description") else None,
                            technologies=[str(t).strip() for t in techs if str(t).strip()],
                            link=str(item.get("link", "")).strip() if item.get("link") else None,
                        )
                    )

        # Sanitize skills
        raw_skills = data.get("skills", [])
        clean_skills: List[str] = []
        if isinstance(raw_skills, list):
            for s in raw_skills:
                cleaned = str(s).strip()
                if cleaned and cleaned not in clean_skills:
                    clean_skills.append(cleaned)
        elif isinstance(raw_skills, str):
            clean_skills = [s.strip() for s in re.split(r'[,|\n|;]', raw_skills) if s.strip()]

        # Sanitize certifications
        raw_certs = data.get("certifications", [])
        clean_certs: List[CertificationItem] = []
        if isinstance(raw_certs, list):
            for item in raw_certs:
                if isinstance(item, dict) and item.get("name"):
                    clean_certs.append(
                        CertificationItem(
                            name=str(item.get("name", "")).strip(),
                            issuer=str(item.get("issuer", "")).strip(),
                            date_obtained=str(item.get("date_obtained", "")).strip(),
                            expiry_date=str(item.get("expiry_date", "")).strip() if item.get("expiry_date") else None,
                        )
                    )

        extraction_source = str(data.get("extraction_source", source)).strip()
        if extraction_source not in ("gemini", "fallback"):
            extraction_source = source

        return StructuredCV(
            full_name=str(data.get("full_name", "")).strip(),
            email=str(data.get("email", "")).strip(),
            phone=str(data.get("phone", "")).strip(),
            location=str(data.get("location", "")).strip(),
            linkedin_url=str(data.get("linkedin_url", "")).strip(),
            github_url=str(data.get("github_url", "")).strip(),
            portfolio_url=str(data.get("portfolio_url", "")).strip(),
            summary=str(data.get("summary", "")).strip(),
            education=clean_education,
            experience=clean_experience,
            projects=clean_projects,
            skills=clean_skills,
            certifications=clean_certs,
            extraction_source=extraction_source,
        )

    def _fallback_rule_based_extraction(self, raw_text: str) -> StructuredCV:
        """
        Deterministic rule-based fallback extraction when Gemini is unavailable.
        Extracts contact info, conservative name heuristic, sections, and itemized bullets.
        Never mistakes job titles, employers, or section headers for a candidate name.
        """
        logger.info("Executing rule-based fallback CV extraction")
        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]

        # Conservative name extraction from top lines
        full_name = ""
        for line in lines[:5]:
            # Skip if contains email, phone, url, or common headers
            if "@" in line or "http" in line or "linkedin" in line or "github" in line:
                continue

            lower_line = line.lower()
            # Skip if contains any section keywords
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lower_line) for kw in _NON_NAME_SECTION_KEYWORDS):
                continue
            # Skip if contains any job title keywords or phrases
            if any(re.search(r'\b' + re.escape(kw) + r'\b', lower_line) for kw in _JOB_TITLE_KEYWORDS):
                continue

            # Plausible candidate name: 2-4 words, each starting with uppercase letter
            words = line.split()
            if 2 <= len(words) <= 4:
                if all(re.match(r'^[A-Z][a-zA-Z\.\'-]*$', w) for w in words):
                    full_name = line
                    break

        # Contact info
        email = ""
        email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', raw_text)
        if email_match:
            email = email_match.group()

        phone = ""
        phone_match = re.search(r'\b(?:\+?(\d{1,3})?[-. ]?)?\(?(\d{3})\)?[-. ]?(\d{3})[-. ]?(\d{4})\b', raw_text)
        if phone_match:
            phone = phone_match.group()

        linkedin_url = ""
        li_match = re.search(r'linkedin\.com/in/[A-Za-z0-9-]+', raw_text)
        if li_match:
            linkedin_url = f"https://{li_match.group()}"

        github_url = ""
        gh_match = re.search(r'github\.com/[A-Za-z0-9-]+', raw_text)
        if gh_match:
            github_url = f"https://{gh_match.group()}"

        portfolio_url = ""
        for match in re.findall(r'\b(?:https?://)?(?:www\.)?[A-Za-z0-9-]+\.(?:com|io|net|org|dev)(?:/[^\s]*)?\b', raw_text):
            if "linkedin" not in match.lower() and "github" not in match.lower():
                portfolio_url = match if match.startswith("http") else f"https://{match}"
                break

        # Fallback skills parsing
        skills: List[str] = []
        skills_match = re.search(r'(?:Skills|Technical Skills|Key Skills)[:\s]+(.*?)(?=\n[A-Z][a-z]+|\Z)', raw_text, re.DOTALL | re.IGNORECASE)
        if skills_match:
            raw_skills_block = skills_match.group(1).strip()
            tokens = [s.strip() for s in re.split(r'[,|\n|;]', raw_skills_block) if s.strip()]
            skills = [t for t in tokens if len(t) < 40][:30]

        return StructuredCV(
            full_name=full_name,
            email=email,
            phone=phone,
            location="",
            linkedin_url=linkedin_url,
            github_url=github_url,
            portfolio_url=portfolio_url,
            summary="",
            education=[],
            experience=[],
            projects=[],
            skills=skills,
            certifications=[],
            extraction_source="fallback",
        )


_gemini_service_instance: Optional[GeminiService] = None


def get_gemini_service() -> GeminiService:
    """Retrieve singleton instance of GeminiService."""
    global _gemini_service_instance
    if _gemini_service_instance is None:
        _gemini_service_instance = GeminiService()
    return _gemini_service_instance
