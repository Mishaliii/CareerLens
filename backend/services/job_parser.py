"""
Job description parsing service for CareerLens AI.

Extracts structured job requirements:
- Job title and employer name
- Required vs. preferred technical skills (using SkillOntology)
- Minimum experience years
- Required vs. preferred qualifications/degrees
- Responsibilities and explicit requirements

Features:
- Deterministic regex and ontology-guided baseline extraction.
- Optional Gemini structured extraction with strict schema validation and fallback.
- Never invents requirements not present in the input.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.models.schemas import ParsedJobDescription
from backend.services.gemini import GeminiService
from backend.services.skill_ontology import SkillOntology

logger = logging.getLogger(__name__)

# Section header regex patterns
_REQUIRED_SECTION_PATTERNS = [
    r"(?:^|\n)\s*(?:###?\s*)?(?:minimum\s+qualifications|basic\s+qualifications|requirements|required\s+skills|must\s+haves?|what\s+you(?:'ll|\s+will)?\s+need|who\s+you\s+are|what\s+we(?:'re|\s+are)?\s+looking\s+for|what\s+we\s+expect(?:\s+of\s+you)?|what\s+you\s+bring)[:\s]*\n",
]

_PREFERRED_SECTION_PATTERNS = [
    r"(?:^|\n)\s*(?:###?\s*)?(?:preferred\s+qualifications|preferred\s+skills|nice\s+to\s+haves?|nice\s+perks(?:\s+to\s+have)?|bonus\s+points?|bonus\s+qualifications|plus|desired\s+skills|good\s+to\s+have|what\s+gives\s+you\s+an\s+edge)[:\s]*\n",
]

_RESPONSIBILITIES_SECTION_PATTERNS = [
    r"(?:^|\n)\s*(?:###?\s*)?(?:responsibilities|duties|what\s+you(?:'ll|\s+will)?\s+do|the\s+role|role\s+overview|key\s+responsibilities|day\s+to\s+day)[:\s]*\n",
]

_ABOUT_SECTION_PATTERNS = [
    r"(?:^|\n)\s*(?:###?\s*)?(?:about\s+(?:the\s+)?company|about\s+us|who\s+we\s+are|company\s+overview)[:\s]*\n",
]

_TITLE_PREFIX_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:job\s+)?(?:title|role|position)\s*:\s*([A-Za-z0-9\s/&,.-]+)",
    re.IGNORECASE,
)

_COMPANY_PREFIX_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:company|employer|organization)\s*:\s*([A-Za-z0-9\s/&,.-]+)",
    re.IGNORECASE,
)

_COMMON_TITLE_KEYWORDS = [
    "software engineer", "backend engineer", "frontend engineer", "full stack engineer",
    "fullstack developer", "python developer", "data scientist", "data engineer",
    "devops engineer", "machine learning engineer", "ml engineer", "cloud architect",
    "systems engineer", "qa engineer", "security engineer", "mobile developer",
    "ios developer", "android developer", "product manager", "scrum master",
    "technical lead", "lead engineer", "senior engineer", "junior developer",
    "intern", "software developer", "web developer",
]


class JobParserService:
    """
    Service for parsing raw job descriptions into structured requirements.
    """

    def __init__(
        self,
        skill_ontology: Optional[SkillOntology] = None,
        gemini_service: Optional[GeminiService] = None,
    ):
        self.skill_ontology = skill_ontology or SkillOntology()
        self.gemini_service = gemini_service or GeminiService()
        logger.info("JobParserService initialized")

    def parse(
        self,
        raw_text: str,
        job_title_hint: Optional[str] = None,
        company_hint: Optional[str] = None,
    ) -> ParsedJobDescription:
        """
        Parse raw job description text into a validated ParsedJobDescription instance.

        Workflow:
        1. If GeminiService is configured, attempt structured extraction.
        2. Validate output schema; if invalid or failed, fall back to deterministic parsing.
        3. If Gemini is not configured, execute deterministic baseline directly.
        """
        if not raw_text or not raw_text.strip():
            raise ValueError("Job description text cannot be empty")

        clean_text = raw_text.strip()

        if self.gemini_service and self.gemini_service.is_configured:
            try:
                gemini_parsed = self._call_gemini_parsing(
                    clean_text,
                    job_title_hint=job_title_hint,
                    company_hint=company_hint,
                )
                if gemini_parsed:
                    return gemini_parsed
            except Exception as e:
                logger.warning("Gemini job parsing failed: %s; falling back to deterministic", type(e).__name__)

        # Deterministic parsing baseline
        return self._deterministic_parse(
            clean_text,
            job_title_hint=job_title_hint,
            company_hint=company_hint,
        )

    def _deterministic_parse(
        self,
        text: str,
        job_title_hint: Optional[str] = None,
        company_hint: Optional[str] = None,
    ) -> ParsedJobDescription:
        """
        Deterministic regex and ontology-guided parsing of job description text.
        """
        # 1. Title and company extraction
        title = job_title_hint or self._extract_title(text)
        company = company_hint or self._extract_company(text)

        # 2. Section segmentation
        sections = self._split_sections(text)
        required_text = sections.get("required", "")
        preferred_text = sections.get("preferred", "")
        resp_text = sections.get("responsibilities", "")

        # 3. Experience years extraction
        exp_years = self._extract_experience_years(text)

        # 4. Technical skills extraction
        req_skills, pref_skills = self._extract_skills(
            full_text=text,
            required_text=required_text,
            preferred_text=preferred_text,
        )

        # 5. Qualifications extraction
        req_quals, pref_quals = self._extract_qualifications(
            full_text=text,
            required_text=required_text,
            preferred_text=preferred_text,
        )

        # 6. Responsibilities extraction
        responsibilities = self._extract_bullet_points(resp_text) if resp_text else []

        # 7. Other explicit requirements (e.g. work mode, location, clearance)
        other_reqs = self._extract_other_requirements(text)

        return ParsedJobDescription(
            job_title=title,
            company=company,
            required_skills=sorted(list(req_skills)),
            preferred_skills=sorted(list(pref_skills)),
            required_qualifications=req_quals,
            preferred_qualifications=pref_quals,
            minimum_experience_years=exp_years,
            responsibilities=responsibilities,
            other_requirements=other_reqs,
            raw_text=text,
            extraction_source="deterministic",
        )

    def _split_sections(self, text: str) -> Dict[str, str]:
        """Split job description into logical sections."""
        result: Dict[str, str] = {
            "required": "",
            "preferred": "",
            "responsibilities": "",
            "about": "",
            "general": "",
        }

        # Find section offsets
        matches: List[Tuple[int, str, int]] = []  # (start, section_type, header_length)

        for pat in _REQUIRED_SECTION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                matches.append((m.start(), "required", m.end() - m.start()))

        for pat in _PREFERRED_SECTION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                matches.append((m.start(), "preferred", m.end() - m.start()))

        for pat in _RESPONSIBILITIES_SECTION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                matches.append((m.start(), "responsibilities", m.end() - m.start()))

        for pat in _ABOUT_SECTION_PATTERNS:
            for m in re.finditer(pat, text, re.IGNORECASE):
                matches.append((m.start(), "about", m.end() - m.start()))

        if not matches:
            result["general"] = text
            return result

        matches.sort(key=lambda x: x[0])

        # Slice text between section headers
        for i, (start, sec_type, header_len) in enumerate(matches):
            content_start = start + header_len
            content_end = matches[i + 1][0] if i + 1 < len(matches) else len(text)
            content = text[content_start:content_end].strip()
            if sec_type in result and result[sec_type]:
                result[sec_type] += "\n" + content
            else:
                result[sec_type] = content

        return result

    def _extract_title(self, text: str) -> Optional[str]:
        """Extract job title using explicit prefixes or common role keywords."""
        # 1. Prefix match
        prefix_match = _TITLE_PREFIX_PATTERN.search(text)
        if prefix_match:
            candidate = prefix_match.group(1).strip()
            # Clean trailing punctuation
            candidate = candidate.split("\n")[0].strip(" -:;,")
            if 3 <= len(candidate) <= 80:
                return candidate

        # 2. Check first 3 lines for standard role keywords
        lines = [line.strip() for line in text.split("\n") if line.strip()][:3]
        for line in lines:
            line_lower = line.lower()
            for kw in _COMMON_TITLE_KEYWORDS:
                if kw in line_lower and len(line) <= 80:
                    return line.strip(" -:;,#*")

        return None

    def _extract_company(self, text: str) -> Optional[str]:
        """Extract company name using explicit prefixes or intro patterns."""
        # 1. Prefix match
        prefix_match = _COMPANY_PREFIX_PATTERN.search(text)
        if prefix_match:
            candidate = prefix_match.group(1).strip()
            candidate = candidate.split("\n")[0].strip(" -:;,")
            if 2 <= len(candidate) <= 60:
                return candidate

        # 2. About Company regex
        about_match = re.search(
            r"About\s+([A-Z][A-Za-z0-9\s&.,'-]{2,40})(?:\s*[:\n]|\s+is|\s+we)",
            text,
        )
        if about_match:
            return about_match.group(1).strip(" -:;,")

        # 3. 'At <Company>, we' pattern
        at_match = re.search(
            r"(?:At|Join)\s+([A-Z][A-Za-z0-9\s&.,'-]{2,30}),\s+we",
            text,
        )
        if at_match:
            return at_match.group(1).strip(" -:;,")

        return None

    def _extract_experience_years(self, text: str) -> Optional[float]:
        """Extract minimum years of experience required from job description."""
        patterns = [
            # e.g. "3+ years of experience", "3-5 years of experience", "3 years experience", "2-4 years of professional experience"
            r"(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)(?:\s+of)?(?:\s+[a-zA-Z-]+){0,3}\s+experience",
            # e.g. "experience: 3+ years", "experience of 3 years"
            r"experience\s*(?:of|:)?\s*(\d+(?:\.\d+)?)\s*(?:\+|-\s*\d+)?\s*(?:years?|yrs?)",
            # e.g. "minimum 3 years", "at least 5 years"
            r"(?:minimum|at\s+least)\s+(\d+(?:\.\d+)?)\s*(?:years?|yrs?)",
            # e.g. "3+ years in Python", "3+ years with Python"
            r"(\d+(?:\.\d+)?)\+?\s*(?:years?|yrs?)\s+(?:in|with)\s+[A-Za-z0-9]",
        ]
        candidates: List[float] = []
        for pat in patterns:
            for match in re.finditer(pat, text, re.IGNORECASE):
                try:
                    val = float(match.group(1))
                    if 0.5 <= val <= 25.0:
                        candidates.append(val)
                except (ValueError, IndexError):
                    pass

        if candidates:
            # Return lowest reasonable positive threshold stated
            return min(candidates)
        return None

    def _extract_skills(
        self,
        full_text: str,
        required_text: str,
        preferred_text: str,
    ) -> Tuple[Set[str], Set[str]]:
        """
        Identify technical skills using SkillOntology canonical definitions.
        Partition into required vs. preferred based on section location.
        """
        req_skills: Set[str] = set()
        pref_skills: Set[str] = set()

        # Build lookup set of all canonical skills and aliases from ontology
        all_canonical = list(SkillOntology.TAXONOMY.keys())

        # Check preferred section first
        if preferred_text:
            for canonical in all_canonical:
                if self._skill_in_text(canonical, preferred_text):
                    pref_skills.add(canonical)

        # Check required section or full text
        scan_target = required_text if required_text else full_text
        for canonical in all_canonical:
            if self._skill_in_text(canonical, scan_target):
                # If already identified in preferred, don't duplicate into required
                if canonical not in pref_skills:
                    req_skills.add(canonical)

        return req_skills, pref_skills

    def _skill_in_text(self, canonical: str, text: str) -> bool:
        """
        Check whether canonical skill or its aliases appear in text with word boundary checks.
        """
        info = SkillOntology.TAXONOMY.get(canonical, {})
        aliases = [canonical.lower()] + [a.lower() for a in info.get("aliases", [])]

        text_lower = text.lower()
        for alias in aliases:
            # Use regex word boundaries for clean matching
            escaped = re.escape(alias)
            # Handle special symbols like C++, C#, .NET
            if not alias[0].isalnum():
                pattern = rf"(?:^|\s){escaped}(?:\s|[,;.!?:)]|$)"
            elif not alias[-1].isalnum():
                pattern = rf"\b{escaped}(?:\s|[,;.!?:)]|$)"
            else:
                pattern = rf"\b{escaped}\b"

            if re.search(pattern, text_lower):
                return True

        return False

    def _extract_qualifications(
        self,
        full_text: str,
        required_text: str,
        preferred_text: str,
    ) -> Tuple[List[str], List[str]]:
        """
        Extract required and preferred educational qualifications and degrees.
        """
        degree_patterns = [
            (r"(?:\b(?:bachelor(?:'s)?|b\.tech|b\.eng|bs(?=\s+(?:degree|in)))\b|\bb\.s\.|\bb\.e\.)(?:\s+degree)?(?:\s+in\s+([A-Za-z\s,]+))?", "Bachelor's degree"),
            (r"(?:\b(?:master(?:'s)?|m\.tech|m\.eng|ms(?=\s+(?:degree|in)))\b|\bm\.s\.|\bm\.e\.)(?:\s+degree)?(?:\s+in\s+([A-Za-z\s,]+))?", "Master's degree"),
            (r"(?:\b(?:doctorate|phd(?=\s+(?:degree|in)))\b|\bph\.?d\.)(?:\s+degree)?(?:\s+in\s+([A-Za-z\s,]+))?", "PhD degree"),
        ]

        def find_quals(src: str) -> List[str]:
            quals = []
            for pat, label in degree_patterns:
                match = re.search(pat, src, re.IGNORECASE)
                if match:
                    field = match.group(1).strip() if (match.lastindex and match.group(1)) else None
                    if field:
                        # Clean field of study
                        field_clean = field.split("\n")[0].split(".")[0].strip(" -:,;")
                        quals.append(f"{label} in {field_clean}")
                    else:
                        quals.append(label)
            return quals

        req_quals = find_quals(required_text) if required_text else find_quals(full_text)
        pref_quals = find_quals(preferred_text) if preferred_text else []

        return req_quals, pref_quals

    def _extract_bullet_points(self, text: str) -> List[str]:
        """Extract clean bullet points or lines from a section."""
        lines = [line.strip() for line in text.split("\n") if line.strip()]
        bullets: List[str] = []
        for line in lines:
            cleaned = re.sub(r"^[•\-*–—\d+.)\s]+", "", line).strip()
            if len(cleaned) >= 15:
                bullets.append(cleaned)
        return bullets[:10]  # Cap at 10 items

    def _extract_other_requirements(self, text: str) -> List[str]:
        """Extract other notable requirements such as remote, on-site, security clearance."""
        other = []
        text_lower = text.lower()
        if "security clearance" in text_lower:
            other.append("Security clearance required or eligible")
        if "remote" in text_lower and ("100% remote" in text_lower or "fully remote" in text_lower or "remote work" in text_lower):
            other.append("Remote work option")
        elif "hybrid" in text_lower:
            other.append("Hybrid work schedule")
        elif "on-site" in text_lower or "onsite" in text_lower:
            other.append("On-site presence required")

        return other

    def _call_gemini_parsing(
        self,
        text: str,
        job_title_hint: Optional[str] = None,
        company_hint: Optional[str] = None,
    ) -> Optional[ParsedJobDescription]:
        """
        Use Gemini 2.0 Flash to extract structured JD requirements and validate schema.
        """
        prompt = f"""You are a precise job description analysis engine.
Extract the structured requirements from the following job description.

Rules:
1. Extract only requirements explicitly stated in the text. DO NOT INVENT any skills, qualifications, or requirements.
2. Distinguish required/mandatory technical skills from preferred/nice-to-have technical skills.
3. If an experience requirement (e.g. 3 years) is stated, extract minimum_experience_years as a number.
4. Output valid JSON adhering strictly to the schema below.

Job Description:
\"\"\"
{text}
\"\"\"

Output format:
```json
{{
  "job_title": string or null,
  "company": string or null,
  "required_skills": [string],
  "preferred_skills": [string],
  "required_qualifications": [string],
  "preferred_qualifications": [string],
  "minimum_experience_years": float or null,
  "responsibilities": [string],
  "other_requirements": [string]
}}
```
"""
        response = self.gemini_service.model.generate_content(prompt)
        raw_output = response.text.strip()

        # Parse JSON from response
        json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_output)
        json_str = json_match.group(1) if json_match else raw_output

        data = json.loads(json_str)

        # Apply overrides if provided
        if job_title_hint:
            data["job_title"] = job_title_hint
        if company_hint:
            data["company"] = company_hint

        data["raw_text"] = text
        data["extraction_source"] = "gemini"

        return ParsedJobDescription.model_validate(data)


_GLOBAL_JOB_PARSER: Optional[JobParserService] = None


def get_job_parser() -> JobParserService:
    """Singleton provider for JobParserService."""
    global _GLOBAL_JOB_PARSER
    if _GLOBAL_JOB_PARSER is None:
        _GLOBAL_JOB_PARSER = JobParserService()
    return _GLOBAL_JOB_PARSER


def reset_job_parser() -> None:
    """Reset singleton instance (useful for testing)."""
    global _GLOBAL_JOB_PARSER
    _GLOBAL_JOB_PARSER = None
