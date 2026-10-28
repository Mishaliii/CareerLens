import copy
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

from backend.models.schemas import (
    AcademicGrade,
    DatePrecision,
    EducationItem,
    ExperienceItem,
    GradeScale,
    NormalizedDate,
    StructuredCV,
)

logger = logging.getLogger(__name__)

MONTH_NAME_TO_INT: Dict[str, int] = {
    "january": 1, "jan": 1,
    "february": 2, "feb": 2,
    "march": 3, "mar": 3,
    "april": 4, "apr": 4,
    "may": 5,
    "june": 6, "jun": 6,
    "july": 7, "jul": 7,
    "august": 8, "aug": 8,
    "september": 9, "sept": 9, "sep": 9,
    "october": 10, "oct": 10,
    "november": 11, "nov": 11,
    "december": 12, "dec": 12,
}

CURRENT_KEYWORDS = {
    "present", "current", "currently working", "currently employed",
    "ongoing", "till date", "to date", "now", "today", "continue", "continuing"
}

DATE_RANGE_PATTERN = re.compile(
    r"""
    (?:
        \b(?:19|20)\d{2}\b\s*(?:[-–—]|to)\s*(?:\b(?:19|20)\d{2}\b|present|current|now|till\s+date|ongoing)
        |
        \b[a-zA-Z]{3,9}\.?\s+(?:19|20)\d{2}\b\s*(?:[-–—]|to)\s*(?:\b[a-zA-Z]{3,9}\.?\s+(?:19|20)\d{2}\b|present|current|now)
        |
        \b\d{1,2}[\-\/\.]\d{1,2}[\-\/\.](?:19|20)\d{2}\b\s*(?:[-–—]|to)\s*(?:\b\d{1,2}[\-\/\.]\d{1,2}[\-\/\.](?:19|20)\d{2}\b|present|current)
    )
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Canonical qualification titles mapped to regex aliases
CANONICAL_DEGREE_PATTERNS: List[Tuple[str, List[str]]] = [
    (
        "Bachelor of Technology",
        [
            r"\bb\.?\s*tech\b",
            r"\bbachelor\s+of\s+technology\b",
            r"\bb\s+tech\b",
        ]
    ),
    (
        "Master of Technology",
        [
            r"\bm\.?\s*tech\b",
            r"\bmaster\s+of\s+technology\b",
            r"\bm\s+tech\b",
        ]
    ),
    (
        "Bachelor of Engineering",
        [
            r"\bb\.?\s*e\.?\b",
            r"\bbachelor\s+of\s+engineering\b",
            r"\bb\.?\s*eng\b",
        ]
    ),
    (
        "Master of Engineering",
        [
            r"\bm\.?\s*e\.?\b",
            r"\bmaster\s+of\s+engineering\b",
            r"\bm\.?\s*eng\b",
        ]
    ),
    (
        "Bachelor of Computer Applications",
        [
            r"\bb\.?\s*c\.?\s*a\.?\b",
            r"\bbachelor\s+of\s+computer\s+applications?\b",
        ]
    ),
    (
        "Master of Computer Applications",
        [
            r"\bm\.?\s*c\.?\s*a\.?\b",
            r"\bmaster\s+of\s+computer\s+applications?\b",
        ]
    ),
    (
        "Bachelor of Science",
        [
            r"\bb\.?\s*sc\.?\b",
            r"\bbachelor\s+of\s+science\b",
            r"\bbsc\b",
            r"\bb\.?\s*s\.?\b",
        ]
    ),
    (
        "Master of Science",
        [
            r"\bm\.?\s*sc\.?\b",
            r"\bmaster\s+of\s+science\b",
            r"\bmsc\b",
            r"\bm\.?\s*s\.?\b",
        ]
    ),
    (
        "Bachelor of Business Administration",
        [
            r"\bb\.?\s*b\.?\s*a\.?\b",
            r"\bbachelor\s+of\s+business\s+administration\b",
        ]
    ),
    (
        "Master of Business Administration",
        [
            r"\bm\.?\s*b\.?\s*a\.?\b",
            r"\bmaster\s+of\s+business\s+administration\b",
        ]
    ),
    (
        "Bachelor of Commerce",
        [
            r"\bb\.?\s*com\.?\b",
            r"\bbachelor\s+of\s+commerce\b",
        ]
    ),
    (
        "Master of Commerce",
        [
            r"\bm\.?\s*com\.?\b",
            r"\bmaster\s+of\s+commerce\b",
        ]
    ),
    (
        "Bachelor of Arts",
        [
            r"\bb\.?\s*a\.?\b",
            r"\bbachelor\s+of\s+arts\b",
        ]
    ),
    (
        "Master of Arts",
        [
            r"\bm\.?\s*a\.?\b",
            r"\bmaster\s+of\s+arts\b",
        ]
    ),
    (
        "Doctor of Philosophy",
        [
            r"\bph\.?\s*d\.?\b",
            r"\bdoctor\s+of\s+philosophy\b",
            r"\bdoctorate\b",
        ]
    ),
    (
        "Diploma",
        [
            r"\bdiploma\b",
        ]
    ),
]


class DataNormalizer:
    """
    Deterministic normalization service for candidate CV data.
    Provides idempotent normalization for dates, academic qualifications, and grades.
    Preserves raw data and provenance without guessing or fabricating values.
    """

    def __init__(self):
        logger.info("DataNormalizer initialized")

    def normalize_date(self, date_input: Optional[Union[str, NormalizedDate]]) -> Optional[NormalizedDate]:
        """
        Normalize unambiguous date representations into NormalizedDate.

        Handles:
        - ISO date: "2022-07-15" -> DAY precision
        - ISO month: "2022-07" -> MONTH precision
        - Month name + Year: "July 2022", "Jul 2022" -> MONTH precision
        - Year only: "2022" -> YEAR precision (never assumes month or day)
        - Current employment: "Present", "Currently working" -> is_current=True
        - Preserves raw string and avoids reinterpreting ambiguous values.

        Args:
            date_input: Raw date string or already normalized date.

        Returns:
            NormalizedDate or None if empty.
        """
        if date_input is None:
            return None

        if isinstance(date_input, NormalizedDate):
            return date_input.model_copy()

        raw_str = str(date_input).strip()
        if not raw_str:
            return None

        # Check if input is a date range in a single-date field (e.g. "2022 - Present", "2020 - 2024")
        if DATE_RANGE_PATTERN.search(raw_str):
            logger.debug("Date range detected in single date field: %s", raw_str)
            return NormalizedDate(
                raw=raw_str,
                formatted=None,
                year=None,
                month=None,
                day=None,
                precision=None,
                is_current=False,
            )

        clean_lower = raw_str.lower().strip()

        # Check for current employment indicator
        if clean_lower in CURRENT_KEYWORDS or any(re.search(rf"\b{re.escape(kw)}\b", clean_lower) for kw in CURRENT_KEYWORDS):
            return NormalizedDate(
                raw=raw_str,
                formatted="Present",
                year=None,
                month=None,
                day=None,
                precision=None,
                is_current=True,
            )

        # 1. Year only: e.g. "2022"
        year_match = re.fullmatch(r"(?:19|20)\d{2}", raw_str)
        if year_match:
            year_val = int(raw_str)
            return NormalizedDate(
                raw=raw_str,
                formatted=str(year_val),
                year=year_val,
                month=None,
                day=None,
                precision=DatePrecision.YEAR,
                is_current=False,
            )

        # 2. ISO / numeric Date with day: "2022-07-15", "2022/07/15", "2022.07.15"
        iso_day_match = re.fullmatch(r"((?:19|20)\d{2})[\-\/\.](\d{1,2})[\-\/\.](\d{1,2})", raw_str)
        if iso_day_match:
            year_val = int(iso_day_match.group(1))
            month_val = int(iso_day_match.group(2))
            day_val = int(iso_day_match.group(3))
            if 1 <= month_val <= 12 and 1 <= day_val <= 31:
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}-{day_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=day_val,
                    precision=DatePrecision.DAY,
                    is_current=False,
                )

        # 3. Numeric Month + Year (year first): "2022-07", "2022/07", "2022.07"
        iso_month_match = re.fullmatch(r"((?:19|20)\d{2})[\-\/\.](\d{1,2})", raw_str)
        if iso_month_match:
            year_val = int(iso_month_match.group(1))
            month_val = int(iso_month_match.group(2))
            if 1 <= month_val <= 12:
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=None,
                    precision=DatePrecision.MONTH,
                    is_current=False,
                )

        # 4. Numeric Month + Year (month first): "07/2022", "07-2022", "7/2022"
        month_year_match = re.fullmatch(r"(\d{1,2})[\-\/\.]((?:19|20)\d{2})", raw_str)
        if month_year_match:
            month_val = int(month_year_match.group(1))
            year_val = int(month_year_match.group(2))
            if 1 <= month_val <= 12:
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=None,
                    precision=DatePrecision.MONTH,
                    is_current=False,
                )

        # 5. Month name + Year: "July 2022", "Jul 2022", "July-2022", "Jul/2022", "Jul.2022", "July, 2022"
        month_name_year_match = re.fullmatch(r"([a-zA-Z]{3,9})\.?,?[\s\-\/\.]+((?:19|20)\d{2})", raw_str)
        if month_name_year_match:
            m_str = month_name_year_match.group(1).lower()
            year_val = int(month_name_year_match.group(2))
            if m_str in MONTH_NAME_TO_INT:
                month_val = MONTH_NAME_TO_INT[m_str]
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=None,
                    precision=DatePrecision.MONTH,
                    is_current=False,
                )

        # 6. Year + Month name: "2022 July", "2022 Jul", "2022-July"
        year_month_name_match = re.fullmatch(r"((?:19|20)\d{2})[\s\-\/\.]+([a-zA-Z]{3,9})\.?", raw_str)
        if year_month_name_match:
            year_val = int(year_month_name_match.group(1))
            m_str = year_month_name_match.group(2).lower()
            if m_str in MONTH_NAME_TO_INT:
                month_val = MONTH_NAME_TO_INT[m_str]
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=None,
                    precision=DatePrecision.MONTH,
                    is_current=False,
                )

        # 7. Day + Month Name + Year: "15 July 2022", "15th Jul 2022", "15-Jul-2022"
        day_month_name_match = re.fullmatch(
            r"(\d{1,2})(?:st|nd|rd|th)?[\s\-\/\.]+([a-zA-Z]{3,9})\.?,?[\s\-\/\.]+((?:19|20)\d{2})",
            raw_str,
        )
        if day_month_name_match:
            day_val = int(day_month_name_match.group(1))
            m_str = day_month_name_match.group(2).lower()
            year_val = int(day_month_name_match.group(3))
            if m_str in MONTH_NAME_TO_INT and 1 <= day_val <= 31:
                month_val = MONTH_NAME_TO_INT[m_str]
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}-{day_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=day_val,
                    precision=DatePrecision.DAY,
                    is_current=False,
                )

        # 8. Month Name + Day + Year: "July 15, 2022", "Jul 15 2022"
        month_name_day_match = re.fullmatch(
            r"([a-zA-Z]{3,9})\.?,?[\s\-\/\.]+(\d{1,2})(?:st|nd|rd|th)?,?[\s\-\/\.]+((?:19|20)\d{2})",
            raw_str,
        )
        if month_name_day_match:
            m_str = month_name_day_match.group(1).lower()
            day_val = int(month_name_day_match.group(2))
            year_val = int(month_name_day_match.group(3))
            if m_str in MONTH_NAME_TO_INT and 1 <= day_val <= 31:
                month_val = MONTH_NAME_TO_INT[m_str]
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{month_val:02d}-{day_val:02d}",
                    year=year_val,
                    month=month_val,
                    day=day_val,
                    precision=DatePrecision.DAY,
                    is_current=False,
                )

        # 9. 3-part numeric date: "DD/MM/YYYY" or "MM/DD/YYYY"
        three_part_match = re.fullmatch(r"(\d{1,2})[\-\/\.](\d{1,2})[\-\/\.]((?:19|20)\d{2})", raw_str)
        if three_part_match:
            p1 = int(three_part_match.group(1))
            p2 = int(three_part_match.group(2))
            year_val = int(three_part_match.group(3))

            if p1 > 12 and 1 <= p2 <= 12 and 1 <= p1 <= 31:
                # Unambiguously Day / Month / Year
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{p2:02d}-{p1:02d}",
                    year=year_val,
                    month=p2,
                    day=p1,
                    precision=DatePrecision.DAY,
                    is_current=False,
                )
            elif p2 > 12 and 1 <= p1 <= 12 and 1 <= p2 <= 31:
                # Unambiguously Month / Day / Year
                return NormalizedDate(
                    raw=raw_str,
                    formatted=f"{year_val:04d}-{p1:02d}-{p2:02d}",
                    year=year_val,
                    month=p1,
                    day=p2,
                    precision=DatePrecision.DAY,
                    is_current=False,
                )
            elif 1 <= p1 <= 12 and 1 <= p2 <= 12:
                # Ambiguous between DD/MM and MM/DD -> Do not silently guess!
                logger.debug("Ambiguous date string encountered: %s", raw_str)
                return NormalizedDate(
                    raw=raw_str,
                    formatted=None,
                    year=year_val,
                    month=None,
                    day=None,
                    precision=None,
                    is_current=False,
                )

        # 10. Check if a 4-digit year exists in text without clear month/day (e.g. "Summer 2021", "Class of 2020")
        isolated_year_match = re.search(r"\b((?:19|20)\d{2})\b", raw_str)
        if isolated_year_match:
            year_val = int(isolated_year_match.group(1))
            return NormalizedDate(
                raw=raw_str,
                formatted=str(year_val),
                year=year_val,
                month=None,
                day=None,
                precision=DatePrecision.YEAR,
                is_current=False,
            )

        # Unrecognized / non-date string: preserve raw value faithfully
        return NormalizedDate(
            raw=raw_str,
            formatted=None,
            year=None,
            month=None,
            day=None,
            precision=None,
            is_current=False,
        )

    def normalize_grade(
        self,
        gpa_value: Any,
        raw_grade_str: Optional[str] = None,
    ) -> Optional[AcademicGrade]:
        """
        Normalize academic grades distinguishing CGPA, GPA, Percentage, and Unknown scales.
        Preserves reported value and scale without universal cross-scale conversions.
        Does not assume GPA out of 4 or CGPA out of 10 without evidence.

        Args:
            gpa_value: Numeric grade value or AcademicGrade instance.
            raw_grade_str: Optional raw grade string providing context (e.g. "CGPA: 8.5/10", "85%").

        Returns:
            AcademicGrade or None if missing.
        """
        if isinstance(gpa_value, AcademicGrade):
            return gpa_value.model_copy()

        # Check if gpa_value is already a validated numeric value
        validated_numeric: Optional[float] = None
        if gpa_value is not None and not isinstance(gpa_value, bool):
            if isinstance(gpa_value, (int, float)):
                validated_numeric = float(gpa_value)
            elif isinstance(gpa_value, str):
                cleaned_val = gpa_value.strip()
                if re.fullmatch(r"\d+(?:\.\d+)?", cleaned_val):
                    try:
                        validated_numeric = float(cleaned_val)
                    except ValueError:
                        pass

        text_to_examine = ""
        if raw_grade_str and str(raw_grade_str).strip():
            text_to_examine = str(raw_grade_str).strip()
        elif gpa_value is not None:
            text_to_examine = str(gpa_value).strip()
        else:
            return None

        if not text_to_examine:
            if validated_numeric is not None:
                return AcademicGrade(
                    value=validated_numeric,
                    raw_value=str(validated_numeric),
                    scale=GradeScale.UNKNOWN,
                    max_scale=None,
                )
            return None

        lower_text = text_to_examine.lower()

        # Regex for denominator separator: "/" or "out of" or "on a scale of" or "scale of"
        denom_pattern = r"(?:\/|\bout\s+of\b|\bon\s+a\s+scale\s+of\b|\bscale\s+of\b)"

        # 1. Percentage check: "85%", "85.5 %", "85 percent", "percentage: 78.4", "10th percentage: 88.5%"
        pct_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:%|\bpercent|\bpercentage\b)", lower_text)
        if not pct_match:
            pct_prefix = re.search(r"(?:percentage|percent)\s*[:=]?\s*(\d+(?:\.\d+)?)", lower_text)
            if pct_prefix:
                pct_match = pct_prefix

        if pct_match or ("%" in lower_text and validated_numeric is not None):
            val = validated_numeric if validated_numeric is not None else float(pct_match.group(1))
            return AcademicGrade(
                value=val,
                raw_value=text_to_examine,
                scale=GradeScale.PERCENTAGE,
                max_scale=100.0,
            )

        # 2. Explicit CGPA check
        if "cgpa" in lower_text:
            # Check for explicit denominator: e.g. "8.5/10", "8.5 on a scale of 10", "8.5 out of 10"
            denom_match = re.search(
                rf"(\d+(?:\.\d+)?)\s*{denom_pattern}\s*(\d+(?:\.\d+)?)",
                lower_text,
            )
            if denom_match:
                val = validated_numeric if validated_numeric is not None else float(denom_match.group(1))
                max_val = float(denom_match.group(2))
                return AcademicGrade(
                    value=val,
                    raw_value=text_to_examine,
                    scale=GradeScale.CGPA,
                    max_scale=max_val,
                )

            # CGPA without explicit denominator
            val: Optional[float] = validated_numeric
            if val is None:
                # Look for number immediately following CGPA: "CGPA: 8.5", "CGPA - 8.5", "CGPA 8.5"
                after_match = re.search(r"\bcgpa\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)", lower_text)
                if after_match:
                    val = float(after_match.group(1))
                else:
                    # Look for number immediately preceding CGPA: "8.5 CGPA", "8.5 - CGPA"
                    before_match = re.search(r"(\d+(?:\.\d+)?)\s*[:=\-]?\s*\bcgpa\b", lower_text)
                    if before_match:
                        val = float(before_match.group(1))

            if val is not None:
                return AcademicGrade(
                    value=val,
                    raw_value=text_to_examine,
                    scale=GradeScale.CGPA,
                    max_scale=None,  # Do not assume 10 without evidence
                )

        # 3. Explicit GPA check
        if re.search(r"\bgpa\b", lower_text):
            denom_match = re.search(
                rf"(\d+(?:\.\d+)?)\s*{denom_pattern}\s*(\d+(?:\.\d+)?)",
                lower_text,
            )
            if denom_match:
                val = validated_numeric if validated_numeric is not None else float(denom_match.group(1))
                max_val = float(denom_match.group(2))
                return AcademicGrade(
                    value=val,
                    raw_value=text_to_examine,
                    scale=GradeScale.GPA,
                    max_scale=max_val,
                )

            val: Optional[float] = validated_numeric
            if val is None:
                # Look for number immediately following GPA: "GPA: 3.8", "GPA - 3.8", "GPA 3.8"
                after_match = re.search(r"\bgpa\b\s*[:=\-]?\s*(\d+(?:\.\d+)?)", lower_text)
                if after_match:
                    val = float(after_match.group(1))
                else:
                    # Look for number immediately preceding GPA: "3.8 GPA", "3.8 - GPA"
                    before_match = re.search(r"(\d+(?:\.\d+)?)\s*[:=\-]?\s*\bgpa\b", lower_text)
                    if before_match:
                        val = float(before_match.group(1))

            if val is not None:
                return AcademicGrade(
                    value=val,
                    raw_value=text_to_examine,
                    scale=GradeScale.GPA,
                    max_scale=None,  # Do not assume 4.0 without evidence
                )

        # 4. Denominator present without explicit scale name: "8.5/10", "3.8 out of 4"
        denom_match = re.search(
            rf"(\d+(?:\.\d+)?)\s*{denom_pattern}\s*(\d+(?:\.\d+)?)",
            lower_text,
        )
        if denom_match:
            val = validated_numeric if validated_numeric is not None else float(denom_match.group(1))
            max_val = float(denom_match.group(2))
            return AcademicGrade(
                value=val,
                raw_value=text_to_examine,
                scale=GradeScale.UNKNOWN,
                max_scale=max_val,
            )

        # 5. Bare float or numeric representation without scale indicator
        val: Optional[float] = validated_numeric
        if val is None:
            # Strip ordinals like 10th, 12th, class 12, etc. to avoid matching them as grades
            cleaned_for_bare = re.sub(r"\b\d+(?:st|nd|rd|th)\b", "", lower_text)
            cleaned_for_bare = re.sub(r"\b(?:class|semester|grade|year)\s+\d+\b", "", cleaned_for_bare)
            num_match = re.search(r"(\d+(?:\.\d+)?)", cleaned_for_bare)
            if num_match:
                try:
                    val = float(num_match.group(1))
                except ValueError:
                    pass

        if val is not None:
            return AcademicGrade(
                value=val,
                raw_value=text_to_examine,
                scale=GradeScale.UNKNOWN,
                max_scale=None,
            )

        return None

    def normalize_education_item(self, item: EducationItem) -> EducationItem:
        """
        Normalize an EducationItem qualification and academic grade deterministically.
        Normalizes known degree aliases to canonical forms while preserving raw values.
        Does not invent degrees, institutions, or dates.

        Args:
            item: Validated EducationItem.

        Returns:
            Normalized EducationItem copy.
        """
        raw_degree = item.raw_degree or item.degree or ""
        canonical_degree: Optional[str] = None

        # Check degree against canonical alias patterns
        if raw_degree:
            for canonical_name, patterns in CANONICAL_DEGREE_PATTERNS:
                matched = False
                for pat in patterns:
                    if re.search(pat, raw_degree, flags=re.IGNORECASE):
                        canonical_degree = canonical_name
                        matched = True
                        break
                if matched:
                    break

        # Grade normalization
        normalized_grade: Optional[AcademicGrade] = None
        if item.grade is not None:
            normalized_grade = self.normalize_grade(item.grade)
        elif item.gpa is not None:
            normalized_grade = self.normalize_grade(item.gpa)

        # Return updated copy preserving original fields
        return item.model_copy(
            update={
                "degree": canonical_degree if canonical_degree else item.degree,
                "raw_degree": raw_degree,
                "normalized_degree": canonical_degree,
                "grade": normalized_grade,
            }
        )

    def normalize_experience_item(self, item: ExperienceItem) -> ExperienceItem:
        """
        Normalize an ExperienceItem start and end dates deterministically.
        Represents current employment explicitly without inventing dates.

        Args:
            item: Validated ExperienceItem.

        Returns:
            Normalized ExperienceItem copy.
        """
        start_norm = self.normalize_date(item.start_date)
        end_norm = self.normalize_date(item.end_date) if item.end_date else None

        is_current = False
        if end_norm and end_norm.is_current:
            is_current = True
        elif not item.end_date:
            is_current = bool(item.is_current)
        else:
            is_current = False

        return item.model_copy(
            update={
                "start_date_normalized": start_norm,
                "end_date_normalized": end_norm,
                "is_current": is_current,
            }
        )

    def normalize_structured_cv(self, cv: StructuredCV) -> StructuredCV:
        """
        Normalize full StructuredCV idempotently and without side effects.

        Args:
            cv: Validated StructuredCV.

        Returns:
            New StructuredCV instance with normalized education and experience items.
        """
        normalized_education = [self.normalize_education_item(edu) for edu in cv.education]
        normalized_experience = [self.normalize_experience_item(exp) for exp in cv.experience]

        # Deep copy other collections to guarantee immutability
        return cv.model_copy(
            update={
                "education": normalized_education,
                "experience": normalized_experience,
                "projects": [p.model_copy() for p in cv.projects],
                "certifications": [c.model_copy() for c in cv.certifications],
                "skills": list(cv.skills),
            }
        )

    def normalize(self, cv: StructuredCV) -> StructuredCV:
        """Convenience alias for normalize_structured_cv."""
        return self.normalize_structured_cv(cv)


# Service alias for consistent naming convention
NormalizerService = DataNormalizer
