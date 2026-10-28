import unittest

from backend.models.schemas import (
    AcademicGrade,
    DatePrecision,
    EducationItem,
    ExperienceItem,
    GradeScale,
    NormalizedDate,
    ProjectItem,
    StructuredCV,
)
from backend.services.normalizer import DataNormalizer, NormalizerService


class TestDataNormalizer(unittest.TestCase):
    """Comprehensive unit tests for DataNormalizer service."""

    def setUp(self):
        self.normalizer = DataNormalizer()

    def test_service_initialization_and_alias(self):
        """Test normalizer initialization and service alias."""
        self.assertIsInstance(self.normalizer, DataNormalizer)
        alias_service = NormalizerService()
        self.assertIsInstance(alias_service, DataNormalizer)

    def test_recognized_date_formats(self):
        """Test normalization of standard unambiguous date formats."""
        # Month Name + Year
        d1 = self.normalizer.normalize_date("July 2022")
        self.assertIsNotNone(d1)
        self.assertEqual(d1.year, 2022)
        self.assertEqual(d1.month, 7)
        self.assertEqual(d1.precision, DatePrecision.MONTH)
        self.assertEqual(d1.formatted, "2022-07")
        self.assertFalse(d1.is_current)

        # Abbreviated Month Name + Year
        d2 = self.normalizer.normalize_date("Jul 2022")
        self.assertIsNotNone(d2)
        self.assertEqual(d2.year, 2022)
        self.assertEqual(d2.month, 7)
        self.assertEqual(d2.precision, DatePrecision.MONTH)
        self.assertEqual(d2.formatted, "2022-07")

        # ISO Month
        d3 = self.normalizer.normalize_date("2022-07")
        self.assertIsNotNone(d3)
        self.assertEqual(d3.year, 2022)
        self.assertEqual(d3.month, 7)
        self.assertEqual(d3.precision, DatePrecision.MONTH)
        self.assertEqual(d3.formatted, "2022-07")

        # Numeric Month/Year
        d4 = self.normalizer.normalize_date("07/2022")
        self.assertIsNotNone(d4)
        self.assertEqual(d4.year, 2022)
        self.assertEqual(d4.month, 7)
        self.assertEqual(d4.precision, DatePrecision.MONTH)
        self.assertEqual(d4.formatted, "2022-07")

        # ISO Day precision
        d5 = self.normalizer.normalize_date("2022-07-15")
        self.assertIsNotNone(d5)
        self.assertEqual(d5.year, 2022)
        self.assertEqual(d5.month, 7)
        self.assertEqual(d5.day, 15)
        self.assertEqual(d5.precision, DatePrecision.DAY)
        self.assertEqual(d5.formatted, "2022-07-15")

        # Month Day Year text
        d6 = self.normalizer.normalize_date("July 15, 2022")
        self.assertIsNotNone(d6)
        self.assertEqual(d6.year, 2022)
        self.assertEqual(d6.month, 7)
        self.assertEqual(d6.day, 15)
        self.assertEqual(d6.precision, DatePrecision.DAY)
        self.assertEqual(d6.formatted, "2022-07-15")

    def test_year_only_dates_preserve_precision(self):
        """Test year-only dates preserve YEAR precision without inventing month or day."""
        d = self.normalizer.normalize_date("2022")
        self.assertIsNotNone(d)
        self.assertEqual(d.year, 2022)
        self.assertIsNone(d.month)
        self.assertIsNone(d.day)
        self.assertEqual(d.precision, DatePrecision.YEAR)
        self.assertEqual(d.formatted, "2022")
        self.assertEqual(d.raw, "2022")

    def test_current_employment_dates(self):
        """Test that current employment keywords set is_current without inventing an end date."""
        for term in ["Present", "Currently working", "Currently Employed", "Ongoing", "Till date", "Now"]:
            d = self.normalizer.normalize_date(term)
            self.assertIsNotNone(d, f"Failed for {term}")
            self.assertTrue(d.is_current, f"Failed is_current for {term}")
            self.assertIsNone(d.year, f"Invented year for {term}")
            self.assertIsNone(d.month, f"Invented month for {term}")
            self.assertEqual(d.formatted, "Present")
            self.assertEqual(d.raw, term)

    def test_missing_and_ambiguous_dates(self):
        """Test missing dates return None and ambiguous dates do not guess day/month."""
        # Empty or None
        self.assertIsNone(self.normalizer.normalize_date(None))
        self.assertIsNone(self.normalizer.normalize_date(""))
        self.assertIsNone(self.normalizer.normalize_date("   "))

        # Ambiguous 3-part date: "03/04/2022" (could be March 4 or April 3)
        amb = self.normalizer.normalize_date("03/04/2022")
        self.assertIsNotNone(amb)
        self.assertEqual(amb.raw, "03/04/2022")
        self.assertEqual(amb.year, 2022)
        self.assertIsNone(amb.month)  # Not guessed
        self.assertIsNone(amb.day)    # Not guessed
        self.assertIsNone(amb.precision)
        self.assertIsNone(amb.formatted)

        # Unambiguous 3-part date: "25/07/2022" (25 can only be day)
        unamb = self.normalizer.normalize_date("25/07/2022")
        self.assertIsNotNone(unamb)
        self.assertEqual(unamb.year, 2022)
        self.assertEqual(unamb.month, 7)
        self.assertEqual(unamb.day, 25)
        self.assertEqual(unamb.precision, DatePrecision.DAY)
        self.assertEqual(unamb.formatted, "2022-07-25")

        # Unparseable text preserves raw string
        unparseable = self.normalizer.normalize_date("Not Applicable")
        self.assertIsNotNone(unparseable)
        self.assertEqual(unparseable.raw, "Not Applicable")
        self.assertIsNone(unparseable.formatted)

    def test_indian_and_common_education_qualifications(self):
        """Test canonical mapping of Indian and standard degree qualifications."""
        test_cases = [
            ("BCA", "Bachelor of Computer Applications"),
            ("MCA", "Master of Computer Applications"),
            ("B.Tech in Computer Science", "Bachelor of Technology"),
            ("BTech", "Bachelor of Technology"),
            ("Bachelor of Technology", "Bachelor of Technology"),
            ("M.Tech", "Master of Technology"),
            ("MTech", "Master of Technology"),
            ("B.Sc", "Bachelor of Science"),
            ("BSc in Physics", "Bachelor of Science"),
            ("Bachelor of Science", "Bachelor of Science"),
            ("M.Sc in Mathematics", "Master of Science"),
            ("MSc", "Master of Science"),
            ("B.E. Mechanical Engineering", "Bachelor of Engineering"),
            ("M.E.", "Master of Engineering"),
            ("MBA", "Master of Business Administration"),
            ("BBA", "Bachelor of Business Administration"),
            ("B.Com", "Bachelor of Commerce"),
            ("M.Com", "Master of Commerce"),
            ("Ph.D in AI", "Doctor of Philosophy"),
            ("Diploma in Engineering", "Diploma"),
        ]

        for raw_degree, expected_canonical in test_cases:
            item = EducationItem(
                institution="National University",
                degree=raw_degree,
                field="Engineering",
                start_year=2018,
                end_year=2022,
            )
            norm_item = self.normalizer.normalize_education_item(item)
            self.assertEqual(norm_item.normalized_degree, expected_canonical)
            self.assertEqual(norm_item.degree, expected_canonical)
            self.assertEqual(norm_item.raw_degree, raw_degree)
            self.assertEqual(norm_item.institution, "National University")
            self.assertEqual(norm_item.start_year, 2018)
            self.assertEqual(norm_item.end_year, 2022)

    def test_unknown_education_degree_preserved_without_guessing(self):
        """Test that unknown qualification is never forced into an invented degree."""
        raw_degree = "Certified AWS Cloud Practitioner Bootcamp"
        item = EducationItem(
            institution="Cloud Academy",
            degree=raw_degree,
            field="Cloud",
        )
        norm_item = self.normalizer.normalize_education_item(item)
        self.assertIsNone(norm_item.normalized_degree)
        self.assertEqual(norm_item.degree, raw_degree)
        self.assertEqual(norm_item.raw_degree, raw_degree)

    def test_academic_grades_cgpa_out_of_10(self):
        """Test CGPA extraction with and without explicit denominator."""
        # Explicit /10
        g1 = self.normalizer.normalize_grade(8.5, raw_grade_str="CGPA: 8.5/10")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 8.5)
        self.assertEqual(g1.scale, GradeScale.CGPA)
        self.assertEqual(g1.max_scale, 10.0)
        self.assertEqual(g1.raw_value, "CGPA: 8.5/10")

        # Explicit CGPA without denominator (must NOT assume 10 without evidence)
        g2 = self.normalizer.normalize_grade(8.5, raw_grade_str="8.5 CGPA")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 8.5)
        self.assertEqual(g2.scale, GradeScale.CGPA)
        self.assertIsNone(g2.max_scale)

    def test_academic_grades_gpa_out_of_4(self):
        """Test GPA extraction with and without explicit denominator."""
        # Explicit /4.0
        g1 = self.normalizer.normalize_grade(3.8, raw_grade_str="GPA: 3.8/4.0")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 3.8)
        self.assertEqual(g1.scale, GradeScale.GPA)
        self.assertEqual(g1.max_scale, 4.0)

        # Explicit GPA without denominator (must NOT assume 4.0 without evidence)
        g2 = self.normalizer.normalize_grade(3.8, raw_grade_str="GPA 3.8")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 3.8)
        self.assertEqual(g2.scale, GradeScale.GPA)
        self.assertIsNone(g2.max_scale)

    def test_academic_grades_percentage(self):
        """Test percentage scale detection."""
        g1 = self.normalizer.normalize_grade(85.5, raw_grade_str="85.5%")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 85.5)
        self.assertEqual(g1.scale, GradeScale.PERCENTAGE)
        self.assertEqual(g1.max_scale, 100.0)

        g2 = self.normalizer.normalize_grade(None, raw_grade_str="Percentage: 78.4")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 78.4)
        self.assertEqual(g2.scale, GradeScale.PERCENTAGE)
        self.assertEqual(g2.max_scale, 100.0)

    def test_academic_grades_unknown_scale(self):
        """Test bare numeric grade with no context maintains UNKNOWN scale and no assumed bounds."""
        # Bare float without context
        g = self.normalizer.normalize_grade(8.5)
        self.assertIsNotNone(g)
        self.assertEqual(g.value, 8.5)
        self.assertEqual(g.scale, GradeScale.UNKNOWN)
        self.assertIsNone(g.max_scale)

        # Denominator without scale label (e.g. "8.5/10") retains reported bound but UNKNOWN scale
        g_denom = self.normalizer.normalize_grade(None, raw_grade_str="8.5/10")
        self.assertIsNotNone(g_denom)
        self.assertEqual(g_denom.value, 8.5)
        self.assertEqual(g_denom.scale, GradeScale.UNKNOWN)
        self.assertEqual(g_denom.max_scale, 10.0)

    def test_experience_item_normalization(self):
        """Test normalizing experience item dates and current role flag."""
        exp = ExperienceItem(
            company="TechCorp India",
            title="Software Engineer",
            start_date="July 2021",
            end_date="Present",
            description="Developing scalable APIs",
        )
        norm_exp = self.normalizer.normalize_experience_item(exp)
        self.assertEqual(norm_exp.company, "TechCorp India")
        self.assertEqual(norm_exp.title, "Software Engineer")
        self.assertEqual(norm_exp.start_date, "July 2021")
        self.assertEqual(norm_exp.end_date, "Present")
        self.assertTrue(norm_exp.is_current)

        self.assertIsNotNone(norm_exp.start_date_normalized)
        self.assertEqual(norm_exp.start_date_normalized.formatted, "2021-07")
        self.assertEqual(norm_exp.start_date_normalized.precision, DatePrecision.MONTH)

        self.assertIsNotNone(norm_exp.end_date_normalized)
        self.assertTrue(norm_exp.end_date_normalized.is_current)

    def test_structured_cv_full_normalization(self):
        """Test full StructuredCV normalization without side-effects."""
        cv = StructuredCV(
            full_name="Mishal Sharma",
            email="mishal@example.com",
            phone="+91 9876543210",
            location="Bengaluru, Karnataka",
            education=[
                EducationItem(
                    institution="VTU",
                    degree="B.Tech in Computer Science",
                    field="Computer Science",
                    start_year=2018,
                    end_year=2022,
                    gpa=8.4,
                )
            ],
            experience=[
                ExperienceItem(
                    company="FinTech Ltd",
                    title="Junior Backend Developer",
                    start_date="Aug 2022",
                    end_date="Currently working",
                    description="Building microservices in FastAPI",
                )
            ],
            projects=[
                ProjectItem(
                    name="CareerLens AI",
                    description="AI career intelligence platform",
                    technologies=["Python", "FastAPI", "React"],
                )
            ],
            skills=["Python", "FastAPI", "Docker", "PostgreSQL"],
            extraction_source="gemini",
        )

        norm_cv = self.normalizer.normalize_structured_cv(cv)

        # Verify candidate info untouched
        self.assertEqual(norm_cv.full_name, "Mishal Sharma")
        self.assertEqual(norm_cv.email, "mishal@example.com")
        self.assertEqual(norm_cv.extraction_source, "gemini")

        # Verify education normalization
        self.assertEqual(len(norm_cv.education), 1)
        edu = norm_cv.education[0]
        self.assertEqual(edu.degree, "Bachelor of Technology")
        self.assertEqual(edu.raw_degree, "B.Tech in Computer Science")
        self.assertEqual(edu.normalized_degree, "Bachelor of Technology")
        self.assertEqual(edu.start_year, 2018)
        self.assertEqual(edu.end_year, 2022)
        self.assertEqual(edu.gpa, 8.4)
        self.assertIsNotNone(edu.grade)
        self.assertEqual(edu.grade.value, 8.4)

        # Verify experience normalization
        self.assertEqual(len(norm_cv.experience), 1)
        exp = norm_cv.experience[0]
        self.assertTrue(exp.is_current)
        self.assertIsNotNone(exp.start_date_normalized)
        self.assertEqual(exp.start_date_normalized.formatted, "2022-08")

        # Verify projects and skills preserved exactly
        self.assertEqual(len(norm_cv.projects), 1)
        self.assertEqual(norm_cv.skills, ["Python", "FastAPI", "Docker", "PostgreSQL"])

    def test_normalization_idempotency(self):
        """Test that normalizing already-normalized data yields identical result."""
        cv = StructuredCV(
            full_name="Aarav Patel",
            education=[
                EducationItem(
                    institution="IIT Bombay",
                    degree="B.Tech",
                    field="Computer Science",
                    start_year=2019,
                    end_year=2023,
                    gpa=9.1,
                )
            ],
            experience=[
                ExperienceItem(
                    company="Tech Corp",
                    title="Software Engineer",
                    start_date="2023-07",
                    end_date="Present",
                )
            ],
        )

        norm_once = self.normalizer.normalize_structured_cv(cv)
        norm_twice = self.normalizer.normalize_structured_cv(norm_once)

        self.assertEqual(norm_once.model_dump(), norm_twice.model_dump())

        # Test date idempotency
        d_once = self.normalizer.normalize_date("2022-07")
        d_twice = self.normalizer.normalize_date(d_once)
        self.assertEqual(d_once.model_dump(), d_twice.model_dump())

        # Test grade idempotency
        g_once = self.normalizer.normalize_grade(8.5, "CGPA: 8.5/10")
        g_twice = self.normalizer.normalize_grade(g_once)
        self.assertEqual(g_once.model_dump(), g_twice.model_dump())

    def test_immutability_input_not_mutated(self):
        """Test that normalization does not mutate the source input object."""
        exp = ExperienceItem(
            company="Startup",
            title="Intern",
            start_date="Jan 2023",
            end_date="Jun 2023",
        )
        original_dict = exp.model_dump()
        _ = self.normalizer.normalize_experience_item(exp)

        # Verify original exp instance was not mutated
        self.assertEqual(exp.model_dump(), original_dict)
        self.assertIsNone(exp.start_date_normalized)
        self.assertFalse(exp.is_current)

    def test_empty_and_incomplete_records(self):
        """Test handling of empty, None, and partial records without exceptions."""
        empty_cv = StructuredCV()
        norm_empty = self.normalizer.normalize_structured_cv(empty_cv)
        self.assertEqual(len(norm_empty.education), 0)
        self.assertEqual(len(norm_empty.experience), 0)

        incomplete_edu = EducationItem(institution="", degree="", field="")
        norm_edu = self.normalizer.normalize_education_item(incomplete_edu)
        self.assertIsNone(norm_edu.grade)
        self.assertIsNone(norm_edu.start_year)

        incomplete_exp = ExperienceItem(company="", title="", start_date="")
        norm_exp = self.normalizer.normalize_experience_item(incomplete_exp)
        self.assertIsNone(norm_exp.start_date_normalized)
        self.assertFalse(norm_exp.is_current)

    def test_regression_grade_extraction_ordinal_and_class_numbers(self):
        """Regression tests for Fix 1: Grade extraction with ordinal and class prefixes."""
        # 10th CGPA: 8.5 -> 8.5
        g1 = self.normalizer.normalize_grade(None, raw_grade_str="10th CGPA: 8.5")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 8.5)
        self.assertEqual(g1.scale, GradeScale.CGPA)
        self.assertIsNone(g1.max_scale)

        # 12th GPA: 3.8 -> 3.8
        g2 = self.normalizer.normalize_grade(None, raw_grade_str="12th GPA: 3.8")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 3.8)
        self.assertEqual(g2.scale, GradeScale.GPA)
        self.assertIsNone(g2.max_scale)

        # Class 12 - 8.5 CGPA -> 8.5
        g3 = self.normalizer.normalize_grade(None, raw_grade_str="Class 12 - 8.5 CGPA")
        self.assertIsNotNone(g3)
        self.assertEqual(g3.value, 8.5)
        self.assertEqual(g3.scale, GradeScale.CGPA)
        self.assertIsNone(g3.max_scale)

        # Semester 4 CGPA: 8.7 -> 8.7
        g4 = self.normalizer.normalize_grade(None, raw_grade_str="Semester 4 CGPA: 8.7")
        self.assertIsNotNone(g4)
        self.assertEqual(g4.value, 8.7)
        self.assertEqual(g4.scale, GradeScale.CGPA)
        self.assertIsNone(g4.max_scale)

        # 10th percentage: 88.5% -> 88.5
        g5 = self.normalizer.normalize_grade(None, raw_grade_str="10th percentage: 88.5%")
        self.assertIsNotNone(g5)
        self.assertEqual(g5.value, 88.5)
        self.assertEqual(g5.scale, GradeScale.PERCENTAGE)
        self.assertEqual(g5.max_scale, 100.0)

    def test_regression_validated_numeric_preservation_in_grade(self):
        """Regression test for Fix 1: Validated numeric grade value is preserved."""
        # When gpa_value is already validated (e.g. 8.5), unrelated numbers in raw string do not override it
        g1 = self.normalizer.normalize_grade(8.5, raw_grade_str="10th Class")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 8.5)
        self.assertEqual(g1.scale, GradeScale.UNKNOWN)
        self.assertIsNone(g1.max_scale)

        # Validated numeric with scale evidence in raw string
        g2 = self.normalizer.normalize_grade(8.5, raw_grade_str="10th CGPA: 8.5")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 8.5)
        self.assertEqual(g2.scale, GradeScale.CGPA)
        self.assertIsNone(g2.max_scale)

        # Bare number without raw string must not infer scale
        g3 = self.normalizer.normalize_grade(3.8)
        self.assertIsNotNone(g3)
        self.assertEqual(g3.value, 3.8)
        self.assertEqual(g3.scale, GradeScale.UNKNOWN)
        self.assertIsNone(g3.max_scale)

    def test_regression_month_name_delimiters(self):
        """Regression tests for Fix 2: Month-name dates separated by -, /, or ."""
        # July-2022 -> 2022-07
        d1 = self.normalizer.normalize_date("July-2022")
        self.assertIsNotNone(d1)
        self.assertEqual(d1.formatted, "2022-07")
        self.assertEqual(d1.precision, DatePrecision.MONTH)
        self.assertEqual(d1.year, 2022)
        self.assertEqual(d1.month, 7)
        self.assertEqual(d1.raw, "July-2022")

        # Jul/2022 -> 2022-07
        d2 = self.normalizer.normalize_date("Jul/2022")
        self.assertIsNotNone(d2)
        self.assertEqual(d2.formatted, "2022-07")
        self.assertEqual(d2.precision, DatePrecision.MONTH)
        self.assertEqual(d2.year, 2022)
        self.assertEqual(d2.month, 7)
        self.assertEqual(d2.raw, "Jul/2022")

        # Jul.2022 -> 2022-07
        d3 = self.normalizer.normalize_date("Jul.2022")
        self.assertIsNotNone(d3)
        self.assertEqual(d3.formatted, "2022-07")
        self.assertEqual(d3.precision, DatePrecision.MONTH)
        self.assertEqual(d3.year, 2022)
        self.assertEqual(d3.month, 7)
        self.assertEqual(d3.raw, "Jul.2022")

        # 2022-July -> 2022-07
        d4 = self.normalizer.normalize_date("2022-July")
        self.assertIsNotNone(d4)
        self.assertEqual(d4.formatted, "2022-07")
        self.assertEqual(d4.precision, DatePrecision.MONTH)
        self.assertEqual(d4.year, 2022)
        self.assertEqual(d4.month, 7)

    def test_regression_preserve_current_employment_when_end_date_none(self):
        """Regression tests for Fix 3: Preserve is_current=True when end_date=None."""
        # Case 1: is_current=True and end_date=None -> remains True
        exp1 = ExperienceItem(
            company="Google",
            title="Senior Engineer",
            start_date="2021",
            end_date=None,
            is_current=True,
        )
        norm1 = self.normalizer.normalize_experience_item(exp1)
        self.assertTrue(norm1.is_current)
        self.assertIsNone(norm1.end_date)
        self.assertIsNone(norm1.end_date_normalized)

        # Idempotency check: re-normalizing preserves is_current=True
        norm1_again = self.normalizer.normalize_experience_item(norm1)
        self.assertTrue(norm1_again.is_current)
        self.assertEqual(norm1.model_dump(), norm1_again.model_dump())

        # Case 2: is_current=False and end_date=None -> must NOT be marked current merely because end_date is missing
        exp2 = ExperienceItem(
            company="Prior Startup",
            title="Developer",
            start_date="2019",
            end_date=None,
            is_current=False,
        )
        norm2 = self.normalizer.normalize_experience_item(exp2)
        self.assertFalse(norm2.is_current)

        # Case 3: end_date explicitly contains recognized current employment term
        exp3 = ExperienceItem(
            company="Current Corp",
            title="Lead Architect",
            start_date="2022",
            end_date="Present",
            is_current=False,
        )
        norm3 = self.normalizer.normalize_experience_item(exp3)
        self.assertTrue(norm3.is_current)

    def test_regression_explicit_verbal_grading_scales(self):
        """Regression tests for Fix 4: Explicit verbal grading scale denominators."""
        # CGPA 8.5 on a scale of 10
        g1 = self.normalizer.normalize_grade(None, raw_grade_str="CGPA 8.5 on a scale of 10")
        self.assertIsNotNone(g1)
        self.assertEqual(g1.value, 8.5)
        self.assertEqual(g1.scale, GradeScale.CGPA)
        self.assertEqual(g1.max_scale, 10.0)

        # GPA 3.8 out of 4.0
        g2 = self.normalizer.normalize_grade(None, raw_grade_str="GPA 3.8 out of 4.0")
        self.assertIsNotNone(g2)
        self.assertEqual(g2.value, 3.8)
        self.assertEqual(g2.scale, GradeScale.GPA)
        self.assertEqual(g2.max_scale, 4.0)

        # Verbal denominator without scale label remains UNKNOWN scale with explicit max_scale
        g3 = self.normalizer.normalize_grade(None, raw_grade_str="8.5 out of 10")
        self.assertIsNotNone(g3)
        self.assertEqual(g3.value, 8.5)
        self.assertEqual(g3.scale, GradeScale.UNKNOWN)
        self.assertEqual(g3.max_scale, 10.0)

        # No denominator stated -> do NOT infer denominator from label alone
        g4 = self.normalizer.normalize_grade(None, raw_grade_str="CGPA: 8.5")
        self.assertIsNotNone(g4)
        self.assertEqual(g4.value, 8.5)
        self.assertEqual(g4.scale, GradeScale.CGPA)
        self.assertIsNone(g4.max_scale)

    def test_regression_single_field_date_ranges(self):
        """Regression tests for Fix 5: Single-field date ranges return conservative result."""
        # 2022 - Present
        d1 = self.normalizer.normalize_date("2022 - Present")
        self.assertIsNotNone(d1)
        self.assertEqual(d1.raw, "2022 - Present")
        self.assertIsNone(d1.formatted)
        self.assertIsNone(d1.year)
        self.assertIsNone(d1.month)
        self.assertIsNone(d1.day)
        self.assertIsNone(d1.precision)
        self.assertFalse(d1.is_current)

        # 2020 - 2024
        d2 = self.normalizer.normalize_date("2020 - 2024")
        self.assertIsNotNone(d2)
        self.assertEqual(d2.raw, "2020 - 2024")
        self.assertIsNone(d2.formatted)
        self.assertIsNone(d2.year)
        self.assertIsNone(d2.month)
        self.assertIsNone(d2.day)
        self.assertIsNone(d2.precision)

        # 2020 to 2024
        d3 = self.normalizer.normalize_date("2020 to 2024")
        self.assertIsNotNone(d3)
        self.assertEqual(d3.raw, "2020 to 2024")
        self.assertIsNone(d3.formatted)
        self.assertIsNone(d3.year)
        self.assertIsNone(d3.month)
        self.assertIsNone(d3.day)
        self.assertIsNone(d3.precision)


if __name__ == "__main__":
    unittest.main()
