import unittest

from backend.models.schemas import (
    CertificationItem,
    EducationItem,
    ExperienceItem,
    ProjectItem,
    ResumeChunk,
    StructuredCV,
)
from backend.services.chunker import ResumeChunker


class TestResumeChunker(unittest.TestCase):
    """Unit tests for ResumeChunker service."""

    def setUp(self):
        self.chunker = ResumeChunker(target_chunk_words=100, overlap_words=20)
        self.resume_id = "test_resume_001"

    def test_empty_or_whitespace_input(self):
        """Test that empty or blank text yields empty list."""
        self.assertEqual(self.chunker.chunk(self.resume_id, ""), [])
        self.assertEqual(self.chunker.chunk(self.resume_id, "   \n\n  "), [])

    def test_chunking_with_structured_cv(self):
        """Test section- and entity-aware chunking from structured CV."""
        structured_cv = StructuredCV(
            summary="Passionate backend engineer with 4 years of experience.",
            experience=[
                ExperienceItem(
                    company="Tech Corp",
                    title="Senior Backend Engineer",
                    start_date="2021",
                    end_date="Present",
                    description="Scaled FastAPI services to 10k RPS. Maintained PostgreSQL databases.",
                ),
                ExperienceItem(
                    company="Startup Inc",
                    title="Junior Developer",
                    start_date="2019",
                    end_date="2021",
                    description="Built REST APIs using Python and Django.",
                ),
            ],
            projects=[
                ProjectItem(
                    name="CareerLens AI",
                    description="Career intelligence platform for freshers.",
                    technologies=["Python", "FastAPI", "React"],
                    link="https://github.com/example/careerlens",
                )
            ],
            education=[
                EducationItem(
                    institution="National Institute of Technology",
                    degree="B.Tech",
                    field="Computer Science",
                    start_year=2015,
                    end_year=2019,
                    gpa=8.8,
                )
            ],
            skills=["Python", "FastAPI", "PostgreSQL", "Docker", "Redis"],
            certifications=[
                CertificationItem(
                    name="AWS Solutions Architect",
                    issuer="Amazon Web Services",
                    date_obtained="2022",
                )
            ],
        )

        page_texts = [
            {"page_number": 1, "text": "Passionate backend engineer. Senior Backend Engineer at Tech Corp. National Institute of Technology."},
            {"page_number": 2, "text": "CareerLens AI project. Skills: Python, FastAPI, Docker. AWS Solutions Architect."},
        ]

        chunks = self.chunker.chunk(
            resume_id=self.resume_id,
            raw_text="Complete text representation",
            structured_cv=structured_cv,
            page_texts=page_texts,
        )

        self.assertGreaterEqual(len(chunks), 6)
        sections = [c.section for c in chunks]
        self.assertIn("summary", sections)
        self.assertIn("experience", sections)
        self.assertIn("projects", sections)
        self.assertIn("education", sections)
        self.assertIn("skills", sections)
        self.assertIn("certifications", sections)

        # Check page attribution
        summary_chunk = next(c for c in chunks if c.section == "summary")
        self.assertEqual(summary_chunk.page_number, 1)

        project_chunk = next(c for c in chunks if c.section == "projects")
        self.assertEqual(project_chunk.page_number, 2)
        self.assertIn("CareerLens AI", project_chunk.content)
        self.assertIn("Technologies: Python, FastAPI, React", project_chunk.content)

        # Check experience metadata preservation
        exp_chunks = [c for c in chunks if c.section == "experience"]
        self.assertEqual(len(exp_chunks), 2)
        self.assertEqual(exp_chunks[0].metadata["company"], "Tech Corp")
        self.assertIn("Tech Corp", exp_chunks[0].item_title)

    def test_chunking_with_parsed_sections_fallback(self):
        """Test chunking from regex-parsed sections when structured CV is absent."""
        parsed_sections = {
            "summary": "Full stack engineer with frontend and backend proficiency.",
            "skills": "Python, TypeScript, React, Node.js, SQL",
            "experience": "Software Engineer at Acme Corp (2020-2023)\nDeveloped microservices.",
            "education": "B.Sc Computer Science, ABC University, 2020",
        }

        chunks = self.chunker.chunk(
            resume_id=self.resume_id,
            raw_text="Raw resume full text",
            parsed_sections=parsed_sections,
        )

        self.assertGreaterEqual(len(chunks), 4)
        sections = {c.section for c in chunks}
        self.assertIn("summary", sections)
        self.assertIn("skills", sections)
        self.assertIn("experience", sections)
        self.assertIn("education", sections)

    def test_unstructured_sliding_window_fallback(self):
        """Test fallback paragraph chunking for raw text with no sections."""
        unstructured_text = (
            "Paragraph one describing early background and self-taught programming.\n\n"
            "Paragraph two describing work on open source projects and web applications.\n\n"
            "Paragraph three describing university coursework and certifications."
        )

        chunks = self.chunker.chunk(
            resume_id=self.resume_id,
            raw_text=unstructured_text,
        )

        self.assertGreaterEqual(len(chunks), 1)
        for c in chunks:
            self.assertEqual(c.section, "general")
            self.assertIn("[Resume Content]", c.content)

    def test_deterministic_chunk_ids(self):
        """Test that identical content generates deterministic, identical chunk IDs."""
        structured_cv = StructuredCV(
            summary="Deterministic summary test.",
            skills=["Python", "FastAPI"],
        )

        chunks1 = self.chunker.chunk(self.resume_id, "Raw text", structured_cv=structured_cv)
        chunks2 = self.chunker.chunk(self.resume_id, "Raw text", structured_cv=structured_cv)

        self.assertEqual(len(chunks1), len(chunks2))
        for c1, c2 in zip(chunks1, chunks2):
            self.assertEqual(c1.chunk_id, c2.chunk_id)
            self.assertEqual(c1.content, c2.content)

    def test_uncertain_page_number_is_none(self):
        """Test that unknown or missing page text maps page_number to None."""
        structured_cv = StructuredCV(
            summary="Unique non-matching text that is definitely absent from pages.",
        )
        page_texts = [{"page_number": 1, "text": "Completely unrelated content."}]

        chunks = self.chunker.chunk(
            self.resume_id,
            "Raw text",
            structured_cv=structured_cv,
            page_texts=page_texts,
        )
        self.assertEqual(len(chunks), 1)
        self.assertIsNone(chunks[0].page_number)


if __name__ == "__main__":
    unittest.main()
