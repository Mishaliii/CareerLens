import unittest
import fitz
from backend.services.cv_parser import CVParser


class TestCVParser(unittest.TestCase):
    """Unit tests for CV Parser."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.parser = CVParser()
    
    def _create_test_pdf(self, text: str) -> bytes:
        """Create a test PDF with given text."""
        doc = fitz.open()
        page = doc.new_page()
        page.insert_text((50, 50), text)
        pdf_bytes = doc.tobytes()
        doc.close()
        return pdf_bytes
    
    def test_valid_pdf(self):
        """Test parsing a valid PDF."""
        test_text = "John Doe\nSoftware Engineer\nExperience: 5 years"
        pdf_bytes = self._create_test_pdf(test_text)
        
        result = self.parser.parse_pdf(pdf_bytes)
        
        self.assertIsNotNone(result)
        self.assertIn("John Doe", result)
        self.assertIn("Software Engineer", result)
    
    def test_empty_pdf(self):
        """Test parsing an empty PDF."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_pdf(b"")
        
        self.assertIn("empty", str(context.exception).lower())
    
    def test_invalid_bytes(self):
        """Test parsing invalid bytes."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_pdf(b"not a pdf")
        
        self.assertIn("invalid", str(context.exception).lower())
    
    def test_too_short_bytes(self):
        """Test parsing bytes that are too short to be a PDF."""
        with self.assertRaises(ValueError) as context:
            self.parser.parse_pdf(b"abc")
        
        self.assertIn("too short", str(context.exception).lower())
    
    def test_resume_with_standard_sections(self):
        """Test extracting sections from a resume with standard sections."""
        resume_text = """
        John Doe
        Software Engineer
        
        Summary
        Experienced software engineer with 5 years in full-stack development.
        
        Experience
        Senior Developer at Tech Corp (2020-Present)
        Junior Developer at Startup Inc (2018-2020)
        
        Education
        B.Tech in Computer Science from ABC University (2018)
        
        Projects
        E-commerce Platform
        Real-time Chat Application
        
        Skills
        Python, JavaScript, React, Node.js, SQL
        
        Certifications
        AWS Certified Developer
        Google Cloud Professional
        """
        
        pdf_bytes = self._create_test_pdf(resume_text)
        raw_text = self.parser.parse_pdf(pdf_bytes)
        sections = self.parser.extract_sections(raw_text)
        
        self.assertIn("summary", sections)
        self.assertIn("experience", sections)
        self.assertIn("education", sections)
        self.assertIn("projects", sections)
        self.assertIn("skills", sections)
        self.assertIn("certifications", sections)
        
        self.assertIn("Experienced software engineer", sections["summary"])
        self.assertIn("Senior Developer", sections["experience"])
        self.assertIn("B.Tech", sections["education"])
        self.assertIn("E-commerce Platform", sections["projects"])
        self.assertIn("Python", sections["skills"])
        self.assertIn("AWS Certified", sections["certifications"])
    
    def test_resume_missing_sections(self):
        """Test extracting sections from a resume missing some sections."""
        resume_text = """
        John Doe
        Software Engineer
        
        Summary
        Entry-level developer looking for opportunities.
        
        Education
        B.Sc in Computer Science from XYZ College (2023)
        
        Skills
        Python, Java, HTML, CSS
        """
        
        pdf_bytes = self._create_test_pdf(resume_text)
        raw_text = self.parser.parse_pdf(pdf_bytes)
        sections = self.parser.extract_sections(raw_text)
        
        self.assertIn("summary", sections)
        self.assertIn("education", sections)
        self.assertIn("skills", sections)
        self.assertIn("experience", sections)
        self.assertIn("projects", sections)
        self.assertIn("certifications", sections)
        
        # Present sections should have content
        self.assertIn("Entry-level developer", sections["summary"])
        self.assertIn("B.Sc", sections["education"])
        self.assertIn("Python", sections["skills"])
        
        # Missing sections should be empty
        self.assertEqual(sections["experience"], "")
        self.assertEqual(sections["projects"], "")
        self.assertEqual(sections["certifications"], "")
    
    def test_section_variations(self):
        """Test that different section name variations are detected."""
        resume_text = """
        Jane Smith
        Data Analyst
        
        Professional Summary
        Data analyst with expertise in Python and SQL.
        
        Work Experience
        Analyst at Data Corp (2022-Present)
        
        Academic Background
        M.Sc in Data Science from DEF University (2022)
        
        Technical Skills
        Python, SQL, Tableau, Power BI
        """
        
        pdf_bytes = self._create_test_pdf(resume_text)
        raw_text = self.parser.parse_pdf(pdf_bytes)
        sections = self.parser.extract_sections(raw_text)
        
        self.assertIn("Data analyst with expertise", sections["summary"])
        self.assertIn("Analyst at Data Corp", sections["experience"])
        self.assertIn("M.Sc in Data Science", sections["education"])
        self.assertIn("Python", sections["skills"])
    
    def test_empty_text_section_extraction(self):
        """Test section extraction with empty text."""
        sections = self.parser.extract_sections("")
        
        self.assertEqual(sections["summary"], "")
        self.assertEqual(sections["experience"], "")
        self.assertEqual(sections["education"], "")
        self.assertEqual(sections["projects"], "")
        self.assertEqual(sections["skills"], "")
        self.assertEqual(sections["certifications"], "")
        self.assertEqual(sections["raw_text"], "")
    
    def test_raw_text_preserved(self):
        """Test that raw_text is preserved in sections."""
        resume_text = "Test Resume Content"
        pdf_bytes = self._create_test_pdf(resume_text)
        raw_text = self.parser.parse_pdf(pdf_bytes)
        sections = self.parser.extract_sections(raw_text)
        
        self.assertEqual(sections["raw_text"], raw_text)
    
    def test_multi_page_pdf(self):
        """Test parsing a multi-page PDF."""
        doc = fitz.open()
        page1 = doc.new_page()
        page1.insert_text((50, 50), "Page 1 Content\nJohn Doe")
        page2 = doc.new_page()
        page2.insert_text((50, 50), "Page 2 Content\nSoftware Engineer")
        pdf_bytes = doc.tobytes()
        doc.close()
        
        result = self.parser.parse_pdf(pdf_bytes)
        
        self.assertIn("Page 1 Content", result)
        self.assertIn("Page 2 Content", result)
        self.assertIn("John Doe", result)
        self.assertIn("Software Engineer", result)
    
    def test_unicode_text(self):
        """Test parsing PDF with unicode characters."""
        unicode_text = "José García\nSoftware Engineer\nFrançois Müller\nSkills: Python, JavaScript"
        pdf_bytes = self._create_test_pdf(unicode_text)
        
        result = self.parser.parse_pdf(pdf_bytes)
        
        self.assertIn("José García", result)
        self.assertIn("François Müller", result)
    
    def test_corrupted_pdf(self):
        """Test parsing a corrupted PDF."""
        # Use a reliably invalid PDF header
        invalid_pdf = b'%PDF-1.4\n' + b'\x00' * 1000 + b'%%EOF'
        
        with self.assertRaises(ValueError):
            self.parser.parse_pdf(invalid_pdf)
    
    def test_parse_orchestration(self):
        """Test the parse() orchestration method."""
        resume_text = """
        John Doe
        Software Engineer
        
        Summary
        Experienced developer.
        
        Skills
        Python, JavaScript
        """
        
        pdf_bytes = self._create_test_pdf(resume_text)
        result = self.parser.parse(pdf_bytes)
        
        self.assertIsInstance(result, dict)
        self.assertIn("summary", result)
        self.assertIn("skills", result)
        self.assertIn("raw_text", result)
        self.assertIn("Experienced developer", result["summary"])
        self.assertIn("Python", result["skills"])
    
    def test_resume_without_section_headings(self):
        """Test parsing resume without recognizable section headings."""
        resume_text = """
        John Doe
        Software Engineer
        Worked at ABC Company
        Python
        FastAPI
        SQL
        """
        
        pdf_bytes = self._create_test_pdf(resume_text)
        raw_text = self.parser.parse_pdf(pdf_bytes)
        sections = self.parser.extract_sections(raw_text)
        
        # Should not crash and should preserve raw_text
        self.assertEqual(sections["raw_text"], raw_text)
        
        # All section fields should be empty since no headings detected
        self.assertEqual(sections["summary"], "")
        self.assertEqual(sections["experience"], "")
        self.assertEqual(sections["education"], "")
        self.assertEqual(sections["projects"], "")
        self.assertEqual(sections["skills"], "")
        self.assertEqual(sections["certifications"], "")

    def test_parse_pdf_pages_multipage(self):
        """Test extracting text page by page with 1-based page numbers."""
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "Page One Content: Professional Summary")
        p2 = doc.new_page()
        p2.insert_text((50, 50), "Page Two Content: Technical Experience")
        pdf_bytes = doc.tobytes()
        doc.close()

        pages = self.parser.parse_pdf_pages(pdf_bytes)
        self.assertEqual(len(pages), 2)
        self.assertEqual(pages[0]["page_number"], 1)
        self.assertIn("Page One Content", pages[0]["text"])
        self.assertEqual(pages[1]["page_number"], 2)
        self.assertIn("Page Two Content", pages[1]["text"])

    def test_parse_pdf_pages_empty_and_corrupted(self):
        """Test parse_pdf_pages on empty and malformed byte inputs."""
        with self.assertRaises(ValueError):
            self.parser.parse_pdf_pages(b"")
        with self.assertRaises(ValueError):
            self.parser.parse_pdf_pages(b"short")
        with self.assertRaises(ValueError):
            self.parser.parse_pdf_pages(b"%PDF-corrupted-bytes-data")

    def test_parse_pdf_pages_no_readable_text(self):
        """Test parse_pdf_pages on empty page PDF without readable text."""
        doc = fitz.open()
        doc.new_page()  # Blank page
        pdf_bytes = doc.tobytes()
        doc.close()

        with self.assertRaises(ValueError) as ctx:
            self.parser.parse_pdf_pages(pdf_bytes)
        self.assertIn("no readable text", str(ctx.exception).lower())


if __name__ == "__main__":
    unittest.main()
