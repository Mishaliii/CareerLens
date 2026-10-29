import logging
import re
import fitz
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

# Precompiled regex patterns for section detection
_SECTION_PATTERNS = {
    "summary": re.compile(
        r'(?:Summary|Professional Summary|Profile|About Me|Objective)\s*[:\s]',
        re.IGNORECASE
    ),
    "experience": re.compile(
        r'(?:Experience|Work Experience|Employment|Work History|Professional Experience)\s*[:\s]',
        re.IGNORECASE
    ),
    "education": re.compile(
        r'(?:Education|Academic Background|Academic|Qualifications|Educational Background)\s*[:\s]',
        re.IGNORECASE
    ),
    "projects": re.compile(
        r'(?:Projects|Project Experience|Personal Projects|Key Projects)\s*[:\s]',
        re.IGNORECASE
    ),
    "skills": re.compile(
        r'(?:Skills|Technical Skills|Core Skills|Key Skills|Skillset|Technologies)\s*[:\s]',
        re.IGNORECASE
    ),
    "certifications": re.compile(
        r'(?:Certifications|Certificates|Certification|Professional Certifications)\s*[:\s]',
        re.IGNORECASE
    ),
}


class CVParser:
    """CV Parser for extracting text and identifying resume sections."""
    
    def __init__(self):
        logger.info("CV Parser initialized")
    
    def parse_pdf(self, file_bytes: bytes) -> str:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_bytes: Raw PDF file bytes
            
        Returns:
            Cleaned text extracted from PDF
            
        Raises:
            ValueError: If PDF is invalid, empty, or unreadable
        """
        if not file_bytes:
            logger.error("Empty PDF bytes provided")
            raise ValueError("PDF file is empty")
        
        if len(file_bytes) < 4:
            logger.error("Invalid PDF bytes: too short")
            raise ValueError("Invalid PDF: file too short to be valid")
        
        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            
            if doc.page_count == 0:
                logger.error("PDF has no pages")
                doc.close()
                raise ValueError("PDF has no pages")
            
            text_parts = []
            total_chars = 0
            
            for page_num in range(doc.page_count):
                page = doc[page_num]
                page_text = page.get_text()
                text_parts.append(page_text)
                total_chars += len(page_text)
            
            page_count = doc.page_count
            doc.close()
            
            full_text = "\n".join(text_parts)
            cleaned_text = self._clean_text(full_text)
            
            logger.info(
                "PDF parsed successfully: %d pages, %d characters extracted",
                page_count,
                total_chars,
            )
            
            if not cleaned_text.strip():
                logger.warning("PDF parsed but no readable text found")
                raise ValueError("PDF contains no readable text")
            
            return cleaned_text
            
        except fitz.FileDataError as e:
            logger.error("Invalid PDF data: %s", str(e))
            raise ValueError("Invalid PDF file format") from e
        except Exception as e:
            logger.exception("Unexpected error parsing PDF")
            raise

    def parse_pdf_pages(self, file_bytes: bytes) -> List[Dict[str, Any]]:
        """
        Extract text page by page from PDF using PyMuPDF, preserving 1-based page numbers.

        Args:
            file_bytes: Raw PDF file bytes

        Returns:
            List of dicts: [{"page_number": int, "text": str, "char_count": int}, ...]

        Raises:
            ValueError: If PDF is empty, invalid, corrupted, or contains no readable text.
        """
        if not file_bytes:
            logger.error("Empty PDF bytes provided to parse_pdf_pages")
            raise ValueError("PDF file is empty")

        if len(file_bytes) < 4:
            logger.error("Invalid PDF bytes: too short (%d bytes)", len(file_bytes))
            raise ValueError("Invalid PDF: file too short to be valid")

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")

            if doc.page_count == 0:
                logger.error("PDF has no pages")
                doc.close()
                raise ValueError("PDF has no pages")

            pages_data: List[Dict[str, Any]] = []
            total_chars = 0

            for page_num in range(doc.page_count):
                page = doc[page_num]
                raw_page_text = page.get_text()
                cleaned_page_text = self._clean_text(raw_page_text)
                pages_data.append({
                    "page_number": page_num + 1,
                    "text": cleaned_page_text,
                    "char_count": len(cleaned_page_text),
                })
                total_chars += len(cleaned_page_text.strip())

            page_count = doc.page_count
            doc.close()

            logger.info(
                "PDF pages parsed successfully: %d pages, %d readable characters extracted",
                page_count,
                total_chars,
            )

            if total_chars == 0:
                logger.warning("PDF parsed page-by-page but no readable text found")
                raise ValueError("PDF contains no readable text")

            return pages_data

        except fitz.FileDataError as e:
            logger.error("Invalid PDF data: %s", str(e))
            raise ValueError("Invalid PDF file format") from e
        except ValueError:
            raise
        except Exception as e:
            logger.exception("Unexpected error parsing PDF page-by-page")
            raise
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving line breaks.
        
        Args:
            text: Raw extracted text
            
        Returns:
            Cleaned text with preserved line structure
        """
        # Replace multiple spaces with single space (but preserve newlines)
        text = re.sub(r'[^\S\n]+', ' ', text)
        # Replace multiple newlines with double newline (paragraph break)
        text = re.sub(r'\n{3,}', '\n\n', text)
        # Remove leading/trailing whitespace
        text = text.strip()
        return text
    
    def extract_sections(self, raw_text: str) -> Dict[str, str]:
        """
        Detect and extract common resume sections using regex.
        
        Args:
            raw_text: Full resume text
            
        Returns:
            Dictionary with extracted sections
        """
        sections = {
            "summary": "",
            "experience": "",
            "education": "",
            "projects": "",
            "skills": "",
            "certifications": "",
            "raw_text": raw_text
        }
        
        if not raw_text:
            logger.warning("Empty text provided for section extraction")
            return sections
        
        # Find all section positions using precompiled patterns
        section_positions = {}
        for section_name, pattern in _SECTION_PATTERNS.items():
            match = pattern.search(raw_text)
            if match:
                section_positions[section_name] = match.start()
        
        # Sort sections by position
        sorted_sections = sorted(section_positions.items(), key=lambda x: x[1])
        
        # Extract section content
        for i, (section_name, start_pos) in enumerate(sorted_sections):
            # Find end position (start of next section or end of text)
            if i + 1 < len(sorted_sections):
                end_pos = sorted_sections[i + 1][1]
            else:
                end_pos = len(raw_text)
            
            section_content = raw_text[start_pos:end_pos].strip()
            # Remove the section header using precompiled pattern
            pattern = _SECTION_PATTERNS[section_name]
            section_content = pattern.sub('', section_content)
            
            sections[section_name] = section_content.strip()
        
        detected_count = sum(1 for v in sections.values() if v and v != raw_text)
        logger.info("Sections detected: %d sections found", detected_count)
        
        return sections
    
    def parse(self, file_bytes: bytes) -> Dict[str, str]:
        """
        Orchestrate the full CV parsing pipeline.
        
        This method will coordinate:
        1. PDF text extraction
        2. Section identification
        3. (Future) Gemini validation
        4. (Future) JSON structuring
        5. (Future) Database storage
        
        Args:
            file_bytes: Raw PDF file bytes
            
        Returns:
            Dictionary containing parsed sections and metadata
            
        Raises:
            ValueError: If PDF parsing fails
        """
        logger.info("Starting CV parsing pipeline")
        
        # Stage 1: Extract text from PDF
        raw_text = self.parse_pdf(file_bytes)
        
        # Stage 2: Identify sections
        sections = self.extract_sections(raw_text)
        
        # Future stages (to be implemented in later sprints):
        # - Stage 3: Validate with Gemini
        # - Stage 4: Structure as JSON
        # - Stage 5: Store in database
        
        logger.info("CV parsing pipeline completed")
        
        return sections
