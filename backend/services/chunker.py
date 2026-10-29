import hashlib
import logging
import re
from typing import Any, Dict, List, Optional

from backend.models.schemas import ResumeChunk, StructuredCV

logger = logging.getLogger(__name__)


class ResumeChunker:
    """
    Section-aware and entity-bound resume chunker.

    Segments resumes into logical retrieval units (summary, experience roles,
    discrete projects, education entries, skills, and certifications) while
    preserving employer names, titles, technologies, and 1-based page attribution.
    Falls back to paragraph-aware sliding window chunking for unstructured resumes.
    """

    def __init__(self, target_chunk_words: int = 250, overlap_words: int = 50):
        self.target_chunk_words = target_chunk_words
        self.overlap_words = overlap_words
        logger.info("ResumeChunker initialized (target_words=%d, overlap=%d)", target_chunk_words, overlap_words)

    def chunk(
        self,
        resume_id: str,
        raw_text: str,
        parsed_sections: Optional[Dict[str, Any]] = None,
        structured_cv: Optional[StructuredCV] = None,
        page_texts: Optional[List[Dict[str, Any]]] = None,
    ) -> List[ResumeChunk]:
        """
        Chunk resume content into deterministic, section-scoped ResumeChunk instances.

        Args:
            resume_id: Deterministic identifier for the resume.
            raw_text: Cleaned full raw text of the resume.
            parsed_sections: Optional dictionary of regex-extracted sections.
            structured_cv: Optional verified StructuredCV instance.
            page_texts: Optional list of page text dicts [{"page_number": int, "text": str}].

        Returns:
            List of validated ResumeChunk objects.
        """
        if not raw_text or not raw_text.strip():
            logger.warning("Empty raw_text provided to ResumeChunker")
            return []

        chunks: List[ResumeChunk] = []
        seen_contents = set()

        # Strategy 1: Entity-bound chunking using structured_cv if populated
        if structured_cv and self._has_structured_content(structured_cv):
            chunks = self._chunk_from_structured_cv(
                resume_id=resume_id,
                structured_cv=structured_cv,
                parsed_sections=parsed_sections,
                page_texts=page_texts,
            )

        # Strategy 2: Section-aware chunking from parsed_sections
        if not chunks and parsed_sections:
            chunks = self._chunk_from_sections(
                resume_id=resume_id,
                parsed_sections=parsed_sections,
                page_texts=page_texts,
            )

        # Strategy 3: Paragraph-aware sliding-window fallback for unstructured resumes
        if not chunks:
            chunks = self._chunk_unstructured_fallback(
                resume_id=resume_id,
                raw_text=raw_text,
                page_texts=page_texts,
            )

        # Deduplicate and remove empty chunks
        valid_chunks: List[ResumeChunk] = []
        for c in chunks:
            clean_content = c.content.strip()
            if not clean_content:
                continue
            content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()
            if content_hash in seen_contents:
                continue
            seen_contents.add(content_hash)
            valid_chunks.append(c)

        logger.info("Resume chunking completed: generated %d chunks for resume_id=%s", len(valid_chunks), resume_id)
        return valid_chunks

    def _has_structured_content(self, cv: StructuredCV) -> bool:
        """Check if structured CV contains meaningful entities."""
        return bool(cv.experience or cv.projects or cv.education or cv.skills or cv.certifications or cv.summary)

    def _chunk_from_structured_cv(
        self,
        resume_id: str,
        structured_cv: StructuredCV,
        parsed_sections: Optional[Dict[str, Any]],
        page_texts: Optional[List[Dict[str, Any]]],
    ) -> List[ResumeChunk]:
        """Generate chunks from structured CV entities."""
        chunks: List[ResumeChunk] = []
        idx = 0

        # 1. Summary
        summary_text = structured_cv.summary.strip()
        if not summary_text and parsed_sections and parsed_sections.get("summary"):
            summary_text = parsed_sections["summary"].strip()

        if summary_text:
            content = f"[Professional Summary]\n{summary_text}"
            page_num = self._find_page_number(summary_text, page_texts)
            chunks.append(self._create_chunk(
                resume_id=resume_id,
                section="summary",
                index=idx,
                content=content,
                item_title="Professional Summary",
                page_number=page_num,
                metadata={"type": "summary"},
            ))
            idx += 1

        # 2. Experience Items
        if structured_cv.experience:
            for exp in structured_cv.experience:
                title_part = exp.title or "Role"
                company_part = exp.company or "Company"
                item_title = f"{title_part} at {company_part}"
                dates = f"{exp.start_date} - {exp.end_date or 'Present'}".strip()
                desc = exp.description or ""
                content = f"[Experience: {item_title} ({dates})]\n{desc}".strip()

                page_num = self._find_page_number(company_part, page_texts) or self._find_page_number(desc[:60], page_texts)
                chunks.append(self._create_chunk(
                    resume_id=resume_id,
                    section="experience",
                    index=idx,
                    content=content,
                    item_title=item_title,
                    page_number=page_num,
                    metadata={
                        "company": exp.company,
                        "title": exp.title,
                        "start_date": exp.start_date,
                        "end_date": exp.end_date,
                        "is_current": exp.is_current,
                    },
                ))
                idx += 1
        elif parsed_sections and parsed_sections.get("experience"):
            exp_text = parsed_sections["experience"].strip()
            if exp_text and exp_text != parsed_sections.get("raw_text"):
                for para in [p.strip() for p in re.split(r'\n{2,}|\n(?=[A-Z0-9•\-\*])', exp_text) if p.strip()]:
                    page_num = self._find_page_number(para[:50], page_texts)
                    chunks.append(self._create_chunk(
                        resume_id=resume_id,
                        section="experience",
                        index=idx,
                        content=f"[Experience]\n{para}",
                        item_title="Experience",
                        page_number=page_num,
                        metadata={"source": "parsed_sections"},
                    ))
                    idx += 1

        # 3. Project Items
        if structured_cv.projects:
            for proj in structured_cv.projects:
                item_title = f"Project: {proj.name}"
                tech_line = f"Technologies: {', '.join(proj.technologies)}" if proj.technologies else ""
                desc = proj.description or ""
                link_line = f"Link: {proj.link}" if proj.link else ""
                parts = [f"[Project: {proj.name}]"]
                if tech_line:
                    parts.append(tech_line)
                if link_line:
                    parts.append(link_line)
                if desc:
                    parts.append(desc)
                content = "\n".join(parts)

                page_num = self._find_page_number(proj.name, page_texts) or self._find_page_number(desc[:60], page_texts)
                chunks.append(self._create_chunk(
                    resume_id=resume_id,
                    section="projects",
                    index=idx,
                    content=content,
                    item_title=item_title,
                    page_number=page_num,
                    metadata={
                        "project_name": proj.name,
                        "technologies": proj.technologies,
                        "link": proj.link,
                    },
                ))
                idx += 1
        elif parsed_sections and parsed_sections.get("projects"):
            proj_text = parsed_sections["projects"].strip()
            if proj_text and proj_text != parsed_sections.get("raw_text"):
                for para in [p.strip() for p in re.split(r'\n{2,}|\n(?=[A-Z0-9•\-\*])', proj_text) if p.strip()]:
                    page_num = self._find_page_number(para[:50], page_texts)
                    chunks.append(self._create_chunk(
                        resume_id=resume_id,
                        section="projects",
                        index=idx,
                        content=f"[Project]\n{para}",
                        item_title="Project",
                        page_number=page_num,
                        metadata={"source": "parsed_sections"},
                    ))
                    idx += 1

        # 4. Education Items
        if structured_cv.education:
            for edu in structured_cv.education:
                deg = edu.degree or "Degree"
                field = f" in {edu.field}" if edu.field else ""
                inst = f" at {edu.institution}" if edu.institution else ""
                item_title = f"{deg}{field}{inst}".strip()
                years = f"Years: {edu.start_year or ''} - {edu.end_year or ''}".strip()
                gpa = f"GPA/Score: {edu.gpa}" if edu.gpa is not None else ""
                content = f"[Education: {item_title}]\n{years} {gpa}".strip()

                search_anchor = edu.institution or deg
                page_num = self._find_page_number(search_anchor, page_texts)
                chunks.append(self._create_chunk(
                    resume_id=resume_id,
                    section="education",
                    index=idx,
                    content=content,
                    item_title=item_title,
                    page_number=page_num,
                    metadata={
                        "institution": edu.institution,
                        "degree": edu.degree,
                        "field": edu.field,
                        "start_year": edu.start_year,
                        "end_year": edu.end_year,
                        "gpa": edu.gpa,
                    },
                ))
                idx += 1
        elif parsed_sections and parsed_sections.get("education"):
            edu_text = parsed_sections["education"].strip()
            if edu_text and edu_text != parsed_sections.get("raw_text"):
                for para in [p.strip() for p in re.split(r'\n{2,}|\n(?=[A-Z0-9•\-\*])', edu_text) if p.strip()]:
                    page_num = self._find_page_number(para[:50], page_texts)
                    chunks.append(self._create_chunk(
                        resume_id=resume_id,
                        section="education",
                        index=idx,
                        content=f"[Education]\n{para}",
                        item_title="Education",
                        page_number=page_num,
                        metadata={"source": "parsed_sections"},
                    ))
                    idx += 1

        # 5. Skills Items
        if structured_cv.skills:
            item_title = "Technical Skills"
            skills_str = ", ".join(structured_cv.skills)
            content = f"[Skills: {item_title}]\n{skills_str}"
            page_num = self._find_page_number("Skills", page_texts) or self._find_page_number(structured_cv.skills[0], page_texts)
            chunks.append(self._create_chunk(
                resume_id=resume_id,
                section="skills",
                index=idx,
                content=content,
                item_title=item_title,
                page_number=page_num,
                metadata={"skills": structured_cv.skills, "skill_count": len(structured_cv.skills)},
            ))
            idx += 1

        # 6. Certifications
        for cert in structured_cv.certifications:
            item_title = f"Certification: {cert.name}"
            issuer_str = f"Issuer: {cert.issuer}" if cert.issuer else ""
            date_str = f"Date: {cert.date_obtained}" if cert.date_obtained else ""
            content = f"[Certification: {cert.name}]\n{issuer_str} {date_str}".strip()

            page_num = self._find_page_number(cert.name, page_texts)
            chunks.append(self._create_chunk(
                resume_id=resume_id,
                section="certifications",
                index=idx,
                content=content,
                item_title=item_title,
                page_number=page_num,
                metadata={"name": cert.name, "issuer": cert.issuer, "date_obtained": cert.date_obtained},
            ))
            idx += 1

        return chunks

    def _chunk_from_sections(
        self,
        resume_id: str,
        parsed_sections: Dict[str, Any],
        page_texts: Optional[List[Dict[str, Any]]],
    ) -> List[ResumeChunk]:
        """Generate chunks from regex-extracted sections."""
        chunks: List[ResumeChunk] = []
        idx = 0

        section_names = ["summary", "experience", "projects", "education", "skills", "certifications"]
        for section in section_names:
            section_content = parsed_sections.get(section, "")
            if not section_content or not section_content.strip() or section_content == parsed_sections.get("raw_text"):
                continue

            # Split large sections into sub-blocks if they contain multiple paragraphs/items
            paragraphs = [p.strip() for p in re.split(r'\n{2,}|\n(?=[A-Z0-9•\-\*])', section_content) if p.strip()]
            if not paragraphs:
                paragraphs = [section_content.strip()]

            for p_idx, para in enumerate(paragraphs):
                item_title = f"{section.capitalize()} Item {p_idx + 1}" if len(paragraphs) > 1 else section.capitalize()
                content = f"[{section.capitalize()}]\n{para}"
                page_num = self._find_page_number(para[:50], page_texts)
                chunks.append(self._create_chunk(
                    resume_id=resume_id,
                    section=section,
                    index=idx,
                    content=content,
                    item_title=item_title,
                    page_number=page_num,
                    metadata={"section": section, "source": "regex_section"},
                ))
                idx += 1

        return chunks

    def _chunk_unstructured_fallback(
        self,
        resume_id: str,
        raw_text: str,
        page_texts: Optional[List[Dict[str, Any]]],
    ) -> List[ResumeChunk]:
        """Sliding-window paragraph-aware fallback for resumes without clear sections."""
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        if not paragraphs:
            paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]

        chunks: List[ResumeChunk] = []
        current_words: List[str] = []
        idx = 0

        for para in paragraphs:
            para_words = para.split()
            if len(current_words) + len(para_words) > self.target_chunk_words and current_words:
                chunk_text = " ".join(current_words)
                page_num = self._find_page_number(chunk_text[:50], page_texts)
                chunks.append(self._create_chunk(
                    resume_id=resume_id,
                    section="general",
                    index=idx,
                    content=f"[Resume Content]\n{chunk_text}",
                    item_title=f"Excerpt {idx + 1}",
                    page_number=page_num,
                    metadata={"source": "unstructured_fallback"},
                ))
                idx += 1
                # Retain overlap
                current_words = current_words[-self.overlap_words:] if self.overlap_words < len(current_words) else []

            current_words.extend(para_words)

        if current_words:
            chunk_text = " ".join(current_words)
            page_num = self._find_page_number(chunk_text[:50], page_texts)
            chunks.append(self._create_chunk(
                resume_id=resume_id,
                section="general",
                index=idx,
                content=f"[Resume Content]\n{chunk_text}",
                item_title=f"Excerpt {idx + 1}",
                page_number=page_num,
                metadata={"source": "unstructured_fallback"},
            ))

        return chunks

    def _create_chunk(
        self,
        resume_id: str,
        section: str,
        index: int,
        content: str,
        item_title: Optional[str],
        page_number: Optional[int],
        metadata: Dict[str, Any],
    ) -> ResumeChunk:
        """Create a validated ResumeChunk with a deterministic chunk_id."""
        clean_content = content.strip()
        content_hash = hashlib.sha256(clean_content.encode("utf-8")).hexdigest()[:8]
        chunk_id = f"{resume_id}_{section}_{index}_{content_hash}"
        return ResumeChunk(
            chunk_id=chunk_id,
            resume_id=resume_id,
            section=section,
            item_title=item_title,
            page_number=page_number,
            content=clean_content,
            metadata=metadata,
        )

    def _find_page_number(
        self,
        anchor_text: Optional[str],
        page_texts: Optional[List[Dict[str, Any]]],
    ) -> Optional[int]:
        """
        Reliably locate the 1-based page number containing anchor text.
        Returns None if anchor text is absent or attribution is uncertain.
        """
        if not anchor_text or not page_texts:
            return None

        clean_anchor = anchor_text.strip().lower()
        if len(clean_anchor) < 4:
            return None

        # 1. Direct substring match
        for page in page_texts:
            page_num = page.get("page_number")
            raw_page = str(page.get("text", "")).lower()
            if clean_anchor in raw_page:
                return page_num

        # 2. Leading prefix match (first 3-4 significant words)
        words = [w for w in re.split(r'\W+', clean_anchor) if len(w) > 2]
        if len(words) >= 3:
            prefix_3 = " ".join(words[:3])
            for page in page_texts:
                page_num = page.get("page_number")
                raw_page = str(page.get("text", "")).lower()
                clean_raw_page = " ".join([w for w in re.split(r'\W+', raw_page) if len(w) > 2])
                if prefix_3 in clean_raw_page:
                    return page_num

        return None
