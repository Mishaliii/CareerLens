import logging
import re
import fitz
from dataclasses import dataclass
from typing import Optional, Dict, Any, List

logger = logging.getLogger(__name__)

# Minimum tokens needed to attempt reliable language detection
_MIN_WORDS_FOR_LANGUAGE_DETECTION = 15

# Common function words (stop words) for high-frequency deterministic language matching
_LANGUAGE_WORD_SETS: Dict[str, set] = {
    "en": {
        "the", "and", "in", "of", "to", "with", "for", "at", "on", "from",
        "by", "as", "is", "an", "this", "that", "are", "was", "be", "has",
        "have", "experience", "education", "skills", "projects", "work",
        "development", "management", "university", "engineer", "software",
    },
    "es": {
        "de", "la", "el", "en", "y", "los", "del", "las", "por", "un",
        "para", "con", "una", "su", "al", "lo", "como", "mas", "pero",
        "sus", "le", "experiencia", "educacion", "habilidades", "proyectos",
        "trabajo", "desarrollo", "universidad", "ingeniero",
    },
    "fr": {
        "de", "la", "le", "et", "les", "des", "en", "du", "une", "un",
        "pour", "dans", "qui", "au", "par", "sur", "avec", "est", "ce",
        "experience", "formation", "competences", "projets", "travail",
        "developpement", "universite", "ingenieur",
    },
    "de": {
        "der", "die", "und", "in", "den", "von", "zu", "das", "mit",
        "sich", "des", "auf", "fur", "ist", "im", "dem", "nicht", "ein",
        "eine", "berufserfahrung", "ausbildung", "kenntnisse", "projekte",
        "arbeit", "entwicklung", "universitat", "ingenieur",
    },
}


@dataclass
class DocumentAnalysisResult:
    """Explicit, typed result structure representing document analysis metrics."""
    page_count: int
    language: Optional[str]
    parser_confidence: Optional[float]
    total_characters: int
    total_words: int
    pages_with_text: int
    pages_sparse_or_empty: int
    character_density_per_page: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary representation."""
        return {
            "page_count": self.page_count,
            "language": self.language,
            "parser_confidence": self.parser_confidence,
            "total_characters": self.total_characters,
            "total_words": self.total_words,
            "pages_with_text": self.pages_with_text,
            "pages_sparse_or_empty": self.pages_sparse_or_empty,
            "character_density_per_page": self.character_density_per_page,
        }


class DocumentAnalyzer:
    """
    Analyzes PDF resume structure and extracted text to extract reliable document
    metadata (page count, language, text yield, and parsing confidence).
    """

    def __init__(self):
        logger.info("DocumentAnalyzer initialized")

    def analyze(self, file_bytes: bytes, extracted_text: Optional[str] = None) -> DocumentAnalysisResult:
        """
        Perform document analysis on raw PDF bytes and optional pre-extracted text.

        Args:
            file_bytes: Raw bytes of the PDF file.
            extracted_text: Optional pre-extracted text from CVParser to avoid re-extraction.

        Returns:
            DocumentAnalysisResult containing verified metrics.

        Raises:
            ValueError: If PDF bytes are empty, too short, or corrupted.
        """
        if not file_bytes:
            logger.error("Empty PDF bytes provided to DocumentAnalyzer")
            raise ValueError("PDF file is empty")

        if len(file_bytes) < 4:
            logger.error("PDF bytes too short (%d bytes)", len(file_bytes))
            raise ValueError("Invalid PDF: file too short to be valid")

        try:
            doc = fitz.open(stream=file_bytes, filetype="pdf")
        except fitz.FileDataError as e:
            logger.error("Corrupted or invalid PDF format: %s", str(e))
            raise ValueError("Invalid PDF file format") from e
        except Exception as e:
            logger.exception("Failed to open PDF document")
            raise ValueError("Failed to open PDF document") from e

        try:
            page_count = doc.page_count
            if page_count == 0:
                logger.error("PDF has 0 pages")
                raise ValueError("PDF has no pages")

            per_page_texts: List[str] = []
            for page_idx in range(page_count):
                page = doc[page_idx]
                text = page.get_text()
                per_page_texts.append(text)

        finally:
            doc.close()

        # Text yield calculations
        combined_text = extracted_text if extracted_text is not None else "\n".join(per_page_texts)
        total_characters = len(combined_text)
        words = combined_text.split()
        total_words = len(words)

        pages_with_text = 0
        pages_sparse_or_empty = 0

        for page_text in per_page_texts:
            stripped = page_text.strip()
            # A page with >= 20 characters of non-whitespace content has readable text
            if len(stripped) >= 20:
                pages_with_text += 1
            else:
                pages_sparse_or_empty += 1

        density = round(total_characters / page_count, 2) if page_count > 0 else 0.0

        # Language detection
        detected_language = self.detect_language(combined_text)

        # Parsing confidence calculation
        confidence = self.calculate_parsing_confidence(
            page_count=page_count,
            pages_with_text=pages_with_text,
            total_words=total_words,
            total_characters=total_characters,
            combined_text=combined_text,
        )

        logger.info(
            "Document analysis completed: pages=%d, readable_pages=%d, words=%d, lang=%s, confidence=%s",
            page_count,
            pages_with_text,
            total_words,
            detected_language,
            f"{confidence:.2f}" if confidence is not None else "None",
        )

        return DocumentAnalysisResult(
            page_count=page_count,
            language=detected_language,
            parser_confidence=confidence,
            total_characters=total_characters,
            total_words=total_words,
            pages_with_text=pages_with_text,
            pages_sparse_or_empty=pages_sparse_or_empty,
            character_density_per_page=density,
        )

    def detect_language(self, text: str) -> Optional[str]:
        """
        Detect likely language deterministically using token frequency against function-word sets.

        Args:
            text: Extracted text sample.

        Returns:
            ISO 639-1 language code (e.g. 'en', 'es', 'fr', 'de') if evidence is sufficient,
            otherwise None (unknown/insufficient evidence).
        """
        if not text:
            return None

        # Extract normalized alphabetic word tokens
        tokens = [token.lower() for token in re.findall(r'\b[a-zA-Z\u00C0-\u017F]{2,}\b', text)]
        if len(tokens) < _MIN_WORDS_FOR_LANGUAGE_DETECTION:
            logger.debug(
                "Sample too short for language detection (%d tokens < %d threshold)",
                len(tokens),
                _MIN_WORDS_FOR_LANGUAGE_DETECTION,
            )
            return None

        token_set = set(tokens)
        scores: Dict[str, int] = {}
        for lang_code, word_set in _LANGUAGE_WORD_SETS.items():
            matches = token_set.intersection(word_set)
            scores[lang_code] = len(matches)

        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        top_lang, top_score = sorted_scores[0]
        runner_up_score = sorted_scores[1][1] if len(sorted_scores) > 1 else 0

        # Require minimum of 3 distinct function word matches and clear separation from runner-up
        if top_score >= 3 and top_score > runner_up_score:
            return top_lang

        return None

    def calculate_parsing_confidence(
        self,
        page_count: int,
        pages_with_text: int,
        total_words: int,
        total_characters: int,
        combined_text: str,
    ) -> Optional[float]:
        """
        Calculate a deterministic parsing confidence heuristic between 0.0 and 1.0.

        Scoring dimensions:
        1. Readable Page Ratio (weight 0.35): Proportion of pages with readable text.
        2. Text Volume Yield (weight 0.35): Volume of extracted text compared to typical resume benchmarks.
        3. Text Cleanliness (weight 0.15): Freedom from unprintable/corrupted character artifacts.
        4. Structural Markers (weight 0.15): Presence of standard resume markers (dates, sections, emails).

        Returns:
            Float between 0.0 and 1.0, or None if the document cannot be evaluated.
        """
        if page_count <= 0:
            return None

        if total_characters == 0 or not combined_text.strip():
            return 0.0

        # Dimension 1: Page coverage ratio (0.0 to 1.0)
        readable_ratio = pages_with_text / page_count

        # Dimension 2: Text volume yield (0.0 to 1.0)
        if total_words < 15:
            yield_score = 0.1
        elif total_words < 50:
            yield_score = 0.4
        elif total_words < 120:
            yield_score = 0.7
        else:
            yield_score = 1.0

        # Dimension 3: Text cleanliness (0.0 to 1.0)
        # Check for unprintable replacement characters and abnormally long tokens
        unprintable_count = len(re.findall(r'[\ufffd\x00-\x08\x0b\x0c\x0e-\x1f]', combined_text))
        unprintable_ratio = unprintable_count / max(total_characters, 1)

        long_token_count = sum(1 for w in combined_text.split() if len(w) > 40)
        long_token_ratio = long_token_count / max(total_words, 1)

        if unprintable_ratio > 0.05 or long_token_ratio > 0.10:
            cleanliness_score = 0.3
        elif unprintable_ratio > 0.01 or long_token_ratio > 0.03:
            cleanliness_score = 0.7
        else:
            cleanliness_score = 1.0

        # Dimension 4: Structural coherence (0.0 to 1.0)
        structural_indicators = 0
        # Check for 4-digit years (e.g. 2020-2024, 1999)
        if re.search(r'\b(19\d\d|20\d\d)\b', combined_text):
            structural_indicators += 1
        # Check for common contact tokens (@ or phone pattern)
        if re.search(r'[@\+]\w+', combined_text) or re.search(r'\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b', combined_text):
            structural_indicators += 1
        # Check for common section keyword tokens
        if re.search(r'(?i)\b(experience|education|skills|projects|summary|profile)\b', combined_text):
            structural_indicators += 1

        structural_score = structural_indicators / 3.0

        # Weighted composite score
        composite = (
            (0.35 * readable_ratio)
            + (0.35 * yield_score)
            + (0.15 * cleanliness_score)
            + (0.15 * structural_score)
        )

        # Clamp between 0.0 and 1.0 and round to 2 decimal places
        clamped_score = max(0.0, min(1.0, round(composite, 2)))
        return clamped_score
