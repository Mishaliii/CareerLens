import logging
import re
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel, Field

from backend.models.schemas import ProjectItem, StructuredCV

logger = logging.getLogger(__name__)


class SkillMatch(BaseModel):
    """Represents the result of matching a single skill string against the ontology."""
    original: str
    canonical: str
    is_recognized: bool
    category: Optional[str] = None


SkillMatchResult = SkillMatch


class SkillOntology:
    """
    Deterministic skill ontology and technology mapping service for CareerLens AI.
    Normalizes candidate skills and project technologies to canonical names while
    preserving original values, avoiding substring false positives, and leaving unknown skills intact.
    """

    # Canonical taxonomy with explicit categories and alias mappings
    TAXONOMY: Dict[str, Dict[str, Any]] = {
        # Programming languages
        "Python": {
            "category": "Programming Languages",
            "aliases": ["python", "python3", "python 3", "python 3.x", "python2", "python 2", "py"],
        },
        "Java": {
            "category": "Programming Languages",
            "aliases": ["java", "core java", "java 8", "java 11", "java 17", "java 21", "java se", "java ee"],
        },
        "JavaScript": {
            "category": "Programming Languages",
            "aliases": ["javascript", "js", "es6", "es6+", "es2015", "ecmascript"],
        },
        "TypeScript": {
            "category": "Programming Languages",
            "aliases": ["typescript", "ts"],
        },
        "C++": {
            "category": "Programming Languages",
            "aliases": ["c++", "cpp", "c plus plus"],
        },
        "C#": {
            "category": "Programming Languages",
            "aliases": ["c#", "csharp", "c sharp", "c#.net"],
        },
        "SQL": {
            "category": "Programming Languages",
            "aliases": ["sql", "structured query language"],
        },
        "R": {
            "category": "Programming Languages",
            "aliases": ["r", "r language", "r programming", "r-lang", "rlang"],
        },

        # Frontend
        "React": {
            "category": "Frontend",
            "aliases": ["react", "reactjs", "react.js", "react js"],
        },
        "Angular": {
            "category": "Frontend",
            "aliases": ["angular", "angularjs", "angular.js", "angular js", "angular 2+", "angular 2"],
        },
        "Vue.js": {
            "category": "Frontend",
            "aliases": ["vue.js", "vue", "vuejs", "vue js", "vue 3", "vue 2"],
        },
        "HTML": {
            "category": "Frontend",
            "aliases": ["html", "html5", "html 5"],
        },
        "CSS": {
            "category": "Frontend",
            "aliases": ["css", "css3", "css 3"],
        },
        "Tailwind CSS": {
            "category": "Frontend",
            "aliases": ["tailwind css", "tailwind", "tailwindcss", "tailwind-css"],
        },

        # Backend
        "FastAPI": {
            "category": "Backend",
            "aliases": ["fastapi", "fast api", "fast-api"],
        },
        "Django": {
            "category": "Backend",
            "aliases": ["django", "django rest framework", "django-rest-framework", "drf"],
        },
        "Flask": {
            "category": "Backend",
            "aliases": ["flask"],
        },
        "Node.js": {
            "category": "Backend",
            "aliases": ["node.js", "node", "nodejs", "node js"],
        },
        "Express.js": {
            "category": "Backend",
            "aliases": ["express.js", "express", "expressjs", "express js"],
        },

        # Machine learning and AI
        "PyTorch": {
            "category": "Machine Learning & AI",
            "aliases": ["pytorch", "torch", "py-torch"],
        },
        "TensorFlow": {
            "category": "Machine Learning & AI",
            "aliases": ["tensorflow", "tf", "tensor flow", "tensor-flow"],
        },
        "scikit-learn": {
            "category": "Machine Learning & AI",
            "aliases": ["scikit-learn", "sklearn", "sk-learn", "sk learn", "scikit learn", "scikitlearn"],
        },
        "Hugging Face Transformers": {
            "category": "Machine Learning & AI",
            "aliases": [
                "hugging face transformers",
                "huggingface transformers",
                "hugging face",
                "huggingface",
                "transformers",
            ],
        },
        "LangChain": {
            "category": "Machine Learning & AI",
            "aliases": ["langchain", "lang chain", "lang-chain"],
        },
        "OpenCV": {
            "category": "Machine Learning & AI",
            "aliases": ["opencv", "open cv", "open-cv", "cv2"],
        },
        "FAISS": {
            "category": "Machine Learning & AI",
            "aliases": ["faiss"],
        },

        # Data engineering
        "Apache Spark": {
            "category": "Data Engineering",
            "aliases": ["apache spark", "spark"],
        },
        "PySpark": {
            "category": "Data Engineering",
            "aliases": ["pyspark", "py-spark", "py spark"],
        },
        "Apache Airflow": {
            "category": "Data Engineering",
            "aliases": ["apache airflow", "airflow"],
        },
        "Kafka": {
            "category": "Data Engineering",
            "aliases": ["kafka", "apache kafka"],
        },
        "dbt": {
            "category": "Data Engineering",
            "aliases": ["dbt", "data build tool"],
        },

        # Databases and search
        "PostgreSQL": {
            "category": "Databases & Search",
            "aliases": ["postgresql", "postgres", "psql", "pg"],
        },
        "MySQL": {
            "category": "Databases & Search",
            "aliases": ["mysql", "my-sql", "my sql"],
        },
        "SQLite": {
            "category": "Databases & Search",
            "aliases": ["sqlite", "sqlite3"],
        },
        "MongoDB": {
            "category": "Databases & Search",
            "aliases": ["mongodb", "mongo", "mongo db"],
        },
        "Redis": {
            "category": "Databases & Search",
            "aliases": ["redis"],
        },
        "Elasticsearch": {
            "category": "Databases & Search",
            "aliases": ["elasticsearch", "elastic search"],
        },
        "pgvector": {
            "category": "Databases & Search",
            "aliases": ["pgvector", "pg vector", "pg_vector"],
        },

        # Cloud and infrastructure
        "AWS": {
            "category": "Cloud & Infrastructure",
            "aliases": ["aws", "amazon web services"],
        },
        "Microsoft Azure": {
            "category": "Cloud & Infrastructure",
            "aliases": ["microsoft azure", "azure", "ms azure"],
        },
        "Google Cloud": {
            "category": "Cloud & Infrastructure",
            "aliases": ["google cloud", "gcp", "google cloud platform"],
        },
        "Docker": {
            "category": "Cloud & Infrastructure",
            "aliases": ["docker"],
        },
        "Kubernetes": {
            "category": "Cloud & Infrastructure",
            "aliases": ["kubernetes", "k8s", "k-8s", "kube"],
        },
        "Git": {
            "category": "Cloud & Infrastructure",
            "aliases": ["git"],
        },
        "GitHub": {
            "category": "Cloud & Infrastructure",
            "aliases": ["github", "git hub"],
        },

        # Data analysis
        "pandas": {
            "category": "Data Analysis",
            "aliases": ["pandas"],
        },
        "NumPy": {
            "category": "Data Analysis",
            "aliases": ["numpy", "num py"],
        },
        "Matplotlib": {
            "category": "Data Analysis",
            "aliases": ["matplotlib"],
        },
        "Power BI": {
            "category": "Data Analysis",
            "aliases": ["power bi", "powerbi", "power-bi"],
        },
        "Tableau": {
            "category": "Data Analysis",
            "aliases": ["tableau"],
        },
    }

    def __init__(self):
        # Build normalized lookup map: cleaned_alias -> canonical_name
        self._alias_map: Dict[str, str] = {}
        self._canonical_categories: Dict[str, str] = {}

        for canonical, info in self.TAXONOMY.items():
            category = info["category"]
            self._canonical_categories[canonical] = category

            # Register canonical name itself
            cleaned_canonical = self._clean_token(canonical)
            self._alias_map[cleaned_canonical] = canonical

            # Register explicit aliases
            for alias in info.get("aliases", []):
                cleaned_alias = self._clean_token(alias)
                self._alias_map[cleaned_alias] = canonical

        logger.info("SkillOntology initialized with %d canonical skills and %d alias mappings",
                    len(self.TAXONOMY), len(self._alias_map))

    def _clean_token(self, text: str) -> str:
        """
        Normalize a skill string for lookup while strictly preserving intentional punctuation
        such as C++, C#, Node.js, and scikit-learn.
        """
        # Lowercase and strip
        cleaned = text.lower().strip()
        # Normalize internal whitespace
        cleaned = re.sub(r"\s+", " ", cleaned)
        # Strip trailing punctuation often left by resume comma-separated lists
        cleaned = re.sub(r"[,;:|]+$", "", cleaned).strip()
        return cleaned

    def normalize_skill(self, skill: str) -> SkillMatch:
        """
        Normalize a single skill string, returning a SkillMatch with full provenance.
        Preserves unknown skills rather than guessing or dropping them.
        Does not use loose substring matching.

        Args:
            skill: Raw skill string.

        Returns:
            SkillMatch instance containing original, canonical, recognition flag, and category.
        """
        if skill is None:
            return SkillMatch(
                original="",
                canonical="",
                is_recognized=False,
                category=None,
            )

        raw_original = str(skill).strip()
        if not raw_original:
            return SkillMatch(
                original=raw_original,
                canonical="",
                is_recognized=False,
                category=None,
            )

        cleaned_key = self._clean_token(raw_original)

        # 1. Direct exact lookup in alias map
        if cleaned_key in self._alias_map:
            canonical = self._alias_map[cleaned_key]
            return SkillMatch(
                original=raw_original,
                canonical=canonical,
                is_recognized=True,
                category=self._canonical_categories.get(canonical),
            )

        # 2. Try normalized spacing / hyphen variations
        # Replace hyphens with spaces: e.g. "open-cv" -> "open cv"
        if "-" in cleaned_key:
            spaced = cleaned_key.replace("-", " ")
            if spaced in self._alias_map:
                canonical = self._alias_map[spaced]
                return SkillMatch(
                    original=raw_original,
                    canonical=canonical,
                    is_recognized=True,
                    category=self._canonical_categories.get(canonical),
                )

        # Remove spaces: e.g. "power bi" -> "powerbi"
        if " " in cleaned_key:
            unspaced = cleaned_key.replace(" ", "")
            if unspaced in self._alias_map:
                canonical = self._alias_map[unspaced]
                return SkillMatch(
                    original=raw_original,
                    canonical=canonical,
                    is_recognized=True,
                    category=self._canonical_categories.get(canonical),
                )

        # 3. Check for trailing version numbers when base is a recognized technology
        # e.g. "python 3.11", "python 3.9" -> "python", "angular 16" -> "angular"
        version_stripped = re.sub(r"\s+v?\d+(?:\.\d+)*(?:\.x)?$", "", cleaned_key).strip()
        if version_stripped and version_stripped != cleaned_key and version_stripped in self._alias_map:
            canonical = self._alias_map[version_stripped]
            return SkillMatch(
                original=raw_original,
                canonical=canonical,
                is_recognized=True,
                category=self._canonical_categories.get(canonical),
            )

        # 4. Unknown skill: preserve the cleaned original string faithfully
        # Strip trailing list separators (e.g. "Rust," -> "Rust")
        clean_preserved = re.sub(r"[,;:|]+$", "", raw_original).strip()
        return SkillMatch(
            original=raw_original,
            canonical=clean_preserved,
            is_recognized=False,
            category=None,
        )

    def get_canonical_name(self, skill: str) -> str:
        """
        Return the canonical name for a recognized skill, or the preserved string if unknown.
        """
        return self.normalize_skill(skill).canonical

    def is_recognized(self, skill: str) -> bool:
        """
        Return True if the skill is recognized in the ontology taxonomy/aliases.
        """
        return self.normalize_skill(skill).is_recognized

    def get_category(self, skill: str) -> Optional[str]:
        """
        Return the category for a recognized skill, or None if unrecognized.
        """
        return self.normalize_skill(skill).category

    def normalize_skills(self, skills: List[str]) -> List[str]:
        """
        Normalize a list of skills into deduplicated canonical names while preserving order
        and preserving unknown skills.
        """
        if not skills:
            return []

        seen: Set[str] = set()
        normalized_list: List[str] = []

        for skill in skills:
            if skill is None:
                continue
            canonical = self.get_canonical_name(skill)
            if not canonical:
                continue
            canonical_lower = canonical.lower()
            if canonical_lower not in seen:
                seen.add(canonical_lower)
                normalized_list.append(canonical)

        return normalized_list

    def normalize_skills_detailed(self, skills: List[str]) -> List[SkillMatch]:
        """
        Normalize a list of skills, returning detailed SkillMatch objects with provenance.
        """
        if not skills:
            return []
        return [self.normalize_skill(s) for s in skills if s is not None and str(s).strip()]

    def get_skills_mapping(self, skills: List[str]) -> Dict[str, SkillMatch]:
        """
        Return a dictionary mapping each input skill string to its SkillMatch result.
        """
        mapping: Dict[str, SkillMatch] = {}
        for s in skills:
            if s is not None and str(s).strip():
                clean_s = str(s).strip()
                mapping[clean_s] = self.normalize_skill(clean_s)
        return mapping

    def normalize_project(self, project: ProjectItem) -> ProjectItem:
        """
        Return a copy of ProjectItem with its technologies normalized.
        Input object is not mutated.
        """
        return project.model_copy(
            update={
                "technologies": self.normalize_skills(project.technologies)
            }
        )

    def normalize_cv_skills(self, cv: StructuredCV) -> StructuredCV:
        """
        Return a copy of StructuredCV with normalized skills and normalized project technologies.
        Input object is not mutated.
        """
        normalized_skills = self.normalize_skills(cv.skills)
        normalized_projects = [self.normalize_project(p) for p in cv.projects]

        return cv.model_copy(
            update={
                "skills": normalized_skills,
                "projects": normalized_projects,
            }
        )
