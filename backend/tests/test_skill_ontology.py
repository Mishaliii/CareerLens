import unittest

from backend.models.schemas import ProjectItem, StructuredCV
from backend.services.skill_ontology import SkillMatch, SkillOntology


class TestSkillOntology(unittest.TestCase):
    """Comprehensive unit tests for SkillOntology service."""

    def setUp(self):
        self.ontology = SkillOntology()

    def test_taxonomy_categories_and_canonical_names(self):
        """Test that all required categories and key canonical skills exist in taxonomy."""
        expected_categories = {
            "Programming Languages",
            "Frontend",
            "Backend",
            "Machine Learning & AI",
            "Data Engineering",
            "Databases & Search",
            "Cloud & Infrastructure",
            "Data Analysis",
        }
        self.assertEqual(set(self.ontology._canonical_categories.values()), expected_categories)
        self.assertEqual(len(self.ontology.TAXONOMY), 50)

    def test_explicitly_defined_aliases(self):
        """Test all explicitly mandated alias-to-canonical mappings."""
        alias_checks = [
            # Prompt mandatory mappings
            ("JS", "JavaScript", "Programming Languages"),
            ("TS", "TypeScript", "Programming Languages"),
            ("sklearn", "scikit-learn", "Machine Learning & AI"),
            ("sk learn", "scikit-learn", "Machine Learning & AI"),
            ("scikit learn", "scikit-learn", "Machine Learning & AI"),
            ("Postgres", "PostgreSQL", "Databases & Search"),
            ("PostgreSQL", "PostgreSQL", "Databases & Search"),
            ("K8s", "Kubernetes", "Cloud & Infrastructure"),
            ("k8s", "Kubernetes", "Cloud & Infrastructure"),
            ("ReactJS", "React", "Frontend"),
            ("React.js", "React", "Frontend"),
            ("PyTorch", "PyTorch", "Machine Learning & AI"),
            ("Open CV", "OpenCV", "Machine Learning & AI"),
            ("opencv", "OpenCV", "Machine Learning & AI"),
            # Programming languages
            ("python", "Python", "Programming Languages"),
            ("python 3", "Python", "Programming Languages"),
            ("python3", "Python", "Programming Languages"),
            ("c++", "C++", "Programming Languages"),
            ("cpp", "C++", "Programming Languages"),
            ("c plus plus", "C++", "Programming Languages"),
            ("c#", "C#", "Programming Languages"),
            ("csharp", "C#", "Programming Languages"),
            ("c sharp", "C#", "Programming Languages"),
            ("sql", "SQL", "Programming Languages"),
            ("r programming", "R", "Programming Languages"),
            # Frontend
            ("react js", "React", "Frontend"),
            ("angularjs", "Angular", "Frontend"),
            ("vue.js", "Vue.js", "Frontend"),
            ("vuejs", "Vue.js", "Frontend"),
            ("html5", "HTML", "Frontend"),
            ("css3", "CSS", "Frontend"),
            ("tailwind", "Tailwind CSS", "Frontend"),
            ("tailwindcss", "Tailwind CSS", "Frontend"),
            # Backend
            ("fastapi", "FastAPI", "Backend"),
            ("fast api", "FastAPI", "Backend"),
            ("django rest framework", "Django", "Backend"),
            ("drf", "Django", "Backend"),
            ("flask", "Flask", "Backend"),
            ("nodejs", "Node.js", "Backend"),
            ("node.js", "Node.js", "Backend"),
            ("express", "Express.js", "Backend"),
            ("expressjs", "Express.js", "Backend"),
            # Machine learning & AI
            ("torch", "PyTorch", "Machine Learning & AI"),
            ("tensorflow", "TensorFlow", "Machine Learning & AI"),
            ("tf", "TensorFlow", "Machine Learning & AI"),
            ("huggingface", "Hugging Face Transformers", "Machine Learning & AI"),
            ("hugging face", "Hugging Face Transformers", "Machine Learning & AI"),
            ("langchain", "LangChain", "Machine Learning & AI"),
            ("faiss", "FAISS", "Machine Learning & AI"),
            # Data engineering
            ("spark", "Apache Spark", "Data Engineering"),
            ("apache spark", "Apache Spark", "Data Engineering"),
            ("pyspark", "PySpark", "Data Engineering"),
            ("airflow", "Apache Airflow", "Data Engineering"),
            ("kafka", "Kafka", "Data Engineering"),
            ("dbt", "dbt", "Data Engineering"),
            # Databases & Search
            ("psql", "PostgreSQL", "Databases & Search"),
            ("mongo", "MongoDB", "Databases & Search"),
            ("mongodb", "MongoDB", "Databases & Search"),
            ("mysql", "MySQL", "Databases & Search"),
            ("sqlite3", "SQLite", "Databases & Search"),
            ("redis", "Redis", "Databases & Search"),
            ("elasticsearch", "Elasticsearch", "Databases & Search"),
            ("pgvector", "pgvector", "Databases & Search"),
            # Cloud & Infrastructure
            ("aws", "AWS", "Cloud & Infrastructure"),
            ("amazon web services", "AWS", "Cloud & Infrastructure"),
            ("azure", "Microsoft Azure", "Cloud & Infrastructure"),
            ("ms azure", "Microsoft Azure", "Cloud & Infrastructure"),
            ("gcp", "Google Cloud", "Cloud & Infrastructure"),
            ("google cloud platform", "Google Cloud", "Cloud & Infrastructure"),
            ("docker", "Docker", "Cloud & Infrastructure"),
            ("kubernetes", "Kubernetes", "Cloud & Infrastructure"),
            ("git", "Git", "Cloud & Infrastructure"),
            ("github", "GitHub", "Cloud & Infrastructure"),
            # Data analysis
            ("pandas", "pandas", "Data Analysis"),
            ("numpy", "NumPy", "Data Analysis"),
            ("matplotlib", "Matplotlib", "Data Analysis"),
            ("powerbi", "Power BI", "Data Analysis"),
            ("power bi", "Power BI", "Data Analysis"),
            ("tableau", "Tableau", "Data Analysis"),
        ]

        for alias, expected_canonical, expected_cat in alias_checks:
            match = self.ontology.normalize_skill(alias)
            self.assertEqual(match.canonical, expected_canonical, f"Failed for alias: {alias}")
            self.assertTrue(match.is_recognized, f"Alias not recognized: {alias}")
            self.assertEqual(match.category, expected_cat, f"Wrong category for: {alias}")
            self.assertEqual(match.original, alias)

    def test_case_and_whitespace_variations(self):
        """Test case-insensitivity and whitespace resilience."""
        variations = [
            ("   javascript   ", "JavaScript"),
            ("PYTHON", "Python"),
            ("  react   js  ", "React"),
            ("Scikit-Learn", "scikit-learn"),
            ("kUbErNeTeS", "Kubernetes"),
            ("pOsTgReS", "PostgreSQL"),
            ("c++", "C++"),
            ("C++", "C++"),
            ("c#", "C#"),
            ("C#", "C#"),
        ]
        for var_input, expected_canonical in variations:
            match = self.ontology.normalize_skill(var_input)
            self.assertEqual(match.canonical, expected_canonical)
            self.assertTrue(match.is_recognized)

    def test_java_vs_javascript_distinct(self):
        """Test that Java and JavaScript remain strictly separated."""
        java_match = self.ontology.normalize_skill("Java")
        self.assertEqual(java_match.canonical, "Java")
        self.assertEqual(java_match.category, "Programming Languages")

        core_java = self.ontology.normalize_skill("Core Java")
        self.assertEqual(core_java.canonical, "Java")

        js_match = self.ontology.normalize_skill("JavaScript")
        self.assertEqual(js_match.canonical, "JavaScript")

        js_abbr = self.ontology.normalize_skill("JS")
        self.assertEqual(js_abbr.canonical, "JavaScript")

        self.assertNotEqual(java_match.canonical, js_match.canonical)

    def test_git_vs_github_distinct(self):
        """Test that Git and GitHub remain strictly separated."""
        git_match = self.ontology.normalize_skill("Git")
        self.assertEqual(git_match.canonical, "Git")

        github_match = self.ontology.normalize_skill("GitHub")
        self.assertEqual(github_match.canonical, "GitHub")

        self.assertNotEqual(git_match.canonical, github_match.canonical)

    def test_c_family_distinction(self):
        """Test that C, C++, and C# are not conflated."""
        # C is not in canonical list, must be preserved as unknown without matching C++ or C#
        c_match = self.ontology.normalize_skill("C")
        self.assertEqual(c_match.canonical, "C")
        self.assertFalse(c_match.is_recognized)

        cpp_match = self.ontology.normalize_skill("C++")
        self.assertEqual(cpp_match.canonical, "C++")
        self.assertTrue(cpp_match.is_recognized)

        csharp_match = self.ontology.normalize_skill("C#")
        self.assertEqual(csharp_match.canonical, "C#")
        self.assertTrue(csharp_match.is_recognized)

    def test_no_substring_false_positives(self):
        """Test that short technology tokens never trigger false-positive substring matches."""
        # 'Java' must not match inside 'JavaScript'
        self.assertEqual(self.ontology.get_canonical_name("JavaScript"), "JavaScript")
        # 'Git' must not match inside 'GitHub'
        self.assertEqual(self.ontology.get_canonical_name("GitHub"), "GitHub")
        # 'R' must not match inside 'React' or 'Rust'
        self.assertEqual(self.ontology.get_canonical_name("React"), "React")
        self.assertEqual(self.ontology.get_canonical_name("Rust"), "Rust")
        self.assertFalse(self.ontology.is_recognized("Rust"))
        # 'C' must not match inside 'CSS'
        self.assertEqual(self.ontology.get_canonical_name("CSS"), "CSS")
        # 'Go' must not match inside 'Google Cloud'
        self.assertEqual(self.ontology.get_canonical_name("Google Cloud"), "Google Cloud")

    def test_ambiguous_and_generic_terms_left_unmapped(self):
        """Test that ambiguous generic concepts are not mapped to specific tools."""
        generic_terms = [
            "AI",
            "programming",
            "database",
            "databases",
            "backend",
            "frontend",
            "ML",
            "machine learning",
            "web development",
            "ES",
        ]
        for term in generic_terms:
            match = self.ontology.normalize_skill(term)
            self.assertFalse(match.is_recognized, f"Generic term was incorrectly recognized: {term}")
            self.assertEqual(match.canonical.lower(), term.lower())
            self.assertIsNone(match.category)

    def test_unknown_skills_preserved(self):
        """Test that unknown/unmapped skills are preserved rather than dropped or forced into nearest."""
        unknowns = [
            "Rust",
            "GraphQL",
            "Solidity",
            "Next.js",
            "Spring Boot",
            "Elixir",
            "Prisma",
            "Supabase",
        ]
        for unk in unknowns:
            match = self.ontology.normalize_skill(unk)
            self.assertEqual(match.original, unk)
            self.assertEqual(match.canonical, unk)
            self.assertFalse(match.is_recognized)
            self.assertIsNone(match.category)

    def test_duplicate_aliases_resolve_to_single_canonical(self):
        """Test that multiple aliases for the same skill deduplicate to one canonical entry."""
        skills_input = [
            "React",
            "ReactJS",
            "react.js",
            "Python",
            "Python 3",
            "JS",
            "JavaScript",
            "Rust",
            "rust",
        ]
        normalized = self.ontology.normalize_skills(skills_input)
        expected = ["React", "Python", "JavaScript", "Rust"]
        self.assertEqual(normalized, expected)

    def test_empty_missing_and_incomplete_inputs(self):
        """Test handling of None, empty, and whitespace strings."""
        m_none = self.ontology.normalize_skill(None)
        self.assertEqual(m_none.canonical, "")
        self.assertFalse(m_none.is_recognized)

        m_empty = self.ontology.normalize_skill("")
        self.assertEqual(m_empty.canonical, "")
        self.assertFalse(m_empty.is_recognized)

        m_space = self.ontology.normalize_skill("   ")
        self.assertEqual(m_space.canonical, "")
        self.assertFalse(m_space.is_recognized)

        self.assertEqual(self.ontology.normalize_skills([]), [])
        self.assertEqual(self.ontology.normalize_skills([None, "", "   "]), [])

    def test_project_technology_normalization(self):
        """Test normalizing technologies on a ProjectItem without mutating original."""
        proj = ProjectItem(
            name="Career Intelligence Platform",
            description="AI career engine",
            technologies=["ReactJS", "NodeJS", "Postgres", "Docker", "Solidity"],
            link="https://github.com/example/careerlens",
        )
        norm_proj = self.ontology.normalize_project(proj)

        expected_techs = ["React", "Node.js", "PostgreSQL", "Docker", "Solidity"]
        self.assertEqual(norm_proj.technologies, expected_techs)
        self.assertEqual(norm_proj.name, "Career Intelligence Platform")

        # Verify immutability of input object
        self.assertEqual(proj.technologies, ["ReactJS", "NodeJS", "Postgres", "Docker", "Solidity"])

    def test_structured_cv_normalization(self):
        """Test normalizing full StructuredCV skills and project technologies."""
        cv = StructuredCV(
            full_name="Aarav Sharma",
            skills=["JS", "TS", "ReactJS", "FastAPI", "K8s", "Docker", "Postgres", "GraphQL"],
            projects=[
                ProjectItem(
                    name="API Service",
                    technologies=["FastAPI", "Postgres", "Redis"],
                )
            ],
            extraction_source="gemini",
        )
        norm_cv = self.ontology.normalize_cv_skills(cv)

        expected_skills = [
            "JavaScript",
            "TypeScript",
            "React",
            "FastAPI",
            "Kubernetes",
            "Docker",
            "PostgreSQL",
            "GraphQL",
        ]
        self.assertEqual(norm_cv.skills, expected_skills)
        self.assertEqual(len(norm_cv.projects), 1)
        self.assertEqual(norm_cv.projects[0].technologies, ["FastAPI", "PostgreSQL", "Redis"])

        # Verify original cv was not mutated
        self.assertEqual(cv.skills[0], "JS")
        self.assertEqual(cv.projects[0].technologies[1], "Postgres")

    def test_idempotency(self):
        """Test that normalizing already normalized skills produces identical output."""
        raw_skills = ["JS", "ReactJS", "K8s", "Postgres", "Python 3", "Rust"]
        once = self.ontology.normalize_skills(raw_skills)
        twice = self.ontology.normalize_skills(once)
        self.assertEqual(once, twice)

        proj = ProjectItem(name="Test", technologies=raw_skills)
        proj_once = self.ontology.normalize_project(proj)
        proj_twice = self.ontology.normalize_project(proj_once)
        self.assertEqual(proj_once.technologies, proj_twice.technologies)

    def test_detailed_normalization_and_provenance(self):
        """Test normalize_skills_detailed and get_skills_mapping for full provenance."""
        skills = ["ReactJS", "Solidity", "K8s"]
        detailed = self.ontology.normalize_skills_detailed(skills)
        self.assertEqual(len(detailed), 3)

        self.assertEqual(detailed[0].original, "ReactJS")
        self.assertEqual(detailed[0].canonical, "React")
        self.assertTrue(detailed[0].is_recognized)
        self.assertEqual(detailed[0].category, "Frontend")

        self.assertEqual(detailed[1].original, "Solidity")
        self.assertEqual(detailed[1].canonical, "Solidity")
        self.assertFalse(detailed[1].is_recognized)
        self.assertIsNone(detailed[1].category)

        self.assertEqual(detailed[2].original, "K8s")
        self.assertEqual(detailed[2].canonical, "Kubernetes")
        self.assertTrue(detailed[2].is_recognized)
        self.assertEqual(detailed[2].category, "Cloud & Infrastructure")

        mapping = self.ontology.get_skills_mapping(skills)
        self.assertIn("ReactJS", mapping)
        self.assertIn("Solidity", mapping)
        self.assertIn("K8s", mapping)
        self.assertEqual(mapping["ReactJS"].canonical, "React")


if __name__ == "__main__":
    unittest.main()
