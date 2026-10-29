"""
Targeted tests for retrieval mechanics, ranking, and threshold boundaries.

This module validates:
1. Retrieval precision: Project-specific queries retrieve project chunks.
2. Experience-specific queries retrieve work experience chunks.
3. Ranking separation: Relevant chunks rank higher than unrelated chunks.
4. Synonymous wording: Queries using alternate vocabulary maintain retrieval relevance.
5. Absent information: Queries for non-existent domains score below threshold and trigger safe handling.
6. Vector-store mechanics vs. true neural semantic retrieval:
   - Synthetic and term-hash projections verify the indexing, cosine similarity math,
     top-k ranking, and threshold rejection mechanics.
   - True semantic generalization across complex natural language requires neural
     embedding models (e.g., SentenceTransformers / BGE), which is explicitly decoupled here.
"""

import unittest
from typing import List
from unittest.mock import MagicMock
import numpy as np

from backend.models.schemas import ResumeChunk
from backend.services.embedder import BaseEmbedder
from backend.services.retriever import ResumeRetriever
from backend.services.resume_agent import ResumeIntelligenceAgent
from backend.services.vector_store import InMemoryVectorStore


class ControlledSemanticConceptEmbedder(BaseEmbedder):
    """
    A controlled deterministic embedder for rigorous retrieval testing.

    Maps semantic concepts to designated orthogonal subspaces (dimension=128):
    - Subspace 0..31: Healthcare / Clinical Analytics / FHIR / Medical Data
    - Subspace 32..63: FinTech / Payment Processing / Stripe / Transactions / Ledger
    - Subspace 64..95: Cloud Infrastructure / Kubernetes / DevOps / CI-CD
    - Subspace 96..127: Education / University / Computer Science / Degree

    This allows testing synonym generalization and semantic subspace separation
    without downloading live neural network weights.
    """

    CONCEPT_CLUSTERS = {
        "healthcare": (0, 32, [
            "healthcare", "hospital", "clinical", "fhir", "patient", "medical",
            "telehealth", "ehr", "hl7", "analytics dashboard", "health", "doctor"
        ]),
        "fintech": (32, 64, [
            "fintech", "payment", "payments", "transaction", "transactions", "stripe",
            "billing", "ledger", "checkout", "processing pipeline", "financial", "gateway"
        ]),
        "infrastructure": (64, 96, [
            "infrastructure", "kubernetes", "k8s", "docker", "devops", "cloud",
            "terraform", "aws", "gcp", "scalability", "microservices"
        ]),
        "education": (96, 128, [
            "education", "university", "bachelor", "master", "degree", "gpa",
            "computer science", "academic", "thesis", "graduated"
        ]),
    }

    def __init__(self, dimension: int = 128):
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_text(self, text: str) -> List[float]:
        return self.embed_batch([text])[0]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        results: List[List[float]] = []
        for text in texts:
            clean = text.lower() if text else ""
            vec = np.zeros(self._dimension, dtype=np.float32)

            for concept, (start, end, keywords) in self.CONCEPT_CLUSTERS.items():
                match_count = sum(1 for kw in keywords if kw in clean)
                if match_count > 0:
                    # Activate concept subspace
                    sub_weight = min(1.0, 0.4 + 0.15 * match_count)
                    vec[start:end] += sub_weight

            # Add minor deterministic perturbation for uniqueness
            rng = np.random.RandomState(abs(hash(clean)) % (2**31))
            vec += rng.randn(self._dimension) * 0.01

            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec = vec / norm
            else:
                vec = np.zeros(self._dimension, dtype=np.float32)

            results.append(vec.tolist())
        return results


class TestRetrievalMechanics(unittest.TestCase):
    """Rigorous tests evaluating retrieval mechanics, ranking, and threshold boundaries."""

    def setUp(self):
        self.embedder = ControlledSemanticConceptEmbedder(dimension=128)
        self.vector_store = InMemoryVectorStore()
        self.retriever = ResumeRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
        )
        self.resume_id = "test_semantic_resume_001"

        # Populate a rich synthetic resume with distinct sections
        self.chunks = [
            ResumeChunk(
                chunk_id=f"{self.resume_id}_c1_summary",
                resume_id=self.resume_id,
                chunk_index=0,
                section="summary",
                item_title="Professional Summary",
                content="Senior Software Engineer with 7 years of experience in distributed systems, fintech payment gateways, and cloud infrastructure.",
                page_number=1,
                metadata={"section": "summary"},
            ),
            ResumeChunk(
                chunk_id=f"{self.resume_id}_c2_exp_stripe",
                resume_id=self.resume_id,
                chunk_index=1,
                section="work_experience",
                item_title="Senior Engineer at Stripe",
                content="Engineered high-throughput payment processing pipelines handling $50M daily transactions. Reduced payment settlement latency by 35% through ledger optimization.",
                page_number=1,
                metadata={"section": "work_experience", "employer": "Stripe"},
            ),
            ResumeChunk(
                chunk_id=f"{self.resume_id}_c3_project_health",
                resume_id=self.resume_id,
                chunk_index=2,
                section="projects",
                item_title="Healthcare Analytics Dashboard",
                content="Developed a real-time clinical analytics dashboard for hospital ICUs ingesting FHIR patient vitals and generating automated medical triage reports.",
                page_number=2,
                metadata={"section": "projects", "tech_stack": ["FHIR", "React", "Python"]},
            ),
            ResumeChunk(
                chunk_id=f"{self.resume_id}_c4_edu",
                resume_id=self.resume_id,
                chunk_index=3,
                section="education",
                item_title="B.S. in Computer Science",
                content="Bachelor of Science in Computer Science from State University. Graduated Magna Cum Laude with focus on distributed algorithms.",
                page_number=2,
                metadata={"section": "education", "degree": "B.S."},
            ),
            ResumeChunk(
                chunk_id=f"{self.resume_id}_c5_infra",
                resume_id=self.resume_id,
                chunk_index=4,
                section="skills",
                item_title="Cloud & DevOps Skills",
                content="Kubernetes cluster orchestration, Docker containerization, AWS cloud architecture, and automated CI/CD deployment pipelines.",
                page_number=2,
                metadata={"section": "skills"},
            ),
        ]

        self.retriever.index_resume(self.resume_id, self.chunks)

    def test_project_query_retrieves_correct_project_chunk(self):
        """Query asking specifically about the healthcare dashboard must rank the project chunk first."""
        query = "Tell me about the candidate's healthcare analytics dashboard and FHIR clinical reports."
        results = self.retriever.retrieve(self.resume_id, query, top_k=3)

        self.assertTrue(len(results) >= 1)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk.chunk_id, f"{self.resume_id}_c3_project_health")
        self.assertEqual(top_chunk.section, "projects")
        self.assertIn("Healthcare Analytics Dashboard", top_chunk.item_title)
        self.assertGreater(top_score, 0.70)

    def test_work_experience_query_retrieves_relevant_experience_chunk(self):
        """Query asking about payment transactions and Stripe must rank the experience chunk first."""
        query = "What did the candidate accomplish with payment processing pipelines at Stripe?"
        results = self.retriever.retrieve(self.resume_id, query, top_k=3)

        self.assertTrue(len(results) >= 1)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk.chunk_id, f"{self.resume_id}_c2_exp_stripe")
        self.assertEqual(top_chunk.section, "work_experience")
        self.assertIn("Stripe", top_chunk.item_title)
        self.assertGreater(top_score, 0.70)

    def test_unrelated_chunks_rank_substantially_lower_than_relevant_chunks(self):
        """Relevant chunks must have a significant similarity margin over unrelated chunks."""
        query = "Details on patient vitals and hospital ICU medical records."
        results = self.retriever.retrieve(self.resume_id, query, top_k=5)

        # The healthcare project should be top
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk.chunk_id, f"{self.resume_id}_c3_project_health")

        # Unrelated chunks (education, devops, etc.) should have substantially lower similarity
        for chunk, score in results[1:]:
            margin = top_score - score
            self.assertGreater(margin, 0.35, f"Expected margin > 0.35 between relevant and unrelated chunk, got {margin:.3f}")

    def test_synonymous_wording_maintains_relevant_retrieval(self):
        """Alternate phrasing ('financial transaction APIs' vs 'payment settlement') retrieves the Stripe chunk."""
        synonym_query = "Experience building financial transaction APIs and ledger checkout gateways."
        results = self.retriever.retrieve(self.resume_id, synonym_query, top_k=3)

        self.assertTrue(len(results) >= 1)
        top_chunk, top_score = results[0]
        self.assertEqual(top_chunk.chunk_id, f"{self.resume_id}_c2_exp_stripe")
        self.assertGreater(top_score, 0.60)

    def test_absent_information_handled_safely_by_agent(self):
        """
        A query for an absent domain (e.g. quantum cryptography) has low similarity.
        When passed to ResumeIntelligenceAgent with min_similarity_threshold=0.25,
        the agent returns has_sufficient_evidence=False without hallucination.
        """
        from backend.models.schemas import ResumeQueryRequest
        mock_gemini = MagicMock()
        agent = ResumeIntelligenceAgent(
            retriever=self.retriever,
            gemini_service=mock_gemini,
            min_similarity_threshold=0.25,
        )

        absent_query = "What experience does the candidate have in quantum cryptography and quantum key distribution?"
        request = ResumeQueryRequest(resume_id=self.resume_id, query=absent_query)
        response = agent.answer_query(request)

        self.assertFalse(response.has_sufficient_evidence)
        self.assertEqual(len(response.citations), 0)
        self.assertIn("does not contain information", response.answer)
        # Verify LLM was NOT called when retrieval evidence is absent
        mock_gemini.generate_content.assert_not_called()

    def test_vector_store_mechanics_vs_semantic_evaluation_distinction(self):
        """
        Document and verify the architectural boundary:
        Synthetic embeddings prove that vector-store indexing, cosine dot-product math,
        L2 normalization, and threshold filtering are computationally correct.
        However, synthetic embeddings do NOT prove real-world natural language comprehension,
        which depends on the neural weights of the configured SentenceTransformer model.
        """
        vec = self.embedder.embed_text("Sample input query")
        norm = np.linalg.norm(vec)
        self.assertAlmostEqual(norm, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
