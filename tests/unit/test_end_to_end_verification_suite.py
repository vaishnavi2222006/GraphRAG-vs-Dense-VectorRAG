"""Automated Pytest Regression Suite for End-to-End Workflows 1-9.

Verifies:
1. Document Ingestion (extraction, chunking, embeddings, vector indexing, graph indexing)
2. Vector RAG (factual passage retrieval)
3. GraphRAG (multi-hop traversal, entity relationships, chunk hydration)
4. Hybrid RAG (RRF combination, no displacement of strong vector results)
5. Answer Generation (evidence grounding, exact facts, offline fallback labeling)
6. Comparison (side-by-side execution of Vector, Graph, and Hybrid on same question)
7. Evaluation (BenchmarkRunner on ground truth dataset against active corpus)
8. Corpus Isolation (conflicting facts in two separate knowledge bases, zero cross-contamination)
9. Absent Topic Honesty (system declines answering unsupported queries)
"""
from __future__ import annotations

from pathlib import Path
import pytest

from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.schemas.evaluation import EvaluationExample
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.generation.llm_service import OllamaLLMService
from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
from rag_benchmark_lab.retrieval.hybrid_retriever import HybridRetriever
from rag_benchmark_lab.retrieval.vector_retriever import VectorRetriever
from rag_benchmark_lab.schemas.documents import DocumentStatus, RetrievalStrategy
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.document_service import DocumentService
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.services.user_registry import UserRegistry
from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore

DOC_QUANTUM_COMPUTING = """
Quantum Dynamics Inc. Technical Brief — October 2024

Quantum Dynamics Inc., headquartered in Zurich, Switzerland, was founded by Dr. Aris Thorne.
In September 2024, the company unveiled the Hyperion-9 superconducting quantum processor featuring 1,124 physical qubits.
The Hyperion-9 achieved a two-qubit gate fidelity of 99.85%, surpassing the previous industry record.
The project received a development grant of $145 million from the European Advanced Research Initiative.
Dr. Thorne collaborated with Chief Engineer Linnea Lindqvist to develop the cryostat cooling system, which maintains an operating temperature of 15 millikelvin.
""".strip()

DOC_QUANTUM_ALGORITHMS = """
Hyperion-9 Application Benchmarks — Quantum Dynamics & Helvetia Labs, 2024

Linnea Lindqvist announced that Hyperion-9 executed Shor's algorithm for 2048-bit integer factorization simulation in 4.2 hours.
The superconducting processor utilizes niobium-titanium superconducting resonators fabricated at the ETH Zurich Nanotechnology Facility.
Quantum Dynamics Inc. entered a strategic technology partnership with Helvetia Cloud Services to provide enterprise API access starting in Q1 2025.
Dr. Aris Thorne stated that quantum error mitigation reduced algorithmic decoherence rates by 74%.
""".strip()

DOC_BIOMEDICAL_NEURO = """
Clinical Trial Phase II Summary — Synapse Therapeutics, Boston MA

Synapse Therapeutics is conducting clinical trials for Synaptin-7, an oral small-molecule BACE1 inhibitor designed for early-stage Alzheimer's disease.
Dr. Miriam Vance directs the clinical trials at Massachusetts General Hospital.
The trial enrolled 840 patients across 14 clinical sites.
Patients receiving a daily dose of 50 mg exhibited a 38.6% reduction in cortical amyloid-beta plaques over a 48-week period.
Synapse Therapeutics was acquired by Global Pharma Corp in June 2023 for $1.85 billion.
""".strip()

DOC_CORPUS_A_ORION = """
Project Orion Mission Report — Alpha Aerospace Syndicate, Houston TX

Project Orion was officially launched in March 2021 by Commander Sarah Jenkins.
The exploration program operated with an annual research budget of $450 million in Houston, Texas.
Commander Jenkins confirmed that the Deep Space Habitat successfully completed 1,200 orbital cycles.
""".strip()

DOC_CORPUS_B_ORION = """
Project Orion Internal Dossier — Beta Robotics Collective, Berlin Germany

Project Orion was officially launched in November 2025 by Dr. Marcus Thorne.
The robotic automation program operated with an annual research budget of $80 million in Berlin, Germany.
Dr. Marcus Thorne confirmed that the Autonomous Rover Fleet deployed 35 exploration probes.
""".strip()


@pytest.fixture
def e2e_env(tmp_path: Path):
    db_path = tmp_path / "test_docs.sqlite3"
    user_db_path = tmp_path / "test_users.sqlite3"

    registry = DocumentRegistry(db_path)
    user_registry = UserRegistry(user_db_path)
    emb_service = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
    qdrant = QdrantStore(":memory:", None, "e2e_collection")
    qdrant.ensure_collection(emb_service.dimension)
    graph_store = NetworkXStore()
    llm_service = OllamaLLMService("http://localhost:11434", "llama3.1:8b", timeout_seconds=5, retries=0, mock_mode=True)
    extractor = EntityRelationshipExtractor(llm_service)
    doc_service = DocumentService(
        registry=registry,
        embedding_service=emb_service,
        qdrant_store=qdrant,
        neo4j_store=graph_store,
        extractor=extractor,
        chunk_size=400,
        chunk_overlap=60,
        upload_size_limit_mb=10,
    )
    vector_retriever = VectorRetriever(emb_service, qdrant, top_k=5)
    graph_retriever = GraphRetriever(graph_store, max_hops=2, max_results=20, registry=registry)
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, fusion_k=60)
    query_service = QueryService(
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        hybrid_retriever=hybrid_retriever,
        llm_service=llm_service,
        top_k=5,
        answer_temperature=0.0,
    )
    result_store = ResultStore(registry)

    test_user, errs = user_registry.register_user("E2E Tester", "tester@example.com", "SecurePass123!")
    assert test_user is not None, f"Failed to register user: {errs}"
    primary_kb = user_registry.create_knowledge_base(test_user.user_id, "Quantum & Bio Research", "Primary KB")

    # Ingest primary documents
    docs_to_ingest = [
        ("quantum_brief.txt", DOC_QUANTUM_COMPUTING),
        ("quantum_benchmarks.txt", DOC_QUANTUM_ALGORITHMS),
        ("synapse_trial.txt", DOC_BIOMEDICAL_NEURO),
    ]
    primary_doc_ids = []
    for name, text in docs_to_ingest:
        outcome = doc_service.ingest_bytes(name, text.encode("utf-8"))
        assert outcome.document.status == DocumentStatus.INDEXED
        user_registry.add_document_to_kb(primary_kb.kb_id, outcome.document.document_id)
        primary_doc_ids.append(outcome.document.document_id)

    return {
        "registry": registry,
        "user_registry": user_registry,
        "test_user": test_user,
        "primary_kb": primary_kb,
        "primary_doc_ids": primary_doc_ids,
        "doc_service": doc_service,
        "query_service": query_service,
        "result_store": result_store,
        "graph_store": graph_store,
    }


def test_workflow_1_ingestion(e2e_env):
    """Verify document ingestion, text extraction, chunking, embeddings, entities, and relationships."""
    assert len(e2e_env["primary_doc_ids"]) == 3
    assert len(e2e_env["graph_store"]._entities) >= 20
    assert len(e2e_env["graph_store"]._relationships) >= 20


def test_workflow_2_vector_rag(e2e_env):
    """Verify vector retrieval retrieves factual evidence."""
    qs = e2e_env["query_service"]
    q = "What is the physical qubit count and two-qubit gate fidelity of the Hyperion-9 processor?"
    res = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=e2e_env["primary_doc_ids"])
    assert res.retrieved_sources
    text = " ".join(c.text for c in res.retrieved_sources)
    assert "1,124" in text or "1124" in text
    assert "99.85%" in text or "99.85" in text


def test_workflow_3_graphrag_traversal(e2e_env):
    """Verify GraphRAG traverses entities and hydrates full chunks."""
    qs = e2e_env["query_service"]
    q = "Who is Dr. Aris Thorne and what role does Linnea Lindqvist play at Quantum Dynamics?"
    res = qs.answer(q, manual_strategy=RetrievalStrategy.GRAPH, document_ids=e2e_env["primary_doc_ids"])
    assert res.retrieved_sources or (res.graph_evidence and res.graph_evidence.entities)
    assert any(len(c.text) > 50 for c in res.retrieved_sources)


def test_workflow_4_hybrid_rag_fusion(e2e_env):
    """Verify Hybrid RAG fuses vector and graph results using RRF."""
    qs = e2e_env["query_service"]
    q = "How long did Shor's algorithm simulation take on Hyperion-9 and what organization provides cloud access?"
    res = qs.answer(q, manual_strategy=RetrievalStrategy.HYBRID, document_ids=e2e_env["primary_doc_ids"])
    assert res.retrieved_sources
    text = " ".join(c.text for c in res.retrieved_sources)
    assert "4.2 hours" in text or "4.2" in text
    assert "Helvetia" in text


def test_workflow_5_answer_generation_and_offline_label(e2e_env):
    """Verify answers preserve exact figures and clearly state offline mode."""
    qs = e2e_env["query_service"]
    q = "What dosage of Synaptin-7 was administered in the clinical trial and what percentage reduction in amyloid-beta plaques was observed?"
    res = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=e2e_env["primary_doc_ids"])
    assert "50 mg" in res.answer or "50" in res.answer
    assert "38.6%" in res.answer or "38.6" in res.answer
    assert "[Offline / Development Mode" in res.answer or "offline" in res.answer.lower() or "unreachable" in res.answer.lower()


def test_workflow_6_comparison(e2e_env):
    """Verify side-by-side comparison across Vector, Graph, and Hybrid."""
    qs = e2e_env["query_service"]
    q = "Explain the cryostat cooling system for Hyperion-9 and its operating temperature."
    results = qs.compare(q, document_ids=e2e_env["primary_doc_ids"])
    assert "VECTOR" in results and "GRAPH" in results and "HYBRID" in results
    for strat in ["VECTOR", "GRAPH", "HYBRID"]:
        res = results[strat]
        assert res.answer
        assert res.total_latency_ms is not None
        assert res.retrieval_latency_ms is not None


def test_workflow_7_evaluation_runner(e2e_env):
    """Verify benchmark evaluation computes metrics and persists results."""
    runner = BenchmarkRunner(e2e_env["query_service"], e2e_env["result_store"])
    examples = [
        EvaluationExample(
            query_id="eval-test-1",
            question="What is the operating temperature of the Hyperion-9 cryostat cooling system?",
            expected_answer="15 millikelvin",
            relevant_document_ids=e2e_env["primary_doc_ids"][:1],
            reference_contexts=["The cryostat cooling system maintains an operating temperature of 15 millikelvin."],
        )
    ]
    output = runner.run(examples, strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID], document_ids=e2e_env["primary_doc_ids"])
    assert "run_id" in output
    assert len(output["rows"]) == 2
    assert "VECTOR" in output["aggregates"]
    assert "HYBRID" in output["aggregates"]


def test_workflow_8_corpus_isolation(e2e_env):
    """Verify strict corpus isolation between conflicting knowledge bases."""
    doc_service = e2e_env["doc_service"]
    user_reg = e2e_env["user_registry"]
    test_user = e2e_env["test_user"]
    qs = e2e_env["query_service"]

    kb_a = user_reg.create_knowledge_base(test_user.user_id, "Corpus A", "Mission A")
    kb_b = user_reg.create_knowledge_base(test_user.user_id, "Corpus B", "Mission B")

    out_a = doc_service.ingest_bytes("orion_alpha.txt", DOC_CORPUS_A_ORION.encode("utf-8"))
    out_b = doc_service.ingest_bytes("orion_beta.txt", DOC_CORPUS_B_ORION.encode("utf-8"))
    user_reg.add_document_to_kb(kb_a.kb_id, out_a.document.document_id)
    user_reg.add_document_to_kb(kb_b.kb_id, out_b.document.document_id)

    q = "Who launched Project Orion, in what year, and with what annual budget?"

    res_a = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=[out_a.document.document_id])
    assert "Sarah Jenkins" in res_a.answer or "450 million" in res_a.answer
    assert "Marcus Thorne" not in res_a.answer
    assert "Berlin" not in res_a.answer
    assert "80 million" not in res_a.answer

    res_b = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=[out_b.document.document_id])
    assert "Marcus Thorne" in res_b.answer or "80 million" in res_b.answer
    assert "Sarah Jenkins" not in res_b.answer
    assert "Houston" not in res_b.answer
    assert "450 million" not in res_b.answer

    # Empty corpus check
    empty_kb = user_reg.create_knowledge_base(test_user.user_id, "Empty KB", "Empty")
    empty_doc_ids = user_reg.get_kb_document_ids(empty_kb.kb_id, test_user.user_id)
    res_empty = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=empty_doc_ids)
    assert len(res_empty.citations) == 0
    assert "no documents" in res_empty.answer.lower() or "empty" in res_empty.answer.lower()


def test_workflow_9_absent_topic_honesty(e2e_env):
    """Verify system honestly states lack of evidence for unsupported topics."""
    qs = e2e_env["query_service"]
    q = "What are the rules of curling and how much does a curling stone weigh in kilograms?"
    res = qs.answer(q, manual_strategy=RetrievalStrategy.VECTOR, document_ids=e2e_env["primary_doc_ids"])
    assert "not appear to contain sufficient evidence" in res.answer.lower() or "insufficient" in res.answer.lower()
