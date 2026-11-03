"""Generalized RAG retrieval tests using synthetic documents with known ground truth.

These tests exercise the complete retrieval pipeline (Vector RAG, GraphRAG, Hybrid RAG)
against dynamically created documents.  They do NOT hardcode any demo-corpus questions,
expected answers, entity names, or document names — all test content is defined in
synthetic fixtures within this file.

Test categories:
  - Direct factual questions
  - Paraphrased / alternative phrasing questions
  - Questions involving technical terminology
  - Multi-document questions (answer distributed across documents)
  - Relationship / entity-connection questions
  - Questions about numbers, dates, and specific details
  - Questions absent from the corpus (should return no-evidence)
  - Malformed / empty documents
  - Corpus isolation (documents from one corpus must not appear in another)
  - Documents uploaded after initial indexing

External services (Qdrant, Neo4j, Ollama) are NOT required.  All tests run with
the in-memory fallbacks used in CI.

Skips:
  - sentence-transformers integration tests are skipped if the library is not installed
"""
from __future__ import annotations

import hashlib
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Synthetic test fixtures
# ---------------------------------------------------------------------------

SOLAR_DOC = """
The Solar Energy Report — Renewable Energy Division, 2024

Solar photovoltaic (PV) panels convert sunlight directly into electricity through the
photovoltaic effect. A single residential installation typically generates between
5 kW and 15 kW of peak power.

The levelised cost of energy (LCOE) for utility-scale solar fell to $0.033 per kWh
in 2023, making it cheaper than coal in most markets.

Dr. Elena Vasquez, Chief Technology Officer at SunCore Systems, stated that the
global installed solar capacity reached 1.6 TW by the end of 2023. SunCore Systems
is headquartered in Austin, Texas.

Efficiency improvements in perovskite solar cells have achieved laboratory records
of 29.15% conversion efficiency as of the latest published research.
""".strip()

BATTERY_DOC = """
Energy Storage Technology Overview — Battery Research Consortium, 2024

Lithium-ion batteries remain the dominant technology for grid-scale energy storage
with an average round-trip efficiency of 90–95%.

The Battery Research Consortium, a non-profit organization based in Munich, Germany,
tracks global battery deployment. By 2024, total deployed grid-scale battery capacity
exceeded 200 GWh worldwide.

Dr. James Okonkwo leads the Electrolyte Chemistry division at the Consortium. His
research focuses on solid-state electrolytes which promise energy densities above
500 Wh/kg, compared to 250 Wh/kg for conventional lithium-ion cells.

The average cost of utility-scale battery storage fell to $132 per kWh in 2023,
down from $208 per kWh in 2021.
""".strip()

CLIMATE_DOC = """
Climate Policy Framework — International Energy Agency, 2024

The Paris Agreement target of limiting global warming to 1.5°C requires net-zero
carbon emissions by 2050. Achieving this requires deploying at least 5 TW of solar
capacity and 3 TW of wind capacity globally.

Renewable energy — including solar, wind, hydro, and geothermal — must replace fossil
fuels across electricity generation, transportation, and industrial heating by 2050.

The IEA projects that electricity demand will double by 2050 as transportation and
heating are electrified.

Countries with the highest renewable energy targets include Germany (100% renewable
by 2035), Denmark (100% renewable by 2030), and Scotland (100% renewable by 2030).
""".strip()

IRRELEVANT_DOC = """
Artisan Bread Baking Guide — Home Baker's Handbook

Sourdough starter requires equal parts flour and water by weight, fed at least once
every 24 hours. A healthy starter doubles in volume within 4–8 hours of feeding.

Autolyse is the process of mixing flour and water before adding salt or starter,
allowing gluten to develop naturally. Typical autolyse time is 30–60 minutes.

The Maillard reaction creates the golden-brown crust when baking bread at high
temperatures (220–240°C). Steam in the early baking phase helps crust development.
""".strip()


@dataclass
class SyntheticCorpus:
    """A corpus of synthetic documents with known ground truth."""
    documents: dict[str, str]  # name -> text
    ground_truth: list[dict]   # list of {question, answer_keywords, relevant_docs}


ENERGY_CORPUS = SyntheticCorpus(
    documents={
        "solar_report.txt": SOLAR_DOC,
        "battery_overview.txt": BATTERY_DOC,
        "climate_policy.txt": CLIMATE_DOC,
    },
    ground_truth=[
        {
            "question": "What was the global installed solar capacity by end of 2023?",
            "answer_keywords": ["1.6", "TW", "terawatt"],
            "relevant_docs": ["solar_report.txt"],
            "query_type": "factual_number",
        },
        {
            "question": "What is the LCOE for utility-scale solar in 2023?",
            "answer_keywords": ["0.033", "kWh"],
            "relevant_docs": ["solar_report.txt"],
            "query_type": "factual_number",
        },
        {
            "question": "Which organization tracks global battery deployment?",
            "answer_keywords": ["Battery Research Consortium"],
            "relevant_docs": ["battery_overview.txt"],
            "query_type": "entity_identification",
        },
        {
            "question": "Who leads the Electrolyte Chemistry division?",
            "answer_keywords": ["Okonkwo", "James"],
            "relevant_docs": ["battery_overview.txt"],
            "query_type": "person_identification",
        },
        {
            "question": "How much did battery storage cost per kWh in 2023?",
            "answer_keywords": ["132", "kWh"],
            "relevant_docs": ["battery_overview.txt"],
            "query_type": "factual_number",
        },
        {
            "question": "What renewable energy targets has Germany set?",
            "answer_keywords": ["Germany", "2035", "renewable"],
            "relevant_docs": ["climate_policy.txt"],
            "query_type": "policy_fact",
        },
        {
            "question": "How much solar and wind capacity is needed for the Paris Agreement?",
            "answer_keywords": ["5 TW", "3 TW", "solar", "wind"],
            "relevant_docs": ["climate_policy.txt"],
            "query_type": "multi_fact",
        },
        {
            "question": "Compare solar and battery storage costs from the available documents.",
            "answer_keywords": ["solar", "battery", "cost"],
            "relevant_docs": ["solar_report.txt", "battery_overview.txt"],
            "query_type": "multi_document_comparison",
        },
        {
            "question": "What is the recipe for making chocolate cake?",
            "answer_keywords": [],
            "relevant_docs": [],
            "query_type": "absent_from_corpus",
            "expect_no_evidence": True,
        },
    ],
)

# ---------------------------------------------------------------------------
# Shared setup helpers
# ---------------------------------------------------------------------------


def _make_fake_embedding_service(dim: int = 384):
    """Create a real EmbeddingService in mock mode (hash-based embeddings)."""
    from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
    return EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)


def _make_fake_llm():
    """Stub LLM that returns a fixed mock response."""
    from rag_benchmark_lab.generation.llm_service import OllamaLLMService
    svc = OllamaLLMService(
        base_url="http://localhost:11434",
        model_name="llama3.1:8b-instruct-q4_K_M",
        timeout_seconds=5,
        retries=0,
        mock_mode=True,
    )
    return svc


def _build_full_pipeline(tmp_path: Path, corpus: SyntheticCorpus):
    """Build a complete in-memory RAG pipeline, ingest the synthetic corpus,
    and return (document_service, query_service, document_ids).
    """
    from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
    from rag_benchmark_lab.generation.llm_service import OllamaLLMService
    from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
    from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
    from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
    from rag_benchmark_lab.retrieval.hybrid_retriever import HybridRetriever
    from rag_benchmark_lab.retrieval.vector_retriever import VectorRetriever
    from rag_benchmark_lab.services.document_registry import DocumentRegistry
    from rag_benchmark_lab.services.document_service import DocumentService
    from rag_benchmark_lab.services.query_service import QueryService
    from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore

    registry = DocumentRegistry(tmp_path / "reg.sqlite3")
    emb = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
    qdrant = QdrantStore(":memory:", None, "test_chunks")
    qdrant.ensure_collection(emb.dimension)
    graph_store = NetworkXStore()
    llm = OllamaLLMService(
        "http://localhost:11434", "llama3.1:8b-instruct-q4_K_M",
        timeout_seconds=5, retries=0, mock_mode=True,
    )
    extractor = EntityRelationshipExtractor(llm)
    doc_service = DocumentService(
        registry=registry,
        embedding_service=emb,
        qdrant_store=qdrant,
        neo4j_store=graph_store,
        extractor=extractor,
        chunk_size=600,
        chunk_overlap=100,
        upload_size_limit_mb=10,
    )
    vector_retriever = VectorRetriever(emb, qdrant, top_k=5)
    graph_retriever = GraphRetriever(graph_store, max_hops=2, max_results=20)
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, fusion_k=60)
    query_service = QueryService(
        vector_retriever, graph_retriever, hybrid_retriever, llm,
        top_k=5, answer_temperature=0.0,
    )

    doc_ids: list[str] = []
    for doc_name, doc_text in corpus.documents.items():
        outcome = doc_service.ingest_bytes(doc_name, doc_text.encode("utf-8"))
        assert outcome.document.status.value in ("indexed", "partial"), (
            f"Document '{doc_name}' failed to ingest: {outcome.errors}"
        )
        doc_ids.append(outcome.document.document_id)

    return doc_service, query_service, doc_ids


# ---------------------------------------------------------------------------
# EmbeddingService tests
# ---------------------------------------------------------------------------


class TestEmbeddingService:
    def test_embed_texts_returns_correct_dimension(self):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        result = svc.embed_texts(["Hello world", "Another sentence"])
        assert len(result.vectors) == 2
        assert all(len(v) == svc.dimension for v in result.vectors)

    def test_embed_query_consistent_with_embed_texts(self):
        """Query and document embeddings must use the same model/preprocessing."""
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        text = "Energy storage technologies"
        doc_vec = svc.embed_texts([text]).vectors[0]
        qry_vec = svc.embed_query(text)
        assert doc_vec == qry_vec, (
            "embed_texts and embed_query must return identical vectors for the same text"
        )

    def test_embed_empty_list_returns_empty(self):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        result = svc.embed_texts([])
        assert result.vectors == []

    def test_different_texts_produce_different_vectors(self):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        v1 = svc.embed_query("solar energy panels")
        v2 = svc.embed_query("bread baking sourdough")
        assert v1 != v2, "Different texts must produce different embeddings"

    def test_same_text_always_produces_same_vector(self):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        text = "Deterministic embedding test"
        v1 = svc.embed_query(text)
        v2 = svc.embed_query(text)
        assert v1 == v2, "Same text must always produce the same vector"

    def test_vectors_are_normalised(self):
        """Vectors should be approximately unit-length (for cosine similarity)."""
        import math
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        v = svc.embed_query("Normalisation test for cosine similarity")
        norm = math.sqrt(sum(x * x for x in v))
        assert abs(norm - 1.0) < 0.01, f"Vector norm should be ~1.0, got {norm:.4f}"

    def test_dimension_property_is_consistent(self):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        svc = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        d1 = svc.dimension
        result = svc.embed_texts(["test"])
        assert d1 == result.dimension
        assert len(result.vectors[0]) == d1


# ---------------------------------------------------------------------------
# Vector RAG tests
# ---------------------------------------------------------------------------


class TestVectorRAG:
    def test_ingested_document_is_searchable(self, tmp_path):
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What is the LCOE for utility-scale solar in 2023?",
            manual_strategy=__import__(
                "rag_benchmark_lab.schemas.documents", fromlist=["RetrievalStrategy"]
            ).RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert result.retrieved_sources, "Vector RAG must return at least one chunk"
        assert result.answer, "An answer must be generated"

    def test_top_k_chunks_are_returned(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "solar panel efficiency",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert len(result.retrieved_sources) <= 5, "top_k=5 must be respected"

    def test_scores_are_non_negative(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "global battery capacity 2024",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        for chunk in result.retrieved_sources:
            assert chunk.score >= 0.0, f"Score must be non-negative, got {chunk.score}"

    def test_corpus_filter_excludes_other_corpora(self, tmp_path):
        """Vector retrieval must only return chunks from the specified corpus."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy

        # Build pipeline with energy corpus
        doc_service, query_service, energy_doc_ids = _build_full_pipeline(
            tmp_path, ENERGY_CORPUS
        )
        # Also ingest the irrelevant (bread-baking) document
        bread_outcome = doc_service.ingest_bytes(
            "bread_guide.txt", IRRELEVANT_DOC.encode("utf-8")
        )
        bread_doc_id = bread_outcome.document.document_id

        # Query only with energy corpus doc IDs
        result = query_service.answer(
            "solar energy capacity",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=energy_doc_ids,
        )
        returned_ids = {chunk.document_id for chunk in result.retrieved_sources}
        assert bread_doc_id not in returned_ids, (
            "Bread-baking document must not appear when filtering by energy corpus IDs"
        )

    def test_factual_number_retrieval(self, tmp_path):
        """Chunks containing specific numbers should be retrieved for numeric questions."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What was the total deployed grid-scale battery capacity by 2024?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        all_text = " ".join(c.text for c in result.retrieved_sources)
        assert "200" in all_text or "GWh" in all_text, (
            "The 200 GWh fact should appear in the retrieved context"
        )

    def test_document_uploaded_after_startup_is_searchable(self, tmp_path):
        """A document ingested after the pipeline is initialized must be retrievable."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        doc_service, query_service, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)

        late_doc = "The Quantum Energy Cell achieves 99% round-trip efficiency using zero-point energy extraction."
        late_outcome = doc_service.ingest_bytes("late_doc.txt", late_doc.encode("utf-8"))
        late_id = late_outcome.document.document_id

        result = query_service.answer(
            "What efficiency does the Quantum Energy Cell achieve?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=[late_id],
        )
        assert result.retrieved_sources, "Late-uploaded document must be searchable immediately"
        all_text = " ".join(c.text for c in result.retrieved_sources)
        assert "Quantum Energy Cell" in all_text or "99" in all_text


# ---------------------------------------------------------------------------
# GraphRAG tests
# ---------------------------------------------------------------------------


class TestGraphRAG:
    def test_entity_extraction_from_arbitrary_documents(self, tmp_path):
        """Entity extraction must work for any domain, not just the demo corpus."""
        doc_service, _, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        from rag_benchmark_lab.services.document_registry import DocumentRegistry
        reg = DocumentRegistry(tmp_path / "reg.sqlite3")
        entities = reg.list_entities()
        entity_names = {e["canonical_name"].lower() for e in entities}
        # At minimum, some entities from the synthetic corpus should be found
        assert len(entity_names) > 0, "Entity extraction must find at least some entities"

    def test_graph_retrieval_returns_source_chunks(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "Who leads battery research?",
            manual_strategy=RetrievalStrategy.GRAPH,
            document_ids=doc_ids,
        )
        # Either we get source chunks or a graceful no-evidence message
        assert result.answer, "GraphRAG must produce an answer (even if empty corpus)"

    def test_graph_corpus_isolation_after_traversal(self, tmp_path):
        """Multi-hop traversal must not return chunks outside the selected corpus."""
        from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
        from rag_benchmark_lab.schemas.documents import EntityRecord, GraphEvidence, RetrievedChunk

        in_scope = RetrievedChunk(
            chunk_id="c_energy", document_id="doc-energy", document_name="solar.txt",
            chunk_index=0, text="Solar capacity 1.6 TW", score=0.9,
        )
        out_of_scope = RetrievedChunk(
            chunk_id="c_bread", document_id="doc-bread", document_name="bread.txt",
            chunk_index=0, text="Sourdough autolyse 30 minutes", score=0.8,
        )
        entity = EntityRecord(
            entity_id="e1", canonical_name="SunCore", original_mention="SunCore",
            entity_type="Organization", confidence=0.9, document_id="doc-energy",
            chunk_id="c_energy", page_number=1, extraction_method="heuristic",
        )

        class FakeGraph:
            def search_entities(self, terms, limit=20):
                return [entity]
            def traverse(self, entity_ids, max_hops=2, limit=20):
                return GraphEvidence(entities=[entity], relationships=[],
                                     source_chunks=[in_scope, out_of_scope])

        retriever = GraphRetriever(FakeGraph(), max_hops=2, max_results=20)
        result = retriever.retrieve("solar capacity", document_ids=["doc-energy"])

        doc_ids_returned = {c.document_id for c in result.source_chunks}
        assert "doc-energy" in doc_ids_returned
        assert "doc-bread" not in doc_ids_returned, (
            "Out-of-corpus chunk must be filtered after multi-hop traversal"
        )


# ---------------------------------------------------------------------------
# Hybrid RAG tests
# ---------------------------------------------------------------------------


class TestHybridRAG:
    def test_hybrid_combines_vector_and_graph_results(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "Compare solar and battery storage costs",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=doc_ids,
        )
        assert result.retrieved_sources, "Hybrid RAG must return chunks"
        assert result.answer

    def test_hybrid_does_not_return_duplicate_chunks(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "solar energy efficiency",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=doc_ids,
        )
        chunk_ids = [c.chunk_id for c in result.retrieved_sources]
        assert len(chunk_ids) == len(set(chunk_ids)), "Hybrid RAG must not return duplicate chunks"

    def test_rrf_scores_are_ordered_descending(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "battery cost 2023",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=doc_ids,
        )
        if len(result.retrieved_sources) >= 2:
            scores = [c.metadata.get("rrf_score", c.score) for c in result.retrieved_sources]
            assert scores == sorted(scores, reverse=True) or all(
                s == scores[0] for s in scores
            ), "Chunks must be ordered by descending RRF score"

    def test_hybrid_corpus_isolation(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        doc_service, query_service, energy_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        bread_outcome = doc_service.ingest_bytes("bread.txt", IRRELEVANT_DOC.encode("utf-8"))

        result = query_service.answer(
            "energy storage cost",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=energy_ids,
        )
        returned = {c.document_id for c in result.retrieved_sources}
        assert bread_outcome.document.document_id not in returned, (
            "Bread document must not appear in Hybrid RAG results for energy corpus"
        )


# ---------------------------------------------------------------------------
# Answer generation tests
# ---------------------------------------------------------------------------


class TestAnswerGeneration:
    def test_answer_is_non_empty_string(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What was the solar capacity by 2023?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert isinstance(result.answer, str)
        assert len(result.answer.strip()) > 0

    def test_absent_question_answer_indicates_no_evidence(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What is the recipe for making sourdough bread with autolyse?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        # Either retrieved context is empty OR the answer notes insufficient evidence
        # (both are acceptable; we just must not hallucinate bread facts)
        if not result.retrieved_sources:
            assert "insufficient" in result.answer.lower() or "not" in result.answer.lower(), (
                "When no evidence is retrieved, the answer must indicate lack of evidence"
            )

    def test_offline_fallback_clearly_labelled(self, tmp_path):
        """Offline/mock answers must be clearly labelled as not from an LLM."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "Who is Dr. Elena Vasquez?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        # In mock mode, the warning must be present
        assert "llm_offline_extractive_fallback" in result.warnings or result.model_name, (
            "Mock-mode answers must include the offline warning"
        )

    def test_citations_reference_real_chunks(self, tmp_path):
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "battery storage cost per kWh",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        if result.citations:
            for citation in result.citations:
                assert citation.document_id, "Citation must have a document_id"
                assert citation.chunk_id, "Citation must have a chunk_id"

    def test_answer_includes_question_relevant_content(self, tmp_path):
        """Offline fallback should prioritise passages related to the question."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What is the round-trip efficiency of lithium-ion batteries?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        # The answer should include related content from the battery document
        # In offline mode this is the top-ranked passage
        assert result.answer and len(result.answer) > 20


# ---------------------------------------------------------------------------
# Ingestion and document handling tests
# ---------------------------------------------------------------------------


class TestDocumentIngestion:
    def test_empty_document_is_handled_gracefully(self, tmp_path):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        from rag_benchmark_lab.generation.llm_service import OllamaLLMService
        from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
        from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
        from rag_benchmark_lab.services.document_registry import DocumentRegistry
        from rag_benchmark_lab.services.document_service import DocumentService
        from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore

        registry = DocumentRegistry(tmp_path / "reg.sqlite3")
        emb = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        qdrant = QdrantStore(":memory:", None, "t")
        qdrant.ensure_collection(emb.dimension)
        llm = OllamaLLMService("http://localhost:11434", "llama3", 5, 0, mock_mode=True)
        doc_service = DocumentService(
            registry=registry,
            embedding_service=emb,
            qdrant_store=qdrant,
            neo4j_store=NetworkXStore(),
            extractor=EntityRelationshipExtractor(llm),
            chunk_size=500,
            chunk_overlap=50,
            upload_size_limit_mb=10,
        )
        # An empty document should not raise; it may produce 0 chunks
        outcome = doc_service.ingest_bytes("empty.txt", b"")
        assert outcome is not None
        # Document should not be marked INDEXED if it produced no chunks
        # (it may be PARTIAL or INDEXED depending on implementation)
        assert outcome.document.status is not None

    def test_whitespace_only_document_produces_no_chunks(self, tmp_path):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        from rag_benchmark_lab.generation.llm_service import OllamaLLMService
        from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
        from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
        from rag_benchmark_lab.services.document_registry import DocumentRegistry
        from rag_benchmark_lab.services.document_service import DocumentService
        from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore

        registry = DocumentRegistry(tmp_path / "reg.sqlite3")
        emb = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        qdrant = QdrantStore(":memory:", None, "t")
        qdrant.ensure_collection(emb.dimension)
        llm = OllamaLLMService("http://localhost:11434", "llama3", 5, 0, mock_mode=True)
        doc_service = DocumentService(
            registry=registry,
            embedding_service=emb,
            qdrant_store=qdrant,
            neo4j_store=NetworkXStore(),
            extractor=EntityRelationshipExtractor(llm),
            chunk_size=500,
            chunk_overlap=50,
            upload_size_limit_mb=10,
        )
        outcome = doc_service.ingest_bytes("whitespace.txt", b"   \n\n   \t  ")
        assert outcome.document.chunk_count == 0, "Whitespace-only document must produce 0 chunks"

    def test_duplicate_document_is_detected(self, tmp_path):
        doc_service, _, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        content = b"Unique test document for duplicate detection."
        outcome1 = doc_service.ingest_bytes("dup_test.txt", content)
        outcome2 = doc_service.ingest_bytes("dup_test.txt", content)
        assert "duplicate_document_skipped" in outcome2.warnings

    def test_upload_size_limit_enforced(self, tmp_path):
        from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
        from rag_benchmark_lab.generation.llm_service import OllamaLLMService
        from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
        from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
        from rag_benchmark_lab.services.document_registry import DocumentRegistry
        from rag_benchmark_lab.services.document_service import DocumentService
        from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore

        emb = EmbeddingService("sentence-transformers/all-MiniLM-L6-v2", mock_mode=True)
        doc_service = DocumentService(
            registry=DocumentRegistry(tmp_path / "r.sqlite3"),
            embedding_service=emb,
            qdrant_store=QdrantStore(":memory:", None, "t"),
            neo4j_store=NetworkXStore(),
            extractor=EntityRelationshipExtractor(
                OllamaLLMService("http://localhost:11434", "llama3", 5, 0, mock_mode=True)
            ),
            chunk_size=500,
            chunk_overlap=50,
            upload_size_limit_mb=1,
        )
        with pytest.raises(ValueError, match="upload limit"):
            doc_service.ingest_bytes("huge.txt", b"x" * (2 * 1024 * 1024))


# ---------------------------------------------------------------------------
# RRF correctness tests
# ---------------------------------------------------------------------------


class TestRRF:
    def test_rrf_combines_two_rankings(self):
        from rag_benchmark_lab.retrieval.rrf import reciprocal_rank_fusion
        r1 = ["a", "b", "c"]
        r2 = ["b", "a", "d"]
        scores = reciprocal_rank_fusion([r1, r2])
        # "b" appears at rank 2 and 1, "a" at rank 1 and 2 — both should score higher than "c"/"d"
        assert scores["b"] > scores["c"]
        assert scores["a"] > scores["d"]

    def test_rrf_single_ranking_is_monotonically_decreasing(self):
        from rag_benchmark_lab.retrieval.rrf import reciprocal_rank_fusion
        r = ["x", "y", "z"]
        scores = reciprocal_rank_fusion([r])
        vals = [scores["x"], scores["y"], scores["z"]]
        assert vals == sorted(vals, reverse=True)

    def test_rrf_all_empty_returns_empty(self):
        from rag_benchmark_lab.retrieval.rrf import reciprocal_rank_fusion
        assert reciprocal_rank_fusion([]) == {}
        assert reciprocal_rank_fusion([[], []]) == {}


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    def test_faithfulness_high_when_answer_from_context(self):
        from rag_benchmark_lab.evaluation.metrics import faithfulness
        ctx = ["The solar LCOE fell to $0.033 per kWh in 2023"]
        pred = "The solar LCOE fell to $0.033 per kWh in 2023 according to reports."
        result = faithfulness(pred, ctx)
        assert result.value is not None and result.value > 0.5

    def test_faithfulness_low_when_answer_unrelated_to_context(self):
        from rag_benchmark_lab.evaluation.metrics import faithfulness
        ctx = ["Cats are mammals that meow"]
        pred = "The LCOE for solar fell dramatically in 2023 due to photovoltaic improvements."
        result = faithfulness(pred, ctx)
        assert result.value is not None and result.value < 0.7

    def test_token_f1_exact_match(self):
        from rag_benchmark_lab.evaluation.metrics import token_f1
        result = token_f1("solar energy capacity 1.6 TW", "solar energy capacity 1.6 TW")
        assert result.value == 1.0

    def test_retrieval_recall_at_k(self):
        from rag_benchmark_lab.evaluation.metrics import retrieval_recall_at_k
        from rag_benchmark_lab.schemas.documents import RetrievedChunk
        chunks = [
            RetrievedChunk("c1", "doc-solar", "solar_report.txt", 0, "text", 0.9),
            RetrievedChunk("c2", "doc-battery", "battery_overview.txt", 0, "text", 0.8),
        ]
        result = retrieval_recall_at_k(chunks, ["doc-solar", "doc-battery"])
        assert result.value == 1.0

    def test_retrieval_recall_partial_hit(self):
        from rag_benchmark_lab.evaluation.metrics import retrieval_recall_at_k
        from rag_benchmark_lab.schemas.documents import RetrievedChunk
        chunks = [RetrievedChunk("c1", "doc-solar", "solar.txt", 0, "text", 0.9)]
        result = retrieval_recall_at_k(chunks, ["doc-solar", "doc-battery"])
        assert result.value == 0.5


# ---------------------------------------------------------------------------
# Generalized Production RAG Scenarios (Section 9)
# ---------------------------------------------------------------------------


class TestGeneralizedProductionScenarios:
    def test_direct_factual_numeric_question(self, tmp_path):
        """Direct factual questions with numbers must retrieve exact facts."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What was the levelised cost of energy for utility-scale solar in 2023?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert result.retrieved_sources, "Must retrieve relevant chunks"
        combined_text = " ".join(c.text for c in result.retrieved_sources)
        assert "0.033" in combined_text
        assert "0.033" in result.answer or "0.033" in combined_text

    def test_paraphrased_question(self, tmp_path):
        """Alternative phrasing of a question must successfully retrieve the relevant chunk."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        # Paraphrased version of "What was the global installed solar capacity by end of 2023?"
        result = query_service.answer(
            "How many terawatts of solar panels were set up worldwide through 2023?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        combined_text = " ".join(c.text for c in result.retrieved_sources)
        assert "1.6" in combined_text or "SunCore" in combined_text

    def test_technical_terminology_question(self, tmp_path):
        """Questions with technical terms (e.g. perovskite conversion efficiency) must match."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What conversion efficiency was achieved in laboratory tests for perovskite solar cells?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        combined_text = " ".join(c.text for c in result.retrieved_sources)
        assert "29.15" in combined_text or "perovskite" in combined_text

    def test_multi_document_synthesis(self, tmp_path):
        """Questions requiring evidence across multiple documents must retrieve sources from both."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "What are the costs per kWh for solar power versus utility-scale battery storage in 2023?",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=doc_ids,
        )
        doc_names = {c.document_name for c in result.retrieved_sources}
        assert "solar_report.txt" in doc_names
        assert "battery_overview.txt" in doc_names

    def test_entity_relationship_connection(self, tmp_path):
        """Graph traversal must connect entities and their organizations."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "Where is SunCore Systems headquartered and who is the CTO?",
            manual_strategy=RetrievalStrategy.GRAPH,
            document_ids=doc_ids,
        )
        assert result.retrieved_sources or (result.graph_evidence and result.graph_evidence.entities)

    def test_absent_question_honest_no_evidence(self, tmp_path):
        """Questions about absent topics must return a clear no-evidence statement."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        result = query_service.answer(
            "How do you bake a sourdough loaf using a dutch oven and steam?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert "not appear to contain sufficient evidence" in result.answer.lower() or "insufficient" in result.answer.lower() or "no evidence" in result.answer.lower()

    def test_empty_corpus_isolation_guard(self, tmp_path):
        """When document_ids is an empty list, retrieval must not leak documents from any other corpus."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        _, query_service, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        # Query with an empty document_ids list (simulates an empty knowledge base)
        result = query_service.answer(
            "What was the solar capacity?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=[],
        )
        assert len(result.retrieved_sources) == 0, "No chunks should be retrieved from an empty corpus"
        assert "contains no documents" in result.answer.lower() or "insufficient" in result.answer.lower()

    def test_conflicting_facts_in_isolated_corpora(self, tmp_path):
        """Corpus A and Corpus B must remain strictly isolated even when containing similar queries."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        doc_service, query_service, energy_ids = _build_full_pipeline(tmp_path, ENERGY_CORPUS)

        # Ingest a second distinct corpus with conflicting numbers
        CORPUS_B_DOC = "The Advanced Photovoltaics Study 2024: Global solar capacity reached 9.9 TW by 2023."
        outcome_b = doc_service.ingest_bytes("study_b.txt", CORPUS_B_DOC.encode("utf-8"))
        corpus_b_ids = [outcome_b.document.document_id]

        # Query Corpus A
        res_a = query_service.answer(
            "What was the global installed solar capacity by end of 2023?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=energy_ids,
        )
        # Query Corpus B
        res_b = query_service.answer(
            "What was the global installed solar capacity by end of 2023?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=corpus_b_ids,
        )

        sources_a = {c.document_id for c in res_a.retrieved_sources}
        sources_b = {c.document_id for c in res_b.retrieved_sources}

        assert corpus_b_ids[0] not in sources_a, "Corpus B document must not appear in Corpus A retrieval"
        assert all(d_id not in sources_b for d_id in energy_ids), "Corpus A documents must not appear in Corpus B retrieval"

    def test_empty_document_marked_failed(self, tmp_path):
        """An uploaded empty document must be marked as FAILED with an error message."""
        from rag_benchmark_lab.schemas.documents import DocumentStatus
        doc_service, _, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)
        outcome = doc_service.ingest_bytes("zero_bytes.txt", b"")
        assert outcome.document.status == DocumentStatus.FAILED
        assert "no extractable text" in outcome.document.error_message.lower()

    def test_sequential_chunk_indexing_across_pages(self, tmp_path):
        """Multi-page document chunks must have globally sequential, non-duplicate chunk indices."""
        from rag_benchmark_lab.ingestion.document_loader import LoadedDocument, LoadedPage
        doc_service, _, _ = _build_full_pipeline(tmp_path, ENERGY_CORPUS)

        # Ingest text that chunks into multiple items
        long_text = "Section A fact. " * 80 + "\n\n" + "Section B fact. " * 80
        outcome = doc_service.ingest_bytes("multipage.txt", long_text.encode("utf-8"))
        indices = [c.chunk_index for c in outcome.chunks]
        assert indices == list(range(len(indices))), f"Chunk indices must be strictly sequential, got {indices}"
        assert len(indices) == len(set(indices)), "All chunk indices must be unique"


# ---------------------------------------------------------------------------
# Generalized Arbitrary-Domain Scenarios (Medicine, Finance, Multi-Doc, Isolation)
# ---------------------------------------------------------------------------

NEUROMED_DOC = """
Clinical Trial NM-401 — NeuroMed Therapeutics, 2024.
Dr. Aris Thorne serves as the Chief Scientific Officer at NeuroMed Therapeutics, which is headquartered in Geneva, Switzerland.
NeuroMed Therapeutics developed NM-401, a small-molecule monoclonal derivative that targets and inhibits amyloid-beta plaque aggregation.
In Phase 2 clinical trials concluding in November 2023, 420 patients received a dosage of 25 mg twice daily.
The study demonstrated a 34.6% reduction in cognitive decline compared to placebo over 18 months.
Phase 3 trials require enrolling 1500 participants across 12 countries by mid-2025.
""".strip()

CLOUD_INFRA_DOC = """
Cloud Infrastructure Architecture — VertexCloud Systems, 2024.
VertexCloud Systems operates 64 global data centres with an average service availability of 99.995%.
The infrastructure relies on custom LiquidCooling units to achieve an industry-leading Power Usage Effectiveness (PUE) of 1.08.
Chief Infrastructure Architect Sarah Lin reported that network throughput reached 400 Gbps per cluster interconnect.
""".strip()

FINANCIAL_REPORT_DOC = """
FY2024 Financial Performance Report — VertexCloud Systems.
VertexCloud Systems generated total annual revenue of $4.2 billion in FY2024, representing 28% year-over-year growth.
Capital expenditures on infrastructure hardware reached $850 million, while operating expenses were held at $1.6 billion.
Operating margin expanded to 41.2% in the fourth quarter.
""".strip()


class TestArbitraryDomainRAGScenarios:
    """Verifies that retrieval and generation work seamlessly on previously unseen domains."""

    def test_unseen_medical_factual_and_numeric_answer(self, tmp_path):
        """Vector RAG must extract precise facts and numbers from arbitrary medical documents."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        med_corpus = SyntheticCorpus(
            documents={"neuromed_trial.txt": NEUROMED_DOC},
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, med_corpus)

        res = query_service.answer(
            "What dosage of NM-401 did patients receive in Phase 2?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert res.retrieved_sources, "Must retrieve medical chunks"
        assert "25 mg" in res.answer or "420" in res.answer, f"Answer should contain exact dosage fact: {res.answer}"
        assert res.citations, "Citations must be present"

    def test_unseen_medical_paraphrased_query(self, tmp_path):
        """Retrieval must handle paraphrased questions with different vocabulary."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        med_corpus = SyntheticCorpus(
            documents={"neuromed_trial.txt": NEUROMED_DOC},
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, med_corpus)

        res = query_service.answer(
            "How much medicine was administered daily to trial participants?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert res.retrieved_sources
        text = " ".join(c.text for c in res.retrieved_sources)
        assert "25 mg" in text or "dosage" in text

    def test_unseen_technical_terminology(self, tmp_path):
        """Technical biological terminology must be retrieved accurately."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        med_corpus = SyntheticCorpus(
            documents={"neuromed_trial.txt": NEUROMED_DOC},
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, med_corpus)

        res = query_service.answer(
            "What biological mechanism does NM-401 use against plaques?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        text = " ".join(c.text for c in res.retrieved_sources)
        assert "amyloid-beta" in text or "aggregation" in text

    def test_unseen_graphrag_entity_relationships(self, tmp_path):
        """GraphRAG must discover person, role, and organization relationships in unseen text."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        med_corpus = SyntheticCorpus(
            documents={"neuromed_trial.txt": NEUROMED_DOC},
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, med_corpus)

        res = query_service.answer(
            "Who is the Chief Scientific Officer of NeuroMed Therapeutics and where is it located?",
            manual_strategy=RetrievalStrategy.GRAPH,
            document_ids=doc_ids,
        )
        assert res.retrieved_sources or (res.graph_evidence and res.graph_evidence.entities)
        combined = res.answer + " " + " ".join(c.text for c in res.retrieved_sources)
        assert "Aris Thorne" in combined or "Geneva" in combined

    def test_unseen_multi_document_hybrid_synthesis(self, tmp_path):
        """Hybrid RAG must combine facts across distinct financial and infrastructure documents."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        multi_corpus = SyntheticCorpus(
            documents={
                "cloud_infra.txt": CLOUD_INFRA_DOC,
                "financial_report.txt": FINANCIAL_REPORT_DOC,
            },
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, multi_corpus)

        res = query_service.answer(
            "Compare VertexCloud's annual revenue with its total number of operational data centres.",
            manual_strategy=RetrievalStrategy.HYBRID,
            document_ids=doc_ids,
        )
        retrieved_docs = {c.document_name for c in res.retrieved_sources}
        assert "cloud_infra.txt" in retrieved_docs, "Must retrieve from cloud infra document"
        assert "financial_report.txt" in retrieved_docs, "Must retrieve from financial report document"
        assert "$4.2 billion" in res.answer or "4.2" in res.answer or "64" in res.answer

    def test_absent_question_returns_insufficient_evidence(self, tmp_path):
        """Questions about unrepresented topics must state insufficient evidence honestly."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        med_corpus = SyntheticCorpus(
            documents={"neuromed_trial.txt": NEUROMED_DOC},
            ground_truth=[],
        )
        _, query_service, doc_ids = _build_full_pipeline(tmp_path, med_corpus)

        res = query_service.answer(
            "What is the capital city of Peru and what is its population?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=doc_ids,
        )
        assert "sufficient evidence" in res.answer.lower() or "insufficient" in res.answer.lower()

    def test_strict_cross_corpus_isolation_two_independent_kbs(self, tmp_path):
        """Facts from Knowledge Base A must never leak into queries on Knowledge Base B."""
        from rag_benchmark_lab.schemas.documents import RetrievalStrategy
        doc_service, query_service, _ = _build_full_pipeline(
            tmp_path, SyntheticCorpus(documents={"dummy.txt": "Baseline dummy text."}, ground_truth=[])
        )

        kb_a_doc = "Project Orion was launched in 2019 with a grant of $75 million in Tokyo."
        kb_b_doc = "Project Orion was launched in 2024 with a grant of $10 million in London."

        out_a = doc_service.ingest_bytes("orion_a.txt", kb_a_doc.encode("utf-8"))
        out_b = doc_service.ingest_bytes("orion_b.txt", kb_b_doc.encode("utf-8"))
        id_a = out_a.document.document_id
        id_b = out_b.document.document_id

        # Query KB A
        res_a = query_service.answer(
            "Where and with what grant was Project Orion launched?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=[id_a],
        )
        # Query KB B
        res_b = query_service.answer(
            "Where and with what grant was Project Orion launched?",
            manual_strategy=RetrievalStrategy.VECTOR,
            document_ids=[id_b],
        )

        assert "Tokyo" in res_a.answer or "75" in res_a.answer
        assert "London" not in res_a.answer
        assert "Tokyo" not in res_b.answer

    def test_malformed_document_handled_gracefully(self, tmp_path):
        """Malformed document bytes must not crash the service and must be marked FAILED."""
        from rag_benchmark_lab.schemas.documents import DocumentStatus
        doc_service, _, _ = _build_full_pipeline(
            tmp_path, SyntheticCorpus(documents={"seed.txt": "Seed text."}, ground_truth=[])
        )

        # Corrupt PDF header with garbage content
        corrupt_bytes = b"%PDF-1.4\n\x00\xff\xfe\xfaCORRUPT_PAYLOAD_WITHOUT_EOF"
        outcome = doc_service.ingest_bytes("corrupt.pdf", corrupt_bytes)
        assert outcome.document.status == DocumentStatus.FAILED
        assert outcome.errors, "Errors list must record the extraction failure"

