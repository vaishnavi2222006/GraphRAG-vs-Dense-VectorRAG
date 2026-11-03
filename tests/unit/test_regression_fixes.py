"""Regression tests for the targeted bug-fixes.

Covers:
1. BenchmarkRunner.run() corpus isolation – document_ids forwarded to QueryService.
2. GraphRetriever secondary traversal filtering – source chunks from outside the
   selected corpus are excluded after multi-hop traversal.
3. Upload helper – _cache_upload writes a safe, deterministic filename and is
   idempotent on repeat calls.
4. Evaluation empty-KB guard logic – _kb_doc_ids is empty when no documents
   exist in the KB (tested at the helper level).
"""
from __future__ import annotations

import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Shared stubs
# ---------------------------------------------------------------------------


@dataclass
class FakeLLM:
    mock_mode: bool = True

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0):
        return type(
            "Response",
            (),
            {
                "text": "stubbed answer",
                "model_name": "mock",
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "latency_ms": 1.0,
                "mock_mode": True,
                "warnings": [],
            },
        )()


# ---------------------------------------------------------------------------
# Fix 1: BenchmarkRunner corpus isolation
# ---------------------------------------------------------------------------


def test_benchmark_runner_forwards_document_ids_to_query_service():
    """BenchmarkRunner.run(document_ids=...) must pass them through to every
    QueryService.answer() call so retrieval is scoped to the selected KB."""
    from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
    from rag_benchmark_lab.evaluation.dataset import sample_dataset
    from rag_benchmark_lab.evaluation.result_store import ResultStore
    from rag_benchmark_lab.schemas.documents import RetrievedChunk, RetrievalStrategy
    from rag_benchmark_lab.services.document_registry import DocumentRegistry

    received_doc_ids: list = []

    class CapturingQueryService:
        llm_service = FakeLLM()

        def answer(self, question: str, manual_strategy=None, document_ids=None):
            received_doc_ids.append(document_ids)
            chunk = RetrievedChunk(
                chunk_id="c1",
                document_id="doc-kb1",
                document_name="kb1_doc",
                chunk_index=0,
                text="evidence text",
                score=0.9,
            )
            return type(
                "Answer",
                (),
                {
                    "answer": f"{manual_strategy.value} answer",
                    "selected_strategy": manual_strategy,
                    "citations": [],
                    "retrieved_sources": [chunk],
                    "retrieval_latency_ms": 1.0,
                    "generation_latency_ms": 1.0,
                    "total_latency_ms": 2.0,
                    "warnings": [],
                    "errors": [],
                },
            )()

    with tempfile.TemporaryDirectory() as tmp:
        registry = DocumentRegistry(Path(tmp) / "reg.sqlite3")
        runner = BenchmarkRunner(CapturingQueryService(), ResultStore(registry))

        expected_ids = ["doc-kb1", "doc-kb2"]
        examples = sample_dataset()[:2]
        result = runner.run(
            examples,
            strategies=[RetrievalStrategy.VECTOR],
            document_ids=expected_ids,
        )

    assert all(doc_ids == expected_ids for doc_ids in received_doc_ids), (
        f"Not all QueryService.answer() calls received the correct document_ids. Got: {received_doc_ids}"
    )
    assert result["rows"], "BenchmarkRunner must produce result rows"


def test_benchmark_runner_without_document_ids_passes_none():
    """When no document_ids are supplied, QueryService.answer() must receive None."""
    from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
    from rag_benchmark_lab.evaluation.dataset import sample_dataset
    from rag_benchmark_lab.evaluation.result_store import ResultStore
    from rag_benchmark_lab.schemas.documents import RetrievedChunk, RetrievalStrategy
    from rag_benchmark_lab.services.document_registry import DocumentRegistry

    received_doc_ids: list = []

    class CapturingQueryService:
        llm_service = FakeLLM()

        def answer(self, question: str, manual_strategy=None, document_ids=None):
            received_doc_ids.append(document_ids)
            chunk = RetrievedChunk(
                chunk_id="c1", document_id="d1", document_name="doc", chunk_index=0, text="ev", score=0.8
            )
            return type(
                "Answer",
                (),
                {
                    "answer": "answer",
                    "selected_strategy": manual_strategy,
                    "citations": [],
                    "retrieved_sources": [chunk],
                    "retrieval_latency_ms": 1.0,
                    "generation_latency_ms": 1.0,
                    "total_latency_ms": 2.0,
                    "warnings": [],
                    "errors": [],
                },
            )()

    with tempfile.TemporaryDirectory() as tmp:
        registry = DocumentRegistry(Path(tmp) / "reg.sqlite3")
        runner = BenchmarkRunner(CapturingQueryService(), ResultStore(registry))
        runner.run(sample_dataset()[:1], strategies=[RetrievalStrategy.VECTOR])

    assert all(doc_ids is None for doc_ids in received_doc_ids), (
        "Without document_ids kwarg, QueryService must receive None"
    )


# ---------------------------------------------------------------------------
# Fix 2: GraphRetriever secondary traversal filtering
# ---------------------------------------------------------------------------


def test_graph_retriever_filters_source_chunks_after_traversal():
    """Source chunks from multi-hop traversal outside the selected corpus must be removed."""
    from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
    from rag_benchmark_lab.schemas.documents import (
        EntityRecord,
        GraphEvidence,
        RetrievedChunk,
    )

    in_scope_chunk = RetrievedChunk(
        chunk_id="c_in", document_id="doc-kb1", document_name="in scope doc",
        chunk_index=0, text="relevant text", score=0.9,
    )
    out_of_scope_chunk = RetrievedChunk(
        chunk_id="c_out", document_id="doc-other-kb", document_name="foreign doc",
        chunk_index=0, text="foreign text", score=0.85,
    )
    seed_entity = EntityRecord(
        entity_id="e1", canonical_name="RAG", original_mention="RAG",
        entity_type="Concept", confidence=0.9, document_id="doc-kb1",
        chunk_id="c_in", page_number=1, extraction_method="heuristic",
    )

    class FakeGraphStore:
        def search_entities(self, terms, limit=20):
            return [seed_entity]

        def traverse(self, entity_ids, max_hops=2, limit=20):
            return GraphEvidence(
                entities=[seed_entity],
                relationships=[],
                source_chunks=[in_scope_chunk, out_of_scope_chunk],
            )

    retriever = GraphRetriever(FakeGraphStore(), max_hops=2, max_results=20)
    result = retriever.retrieve("What is RAG?", document_ids=["doc-kb1"])

    returned_doc_ids = {chunk.document_id for chunk in result.source_chunks}
    assert "doc-kb1" in returned_doc_ids, "In-scope chunk must be present"
    assert "doc-other-kb" not in returned_doc_ids, (
        "Out-of-corpus chunk from multi-hop traversal must be filtered out"
    )
    assert "doc-other-kb" not in {chunk.document_id for chunk in result.evidence.source_chunks}, (
        "GraphEvidence.source_chunks must also be filtered"
    )


def test_graph_retriever_no_filter_when_document_ids_is_none():
    """When document_ids is None, all source chunks from traversal are returned."""
    from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
    from rag_benchmark_lab.schemas.documents import EntityRecord, GraphEvidence, RetrievedChunk

    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}", document_id=f"doc{i}", document_name=f"doc{i}",
            chunk_index=0, text="text", score=0.8,
        )
        for i in range(3)
    ]
    entity = EntityRecord(
        entity_id="e1", canonical_name="Test", original_mention="Test",
        entity_type="Concept", confidence=0.9, document_id="doc0",
        chunk_id="c0", page_number=1, extraction_method="heuristic",
    )

    class FakeGraphStore:
        def search_entities(self, terms, limit=20):
            return [entity]

        def traverse(self, entity_ids, max_hops=2, limit=20):
            return GraphEvidence(entities=[entity], relationships=[], source_chunks=chunks)

    retriever = GraphRetriever(FakeGraphStore())
    result = retriever.retrieve("Test query", document_ids=None)
    assert len(result.source_chunks) == 3, "All chunks should be returned when no filter is applied"


# ---------------------------------------------------------------------------
# Fix 3: Upload caching helper (_cache_upload equivalent)
# ---------------------------------------------------------------------------


def _cache_upload_impl(file_bytes: bytes, original_name: str, upload_dir: Path) -> str:
    """Local copy of the helper from pages/2_Documents.py for unit testing."""
    content_hash = hashlib.sha256(file_bytes).hexdigest()
    safe_stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original_name).stem)
    suffix = Path(original_name).suffix or ""
    filename = f"{content_hash[:12]}_{safe_stem}{suffix}"
    dest = upload_dir / filename
    if not dest.exists():
        upload_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(file_bytes)
    return str(dest)


def test_cache_upload_creates_file_on_disk(tmp_path):
    data = b"Hello, world! This is a test document."
    cached = _cache_upload_impl(data, "my report.pdf", tmp_path)
    path = Path(cached)
    assert path.exists(), "Cached file must exist on disk"
    assert path.read_bytes() == data, "Cached content must match the original bytes"


def test_cache_upload_filename_is_safe(tmp_path):
    data = b"safe filename test"
    cached = _cache_upload_impl(data, "report (v2) final!.pdf", tmp_path)
    fname = Path(cached).name
    assert re.match(r"^[A-Za-z0-9._-]+$", fname), (
        f"Cached filename '{fname}' contains unsafe characters"
    )


def test_cache_upload_is_idempotent(tmp_path):
    data = b"idempotency test content"
    path1 = _cache_upload_impl(data, "doc.txt", tmp_path)
    path2 = _cache_upload_impl(data, "doc.txt", tmp_path)
    assert path1 == path2, "Both calls must return the same path"
    assert Path(path1).read_bytes() == data


def test_cache_upload_different_content_gives_different_path(tmp_path):
    data_a = b"document A content"
    data_b = b"document B content"
    path_a = _cache_upload_impl(data_a, "doc.txt", tmp_path)
    path_b = _cache_upload_impl(data_b, "doc.txt", tmp_path)
    assert path_a != path_b, "Different content must produce different cached paths"
    assert Path(path_a).read_bytes() == data_a
    assert Path(path_b).read_bytes() == data_b


# ---------------------------------------------------------------------------
# Fix 4: Empty KB guard
# ---------------------------------------------------------------------------


def test_empty_kb_returns_no_document_ids(tmp_path):
    """get_kb_document_ids returns [] for a KB with no documents."""
    from rag_benchmark_lab.services.user_registry import UserRegistry

    registry = UserRegistry(tmp_path / "users.sqlite3")
    user, _ = registry.register_user("Tester", "tester@example.com", "Password1!")
    user_id = user.user_id
    kb = registry.create_knowledge_base(user_id, "Empty KB", "")
    doc_ids = registry.get_kb_document_ids(kb.kb_id, user_id)

    assert doc_ids == [], "An empty KB must return an empty list of document IDs"
    assert len(doc_ids) == 0, "len(doc_ids) == 0 triggers the UI guard"


def test_kb_returns_document_ids_after_document_is_added(tmp_path):
    """get_kb_document_ids returns the correct IDs once a document is associated."""
    from rag_benchmark_lab.services.user_registry import UserRegistry

    registry = UserRegistry(tmp_path / "users.sqlite3")
    user, _ = registry.register_user("User2", "user2@example.com", "Password1!")
    user_id = user.user_id
    kb = registry.create_knowledge_base(user_id, "KB with docs", "")

    fake_doc_id = "deadbeef" * 8
    registry.add_document_to_kb(kb.kb_id, fake_doc_id)

    doc_ids = registry.get_kb_document_ids(kb.kb_id, user_id)
    assert fake_doc_id in doc_ids
    assert len(doc_ids) == 1
