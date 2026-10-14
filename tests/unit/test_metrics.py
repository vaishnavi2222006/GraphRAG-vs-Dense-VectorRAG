from __future__ import annotations

from rag_benchmark_lab.evaluation.metrics import exact_match, latency_summary, retrieval_recall_at_k, token_f1
from rag_benchmark_lab.schemas.documents import RetrievedChunk


def test_exact_match_is_normalized() -> None:
    assert exact_match("Hello, World!", "hello world").value == 1.0


def test_token_f1_returns_positive_overlap() -> None:
    assert token_f1("alpha beta", "alpha gamma").value > 0


def test_retrieval_recall_at_k_uses_document_ids() -> None:
    chunks = [RetrievedChunk(chunk_id="1", document_id="doc-1", document_name="doc", chunk_index=0, text="x", score=0.9)]
    assert retrieval_recall_at_k(chunks, ["doc-1"]).value == 1.0


def test_latency_summary_handles_empty_and_real_values() -> None:
    assert latency_summary([])["mean"] is None
    summary = latency_summary([10.0, 20.0, 30.0])
    assert summary["median"] == 20.0
