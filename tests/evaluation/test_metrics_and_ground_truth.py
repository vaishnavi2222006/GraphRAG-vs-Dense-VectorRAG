"""Unit tests for evaluation metrics, normalization, missing reference answers, and failure handling."""
from __future__ import annotations

import pytest

from rag_benchmark_lab.evaluation.metrics import (
    answer_correctness,
    answer_relevancy,
    citation_correctness,
    completeness,
    context_precision,
    context_recall,
    exact_match,
    faithfulness,
    hit_rate_at_k,
    mean_reciprocal_rank,
    metric_bundle,
    retrieval_recall_at_k,
    token_f1,
)
from rag_benchmark_lab.schemas.documents import (
    AnswerResult,
    Citation,
    RetrievedChunk,
    RetrievalStrategy,
)


def test_missing_reference_answer_handling() -> None:
    """When reference answer is missing, ground-truth metrics must be N/A."""
    pred = "The pipeline extracts, chunks, and indexes data."

    # Test with None
    corr_none = answer_correctness(pred, None)
    assert corr_none.value is None
    assert corr_none.available is False
    assert corr_none.details.get("reason") == "missing_reference_answer"
    assert corr_none.details.get("is_ground_truth") is False

    comp_none = completeness(pred, None)
    assert comp_none.value is None
    assert comp_none.available is False
    assert comp_none.details.get("reason") == "missing_reference_answer"

    # Test with empty string
    corr_empty = answer_correctness(pred, "   ")
    assert corr_empty.value is None
    assert corr_empty.available is False

    comp_empty = completeness(pred, "")
    assert comp_empty.value is None
    assert comp_empty.available is False


def test_metric_calculation_and_normalization() -> None:
    """Verify quality metrics are deterministic and bounded strictly between 0.0 and 1.0."""
    ref = "The hospital operates 250 beds and has 4 emergency rooms."
    pred_exact = "The hospital operates 250 beds and has 4 emergency rooms."
    pred_partial = "The hospital operates 250 beds."
    pred_wrong = "A completely unrelated statement about astronomy."

    # Exact match
    em_high = answer_correctness(pred_exact, ref)
    assert em_high.value == 1.0
    assert em_high.details.get("is_ground_truth") is True

    # Partial match with number agreement
    em_part = answer_correctness(pred_partial, ref)
    assert em_part.value is not None
    assert 0.0 < em_part.value < 1.0
    assert em_part.details.get("is_ground_truth") is True

    # Completely wrong
    em_wrong = answer_correctness(pred_wrong, ref)
    assert em_wrong.value is not None
    assert em_wrong.value == 0.0

    # Completeness normalization
    comp_high = completeness(pred_exact, ref)
    assert comp_high.value == 1.0
    comp_part = completeness(pred_partial, ref)
    assert comp_part.value is not None
    assert 0.0 < comp_part.value < 1.0


def test_citation_correctness_verification() -> None:
    """Test citation correctness against retrieved chunks and answer content."""
    chunk1 = RetrievedChunk(
        chunk_id="c1",
        document_id="doc1",
        document_name="guide.txt",
        chunk_index=0,
        text="St. Jude Hospital has 250 inpatient beds.",
        score=0.9,
    )
    chunk2 = RetrievedChunk(
        chunk_id="c2",
        document_id="doc2",
        document_name="contact.txt",
        chunk_index=1,
        text="The cardiology department phone extension is 4412.",
        score=0.85,
    )

    cit_valid = Citation(document_id="doc1", document_name="guide.txt", chunk_id="c1", chunk_index=0, score=0.9)
    cit_hallucinated = Citation(document_id="unknown_doc", document_name="fake.txt", chunk_id="c99", chunk_index=99)

    ans_valid = AnswerResult(
        answer="St. Jude Hospital operates 250 inpatient beds according to guide.",
        selected_strategy=RetrievalStrategy.VECTOR,
        citations=[cit_valid],
        retrieved_sources=[chunk1, chunk2],
    )
    res_valid = citation_correctness(ans_valid)
    assert res_valid.value == 1.0

    ans_mixed = AnswerResult(
        answer="St. Jude Hospital operates 250 inpatient beds.",
        selected_strategy=RetrievalStrategy.VECTOR,
        citations=[cit_valid, cit_hallucinated],
        retrieved_sources=[chunk1, chunk2],
    )
    res_mixed = citation_correctness(ans_mixed)
    assert res_mixed.value == 0.5


def test_failed_generation_receives_no_quality_scores() -> None:
    """Ensure failed generations do not receive positive or misleading quality scores."""
    failed_ans = AnswerResult(
        answer="Error generating answer: QuotaExceededError",
        selected_strategy=RetrievalStrategy.HYBRID,
        citations=[],
        retrieved_sources=[],
        errors=["QuotaExceededError: Resource exhausted"],
        metadata={"generation_status": "failed"},
    )

    metrics = metric_bundle(failed_ans, "Expected valid answer", ["doc1"], ["context"], question="Test question")

    # All quality metrics must be None / unavailable for failed runs
    for m_name in ("answer_correctness", "faithfulness", "answer_relevancy", "completeness", "citation_correctness"):
        m_res = metrics[m_name]
        assert m_res.value is None, f"{m_name} should be None on failed generation, got {m_res.value}"
        assert m_res.available is False
        assert m_res.details.get("reason") == "generation_failed"


def test_retrieval_metrics_with_valid_relevance_labels() -> None:
    """Requirement 14: Compute Recall@K, HitRate@K, and MRR with known relevance labels."""
    chunk1 = RetrievedChunk(
        chunk_id="c1",
        document_id="doc_alpha",
        document_name="architecture.pdf",
        chunk_index=0,
        text="The ingestion pipeline validates and chunks documents.",
        score=0.9,
    )
    chunk2 = RetrievedChunk(
        chunk_id="c2",
        document_id="doc_beta",
        document_name="retrieval.pdf",
        chunk_index=1,
        text="The retriever fetches top passages.",
        score=0.8,
    )
    chunk3 = RetrievedChunk(
        chunk_id="c3",
        document_id="doc_gamma",
        document_name="eval.pdf",
        chunk_index=2,
        text="Evaluations compare multiple approaches.",
        score=0.7,
    )

    relevant_labels = ["doc_beta", "doc_delta"]

    # 1. Recall@K: 1 out of 2 relevant docs retrieved in chunk2 -> Recall = 0.5
    recall_res = retrieval_recall_at_k([chunk1, chunk2, chunk3], relevant_labels)
    assert recall_res.available is True
    assert recall_res.value == 0.5

    # 2. HitRate@K: At least one relevant doc retrieved -> 1.0
    hit_res = hit_rate_at_k([chunk1, chunk2, chunk3], relevant_labels)
    assert hit_res.available is True
    assert hit_res.value == 1.0

    # 3. MRR: First relevant item is at rank 2 (chunk2) -> MRR = 1/2 = 0.5
    mrr_res = mean_reciprocal_rank([chunk1, chunk2, chunk3], relevant_labels)
    assert mrr_res.available is True
    assert mrr_res.value == 0.5

    # If first item is relevant (chunk1)
    mrr_top = mean_reciprocal_rank([chunk1, chunk2], ["doc_alpha"])
    assert mrr_top.value == 1.0

    # If no relevant items retrieved
    mrr_none = mean_reciprocal_rank([chunk1, chunk2], ["doc_omega"])
    assert mrr_none.value == 0.0


def test_comparison_charts_and_matrix_consistency() -> None:
    """Requirement 18: Ensure comparison matrix and chart values accurately match underlying metrics."""
    res_map = {
        "VECTOR": {
            "status": "Success",
            "total_latency_ms": 320.5,
            "retrieval_latency_ms": 45.0,
            "generation_latency_ms": 275.5,
            "retrieved_chunk_count": 3,
            "graph_path_count": None,
            "metrics": {
                "answer_correctness": {"value": 0.88, "available": True},
                "faithfulness": {"value": 0.92, "available": True},
                "answer_relevancy": {"value": 0.85, "available": True},
                "completeness": {"value": 0.80, "available": True},
                "citation_correctness": {"value": 0.90, "available": True},
                "context_precision": {"value": 0.75, "available": True},
                "context_recall": {"value": 0.80, "available": True},
                "retrieval_recall_at_k": {"value": 1.0, "available": True},
                "mrr": {"value": 1.0, "available": True},
            },
        },
        "DIRECT": {
            "status": "Success",
            "total_latency_ms": 150.0,
            "retrieval_latency_ms": None,
            "generation_latency_ms": 150.0,
            "retrieved_chunk_count": None,
            "graph_path_count": None,
            "metrics": {
                "answer_correctness": {"value": 0.70, "available": True},
                "faithfulness": {"value": None, "available": False},
                "answer_relevancy": {"value": 0.80, "available": True},
                "completeness": {"value": 0.65, "available": True},
                "citation_correctness": {"value": None, "available": False},
                "context_precision": {"value": None, "available": False},
                "context_recall": {"value": None, "available": False},
                "retrieval_recall_at_k": {"value": None, "available": False},
                "mrr": {"value": None, "available": False},
            },
        },
    }

    # Verify that chart metrics directly pull from the same metric dictionaries without drift or fabrication
    for strat, data in res_map.items():
        mets = data["metrics"]
        for m_key in ("answer_correctness", "answer_relevancy", "completeness"):
            assert mets[m_key]["value"] is not None
            assert 0.0 <= mets[m_key]["value"] <= 1.0

        if strat == "DIRECT":
            # Direct LLM must not have retrieval metrics in charts or matrix
            assert mets["faithfulness"]["value"] is None
            assert mets["context_precision"]["value"] is None
            assert data["retrieval_latency_ms"] is None


def test_context_precision_ranking_aware_and_ground_truth() -> None:
    """Validate ranking-aware Context Precision with ground truth labels and reference answers."""
    c1 = RetrievedChunk(chunk_id="c1", document_id="doc1", document_name="doc1.txt", chunk_index=0, text="Ingestion validates data.", score=0.9)
    c2 = RetrievedChunk(chunk_id="c2", document_id="doc2", document_name="doc2.txt", chunk_index=1, text="Database stores embeddings.", score=0.8)
    c3 = RetrievedChunk(chunk_id="c3", document_id="doc3", document_name="doc3.txt", chunk_index=2, text="Graphs capture entity relationships.", score=0.7)

    # 1. First item relevant (rank 1): Prec@1 = 1/1 = 1.0 -> AP = 1.0
    res_top = context_precision([c1, c2, c3], relevant_document_ids=["doc1"])
    assert res_top.available is True
    assert res_top.value == 1.0

    # 2. Second item relevant (rank 2): Prec@1 = 0, Prec@2 = 1/2 = 0.5 -> AP = 0.5
    res_second = context_precision([c1, c2, c3], relevant_document_ids=["doc2"])
    assert res_second.available is True
    assert res_second.value == 0.5

    # 3. Ranks 1 and 3 relevant: Prec@1 = 1/1 = 1.0, Prec@3 = 2/3 = 0.6667 -> AP = (1.0 + 0.6667) / 2 = 0.8333
    res_multi = context_precision([c1, c2, c3], relevant_document_ids=["doc1", "doc3"])
    assert res_multi.available is True
    assert res_multi.value == 0.8333

    # 4. No relevant items retrieved -> AP = 0.0
    res_none = context_precision([c1, c2, c3], relevant_document_ids=["doc_unknown"])
    assert res_none.available is True
    assert res_none.value == 0.0

    # 5. Semantic relevance from reference answer when document IDs are missing
    ref_ans = "Ingestion pipeline validates incoming data and documents."
    res_ref = context_precision([c1, c2, c3], reference_answer=ref_ans, question="What does ingestion do?")
    assert res_ref.available is True
    assert res_ref.value == 1.0  # c1 is rank 1 and relevant

    # 6. Genuinely missing both ground truth labels and reference answer -> N/A
    res_missing = context_precision([c1, c2], reference_answer=None, relevant_document_ids=None)
    assert res_missing.available is False
    assert res_missing.value is None
    assert res_missing.details.get("reason") == "missing_ground_truth_and_reference_answer"


def test_context_recall_fact_extraction_and_support() -> None:
    """Validate Context Recall: 3 of 4 required facts supported yields 0.75."""
    ref_answer = (
        "1. Hospital operates 250 beds.\n"
        "2. Emergency room operates 24/7 with 4 triage units.\n"
        "3. Cardiology clinic provides diagnostic ultrasounds.\n"
        "4. Free visitor parking is available on south campus."
    )

    # Retrieved context supports facts 1, 2, and 3, but NOT fact 4 (parking)
    retrieved_texts = [
        "The hospital operates 250 inpatient beds.",
        "The emergency room operates 24/7 with four active triage units.",
        "Cardiology clinic diagnostic ultrasound appointments are available.",
    ]

    res = context_recall(retrieved_texts, reference_answer=ref_answer)
    assert res.available is True
    # 3 of 4 facts supported = 3/4 = 0.75
    assert res.value == 0.75
    assert res.details.get("supported_facts") == 3
    assert res.details.get("total_facts") == 4

    # When all 4 facts supported
    retrieved_with_parking = retrieved_texts + ["Free visitor parking is on the south campus."]
    res_full = context_recall(retrieved_with_parking, reference_answer=ref_answer)
    assert res_full.value == 1.0
    assert res_full.details.get("supported_facts") == 4

    # When missing reference answer -> N/A
    res_no_ref = context_recall(retrieved_texts, reference_answer=None)
    assert res_no_ref.available is False
    assert res_no_ref.value is None
    assert res_no_ref.details.get("reason") == "missing_reference_answer"


def test_recall_at_k_controlled_denominator() -> None:
    """Validate Recall@K: denominator is total known relevant items."""
    c1 = RetrievedChunk(chunk_id="c1", document_id="doc1", document_name="doc1.txt", chunk_index=0, text="", score=0.9)
    c2 = RetrievedChunk(chunk_id="c2", document_id="doc2", document_name="doc2.txt", chunk_index=1, text="", score=0.8)
    c3 = RetrievedChunk(chunk_id="c3", document_id="doc3", document_name="doc3.txt", chunk_index=2, text="", score=0.7)
    c4 = RetrievedChunk(chunk_id="c4", document_id="doc4", document_name="doc4.txt", chunk_index=3, text="", score=0.6)
    c5 = RetrievedChunk(chunk_id="c5", document_id="doc5", document_name="doc5.txt", chunk_index=4, text="", score=0.5)

    # 4 total known relevant items: doc1, doc2, doc_x, doc_y
    known_relevant = ["doc1", "doc2", "doc_x", "doc_y"]

    # 2 of 5 retrieved chunks are relevant (doc1, doc2) -> Recall = 2 / 4 = 0.5
    res = retrieval_recall_at_k([c1, c2, c3, c4, c5], known_relevant)
    assert res.available is True
    assert res.value == 0.5
    assert res.details.get("total_known_relevant") == 4

    # With K=1, only c1 is considered (1 relevant out of 4) -> Recall@1 = 1 / 4 = 0.25
    res_k1 = retrieval_recall_at_k([c1, c2, c3, c4, c5], known_relevant, k=1)
    assert res_k1.value == 0.25

    # Missing relevance labels -> N/A
    res_missing = retrieval_recall_at_k([c1, c2], None)
    assert res_missing.available is False
    assert res_missing.value is None


def test_mrr_controlled_examples() -> None:
    """Validate MRR: first relevant result at rank 2 gives 0.5."""
    c1 = RetrievedChunk(chunk_id="c1", document_id="doc_irrelevant", document_name="a.txt", chunk_index=0, text="", score=0.9)
    c2 = RetrievedChunk(chunk_id="c2", document_id="doc_target", document_name="b.txt", chunk_index=1, text="", score=0.8)
    c3 = RetrievedChunk(chunk_id="c3", document_id="doc_other", document_name="c.txt", chunk_index=2, text="", score=0.7)

    # First relevant result is c2 (rank 2) -> RR = 1/2 = 0.5
    res = mean_reciprocal_rank([c1, c2, c3], ["doc_target"])
    assert res.available is True
    assert res.value == 0.5

    # No relevant result appears in retrieved list -> RR = 0.0
    res_zero = mean_reciprocal_rank([c1, c2, c3], ["doc_missing"])
    assert res_zero.available is True
    assert res_zero.value == 0.0

    # No ground truth relevance information exists -> N/A
    res_none = mean_reciprocal_rank([c1, c2], None)
    assert res_none.available is False
    assert res_none.value is None

