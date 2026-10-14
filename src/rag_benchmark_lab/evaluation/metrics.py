from __future__ import annotations

from collections import Counter
from statistics import median
from typing import Iterable

from ..schemas.evaluation import MetricResult
from ..schemas.documents import AnswerResult, RetrievedChunk


def _normalize(text: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text).split())


def exact_match(prediction: str, reference: str | None) -> MetricResult:
    if reference is None:
        return MetricResult(name="exact_match", value=None, available=False, details={"reason": "missing_reference_answer"})
    return MetricResult(name="exact_match", value=float(_normalize(prediction) == _normalize(reference)))


def token_f1(prediction: str, reference: str | None) -> MetricResult:
    if reference is None:
        return MetricResult(name="token_f1", value=None, available=False, details={"reason": "missing_reference_answer"})
    pred_tokens = _normalize(prediction).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return MetricResult(name="token_f1", value=0.0)
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return MetricResult(name="token_f1", value=0.0)
    return MetricResult(name="token_f1", value=2 * precision * recall / (precision + recall))


def _normalize_doc_ref(val: str) -> str:
    return val.lower().replace("-", "_").removesuffix(".txt").removesuffix(".pdf").removesuffix(".docx").strip()


def retrieval_recall_at_k(retrieved: list[RetrievedChunk], relevant_document_ids: list[str] | None) -> MetricResult:
    if not relevant_document_ids:
        return MetricResult(name="retrieval_recall_at_k", value=None, available=False, details={"reason": "missing_relevant_document_ids"})
    raw_relevant = set(relevant_document_ids)
    norm_relevant = {_normalize_doc_ref(r) for r in relevant_document_ids}
    matched_relevant = set()
    for chunk in retrieved:
        for r in relevant_document_ids:
            if chunk.document_id == r or chunk.document_name == r or _normalize_doc_ref(chunk.document_id) == _normalize_doc_ref(r) or _normalize_doc_ref(chunk.document_name) == _normalize_doc_ref(r):
                matched_relevant.add(r)
    return MetricResult(name="retrieval_recall_at_k", value=len(matched_relevant) / len(raw_relevant))


def hit_rate_at_k(retrieved: list[RetrievedChunk], relevant_document_ids: list[str] | None) -> MetricResult:
    if not relevant_document_ids:
        return MetricResult(name="hit_rate_at_k", value=None, available=False, details={"reason": "missing_relevant_document_ids"})
    for chunk in retrieved:
        for r in relevant_document_ids:
            if chunk.document_id == r or chunk.document_name == r or _normalize_doc_ref(chunk.document_id) == _normalize_doc_ref(r) or _normalize_doc_ref(chunk.document_name) == _normalize_doc_ref(r):
                return MetricResult(name="hit_rate_at_k", value=1.0)
    return MetricResult(name="hit_rate_at_k", value=0.0)


def mean_reciprocal_rank(retrieved: list[RetrievedChunk], relevant_document_ids: list[str] | None) -> MetricResult:
    if not relevant_document_ids:
        return MetricResult(name="mrr", value=None, available=False, details={"reason": "missing_relevant_document_ids"})
    for index, chunk in enumerate(retrieved, start=1):
        for r in relevant_document_ids:
            if chunk.document_id == r or chunk.document_name == r or _normalize_doc_ref(chunk.document_id) == _normalize_doc_ref(r) or _normalize_doc_ref(chunk.document_name) == _normalize_doc_ref(r):
                return MetricResult(name="mrr", value=1.0 / index)
    return MetricResult(name="mrr", value=0.0)


def context_precision(retrieved_contexts: list[str], reference_contexts: list[str] | None) -> MetricResult:
    if not reference_contexts:
        return MetricResult(name="context_precision", value=None, available=False, details={"reason": "missing_reference_contexts"})
    joined_retrieved = " ".join(_normalize(text) for text in retrieved_contexts)
    joined_reference = " ".join(_normalize(text) for text in reference_contexts)
    if not joined_retrieved:
        return MetricResult(name="context_precision", value=0.0)
    overlap = len(set(joined_retrieved.split()) & set(joined_reference.split()))
    return MetricResult(name="context_precision", value=overlap / max(1, len(set(joined_retrieved.split()))))


def context_recall(retrieved_contexts: list[str], reference_contexts: list[str] | None) -> MetricResult:
    if not reference_contexts:
        return MetricResult(name="context_recall", value=None, available=False, details={"reason": "missing_reference_contexts"})
    joined_retrieved = " ".join(_normalize(text) for text in retrieved_contexts)
    joined_reference = " ".join(_normalize(text) for text in reference_contexts)
    if not joined_reference:
        return MetricResult(name="context_recall", value=0.0)
    overlap = len(set(joined_retrieved.split()) & set(joined_reference.split()))
    return MetricResult(name="context_recall", value=overlap / max(1, len(set(joined_reference.split()))))


def latency_summary(latencies_ms: list[float]) -> dict[str, float | None]:
    if not latencies_ms:
        return {"mean": None, "median": None, "p95": None}
    values = sorted(latencies_ms)
    p95_index = min(len(values) - 1, int(round(len(values) * 0.95)) - 1 if len(values) > 1 else 0)
    return {
        "mean": sum(values) / len(values),
        "median": median(values),
        "p95": values[p95_index] if len(values) >= 20 else None,
    }


def metric_bundle(answer: AnswerResult, reference_answer: str | None, relevant_document_ids: list[str] | None, reference_contexts: list[str] | None) -> dict[str, MetricResult]:
    retrieved_texts = [chunk.text for chunk in answer.retrieved_sources]
    return {
        "exact_match": exact_match(answer.answer, reference_answer),
        "token_f1": token_f1(answer.answer, reference_answer),
        "retrieval_recall_at_k": retrieval_recall_at_k(answer.retrieved_sources, relevant_document_ids),
        "hit_rate_at_k": hit_rate_at_k(answer.retrieved_sources, relevant_document_ids),
        "mrr": mean_reciprocal_rank(answer.retrieved_sources, relevant_document_ids),
        "context_precision": context_precision(retrieved_texts, reference_contexts),
        "context_recall": context_recall(retrieved_texts, reference_contexts),
    }

