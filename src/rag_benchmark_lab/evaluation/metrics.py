from __future__ import annotations

from collections import Counter
import re
from statistics import median
from typing import Iterable

from ..schemas.evaluation import MetricResult
from ..schemas.documents import AnswerResult, RetrievedChunk, RetrievalStrategy


STOPWORDS = {
    "what", "which", "when", "where", "why", "how", "does", "do", "did", "is", "are", "was", "were",
    "the", "and", "with", "that", "from", "into", "about", "for", "in", "on", "at", "to", "by", "an", "a",
    "of", "or", "as", "if", "not", "no", "nor", "so", "up", "out", "then", "than", "too", "very", "can",
    "may", "just", "now", "also", "any", "some", "such", "only", "own", "same", "our", "my", "your",
    "their", "his", "her", "its", "it", "them", "they", "we", "you", "me", "him", "us", "this", "these",
    "those", "have", "has", "had", "been", "will", "would", "could", "should", "tell", "describe", "explain"
}


def _normalize(text: str) -> str:
    return " ".join("".join(char.lower() if char.isalnum() or char.isspace() else " " for char in text).split())


def _clean_offline_disclaimers(text: str) -> str:
    cleaned = re.sub(r"^\[Development mode:.*?\]\s*", "", text, flags=re.IGNORECASE)
    cleaned = re.sub(r"\*\(Note:.*?\)\*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def exact_match(prediction: str, reference: str | None, is_failed: bool = False) -> MetricResult:
    if is_failed:
        return MetricResult(name="exact_match", value=None, available=False, details={"reason": "generation_failed"})
    if reference is None or not str(reference).strip():
        return MetricResult(name="exact_match", value=None, available=False, details={"reason": "missing_reference_answer"})
    cleaned = _clean_offline_disclaimers(prediction)
    return MetricResult(name="exact_match", value=float(_normalize(cleaned) == _normalize(reference)))


def token_f1(prediction: str, reference: str | None, is_failed: bool = False) -> MetricResult:
    if is_failed:
        return MetricResult(name="token_f1", value=None, available=False, details={"reason": "generation_failed"})
    if reference is None or not str(reference).strip():
        return MetricResult(name="token_f1", value=None, available=False, details={"reason": "missing_reference_answer"})
    cleaned = _clean_offline_disclaimers(prediction)
    pred_tokens = _normalize(cleaned).split()
    ref_tokens = _normalize(reference).split()
    if not pred_tokens or not ref_tokens:
        return MetricResult(name="token_f1", value=0.0)
    overlap = Counter(pred_tokens) & Counter(ref_tokens)
    common = sum(overlap.values())
    precision = common / len(pred_tokens)
    recall = common / len(ref_tokens)
    if precision + recall == 0:
        return MetricResult(name="token_f1", value=0.0)
    return MetricResult(name="token_f1", value=round(2 * precision * recall / (precision + recall), 4))


def answer_correctness(prediction: str, reference: str | None, is_failed: bool = False) -> MetricResult:
    """Evaluate factual correctness against a known reference answer.

    Uses deterministic exact match and token/fact overlap.
    Clearly marked as objective ground truth when reference exists.
    Returns N/A (None) if reference answer is unavailable or generation failed.
    """
    if is_failed:
        return MetricResult(
            name="answer_correctness",
            value=None,
            available=False,
            details={"reason": "generation_failed", "is_ground_truth": False},
        )
    if reference is None or not str(reference).strip():
        return MetricResult(
            name="answer_correctness",
            value=None,
            available=False,
            details={"reason": "missing_reference_answer", "is_ground_truth": False},
        )

    cleaned = _clean_offline_disclaimers(prediction)
    em_res = exact_match(cleaned, reference)
    if em_res.value == 1.0:
        return MetricResult(
            name="answer_correctness",
            value=1.0,
            available=True,
            details={"is_ground_truth": True, "method": "deterministic_exact_match"},
        )

    f1_res = token_f1(cleaned, reference)
    f1_val = f1_res.value or 0.0

    ref_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", reference))
    pred_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", cleaned))

    if ref_numbers:
        num_recall = len(ref_numbers & pred_numbers) / len(ref_numbers)
        score = round(0.4 * f1_val + 0.6 * num_recall, 4)
    else:
        score = f1_val

    return MetricResult(
        name="answer_correctness",
        value=score,
        available=True,
        details={
            "is_ground_truth": True,
            "method": "deterministic_token_and_fact_overlap",
            "token_f1": f1_val,
        },
    )


def completeness(prediction: str, reference: str | None, is_failed: bool = False) -> MetricResult:
    """Measure proportion of essential information from reference answer covered by prediction."""
    if is_failed:
        return MetricResult(name="completeness", value=None, available=False, details={"reason": "generation_failed"})
    if reference is None or not str(reference).strip():
        return MetricResult(name="completeness", value=None, available=False, details={"reason": "missing_reference_answer"})

    cleaned = _clean_offline_disclaimers(prediction)
    ref_norm = _normalize(reference)
    ref_tokens = [t for t in ref_norm.split() if len(t) > 2 and t not in STOPWORDS]
    if not ref_tokens:
        return MetricResult(name="completeness", value=1.0, available=True)

    pred_tokens = set(_normalize(cleaned).split())
    covered = sum(1 for t in ref_tokens if t in pred_tokens)
    score = round(min(1.0, covered / len(ref_tokens)), 4)
    return MetricResult(name="completeness", value=score, available=True)


def citation_correctness(answer: AnswerResult, is_direct: bool = False, is_failed: bool = False) -> MetricResult:
    """Verify that cited sources exist in retrieved evidence and support the claims.

    Marked N/A for Direct LLM.
    """
    if is_direct:
        return MetricResult(
            name="citation_correctness",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )
    if is_failed:
        return MetricResult(
            name="citation_correctness",
            value=None,
            available=False,
            details={"reason": "generation_failed"},
        )
    retrieved_sources = getattr(answer, "retrieved_sources", []) or []
    citations = getattr(answer, "citations", []) or []
    ans_text = getattr(answer, "answer", "")

    if not retrieved_sources:
        return MetricResult(
            name="citation_correctness",
            value=None,
            available=False,
            details={"reason": "no_retrieval_evidence"},
        )
    if not citations:
        return MetricResult(
            name="citation_correctness",
            value=0.0,
            available=True,
            details={"reason": "no_citations_provided", "citation_count": 0},
        )

    retrieved_chunk_ids = {getattr(c, "chunk_id", "") for c in retrieved_sources}
    retrieved_doc_ids = {getattr(c, "document_id", "") for c in retrieved_sources} | {getattr(c, "document_name", "") for c in retrieved_sources}
    cleaned_ans = _clean_offline_disclaimers(ans_text)
    ans_terms = set(t for t in _normalize(cleaned_ans).split() if len(t) > 3 and t not in STOPWORDS)

    valid_supporting = 0
    for cit in citations:
        c_chunk_id = getattr(cit, "chunk_id", "")
        c_doc_id = getattr(cit, "document_id", "")
        c_doc_name = getattr(cit, "document_name", "")
        is_known = (c_chunk_id in retrieved_chunk_ids) or (c_doc_id in retrieved_doc_ids) or (c_doc_name in retrieved_doc_ids)
        if is_known:
            matching_chunks = [c for c in retrieved_sources if getattr(c, "chunk_id", "") == c_chunk_id or getattr(c, "document_id", "") == c_doc_id]
            if matching_chunks:
                chunk_terms = set(t for t in _normalize(getattr(matching_chunks[0], "text", "")).split() if len(t) > 3 and t not in STOPWORDS)
                if ans_terms and (ans_terms & chunk_terms):
                    valid_supporting += 1
                elif not ans_terms:
                    valid_supporting += 1
            else:
                valid_supporting += 1

    score = round(valid_supporting / max(1, len(citations)), 4)
    return MetricResult(
        name="citation_correctness",
        value=score,
        available=True,
        details={"valid_citations": valid_supporting, "total_citations": len(citations)},
    )



def _normalize_doc_ref(val: str) -> str:
    return val.lower().replace("-", "_").removesuffix(".txt").removesuffix(".pdf").removesuffix(".docx").strip()


def retrieval_recall_at_k(
    retrieved: list[RetrievedChunk],
    relevant_document_ids: list[str] | None,
    k: int | None = None,
    is_direct: bool = False,
) -> MetricResult:
    if is_direct:
        return MetricResult(
            name="retrieval_recall_at_k",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )
    if not relevant_document_ids:
        return MetricResult(
            name="retrieval_recall_at_k",
            value=None,
            available=False,
            details={"reason": "missing_relevant_document_ids"},
        )

    clean_relevant = [str(r).strip() for r in relevant_document_ids if r and str(r).strip()]
    if not clean_relevant:
        return MetricResult(
            name="retrieval_recall_at_k",
            value=None,
            available=False,
            details={"reason": "missing_relevant_document_ids"},
        )

    raw_relevant_set = set(clean_relevant)
    total_known_relevant = len(raw_relevant_set)

    retrieved_items = retrieved[:k] if (k is not None and k > 0) else retrieved

    matched_relevant: set[str] = set()
    for chunk in retrieved_items:
        c_doc_id = getattr(chunk, "document_id", "") or ""
        c_doc_name = getattr(chunk, "document_name", "") or ""
        c_chunk_id = getattr(chunk, "chunk_id", "") or ""
        for r in raw_relevant_set:
            r_norm = _normalize_doc_ref(r)
            if (
                c_doc_id == r
                or c_doc_name == r
                or c_chunk_id == r
                or _normalize_doc_ref(c_doc_id) == r_norm
                or _normalize_doc_ref(c_doc_name) == r_norm
                or _normalize_doc_ref(c_chunk_id) == r_norm
            ):
                matched_relevant.add(r)

    score = round(len(matched_relevant) / total_known_relevant, 4)
    return MetricResult(
        name="retrieval_recall_at_k",
        value=score,
        available=True,
        details={
            "matched_relevant": sorted(list(matched_relevant)),
            "total_known_relevant": total_known_relevant,
            "retrieved_count": len(retrieved_items),
            "k": k or len(retrieved_items),
        },
    )


def hit_rate_at_k(
    retrieved: list[RetrievedChunk],
    relevant_document_ids: list[str] | None,
    is_direct: bool = False,
) -> MetricResult:
    if is_direct:
        return MetricResult(name="hit_rate_at_k", value=None, available=False, details={"reason": "not_applicable_for_direct_llm"})
    if not relevant_document_ids:
        return MetricResult(name="hit_rate_at_k", value=None, available=False, details={"reason": "missing_relevant_document_ids"})
    for chunk in retrieved:
        for r in relevant_document_ids:
            if chunk.document_id == r or chunk.document_name == r or _normalize_doc_ref(chunk.document_id) == _normalize_doc_ref(r) or _normalize_doc_ref(chunk.document_name) == _normalize_doc_ref(r):
                return MetricResult(name="hit_rate_at_k", value=1.0)
    return MetricResult(name="hit_rate_at_k", value=0.0)


def mean_reciprocal_rank(
    retrieved: list[RetrievedChunk],
    relevant_document_ids: list[str] | None,
    is_direct: bool = False,
) -> MetricResult:
    if is_direct:
        return MetricResult(
            name="mrr",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )
    if not relevant_document_ids:
        return MetricResult(
            name="mrr",
            value=None,
            available=False,
            details={"reason": "missing_relevant_document_ids"},
        )

    clean_relevant = [str(r).strip() for r in relevant_document_ids if r and str(r).strip()]
    if not clean_relevant:
        return MetricResult(
            name="mrr",
            value=None,
            available=False,
            details={"reason": "missing_relevant_document_ids"},
        )

    raw_relevant_set = set(clean_relevant)

    for rank, chunk in enumerate(retrieved, start=1):
        c_doc_id = getattr(chunk, "document_id", "") or ""
        c_doc_name = getattr(chunk, "document_name", "") or ""
        c_chunk_id = getattr(chunk, "chunk_id", "") or ""
        for r in raw_relevant_set:
            r_norm = _normalize_doc_ref(r)
            if (
                c_doc_id == r
                or c_doc_name == r
                or c_chunk_id == r
                or _normalize_doc_ref(c_doc_id) == r_norm
                or _normalize_doc_ref(c_doc_name) == r_norm
                or _normalize_doc_ref(c_chunk_id) == r_norm
            ):
                return MetricResult(
                    name="mrr",
                    value=round(1.0 / rank, 4),
                    available=True,
                    details={"first_rank": rank, "matched_identifier": r},
                )

    return MetricResult(
        name="mrr",
        value=0.0,
        available=True,
        details={"first_rank": None, "reason": "no_relevant_chunk_in_retrieved"},
    )


def extract_reference_facts(text: str) -> list[str]:
    """Decompose reference answer into discrete factual statements or clauses."""
    if not text or not str(text).strip():
        return []
    cleaned = str(text).strip()

    # Split into lines or bullet points first
    lines = [line.strip().lstrip("-*•0123456789. ") for line in cleaned.splitlines() if line.strip()]
    raw_units: list[str] = []
    for line in lines:
        # Split on sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", line)
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            # Split compound coordinating clauses: ", and ", ", while ", ", but ", "; "
            sub_clauses = re.split(r"(?:;\s*|,\s+(?:and|while|but|whereas|as well as)\s+)", s, flags=re.IGNORECASE)
            for clause in sub_clauses:
                c_clean = clause.strip().rstrip(".,;!?")
                if c_clean and len(c_clean) > 3:
                    raw_units.append(c_clean)

    if not raw_units and cleaned:
        raw_units = [cleaned.rstrip(".,;!?")]

    seen: set[str] = set()
    facts: list[str] = []
    for u in raw_units:
        u_norm = _normalize(u)
        if u_norm and u_norm not in seen:
            seen.add(u_norm)
            facts.append(u)

    return facts


def context_recall(
    retrieved_contexts: list[str] | list[RetrievedChunk] | None,
    reference_answer: str | list[str] | None = None,
    reference_contexts: list[str] | None = None,
    is_direct: bool = False,
) -> MetricResult:
    """Calculate Context Recall as the proportion of required reference facts supported by retrieved context."""
    if is_direct:
        return MetricResult(
            name="context_recall",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )

    # Handle backwards compatibility if reference_contexts was passed as 2nd positional arg
    if isinstance(reference_answer, list) and reference_contexts is None:
        reference_contexts = reference_answer
        reference_answer = " ".join(reference_contexts)

    target_answer = reference_answer
    if not target_answer or not str(target_answer).strip():
        if reference_contexts and any(str(c).strip() for c in reference_contexts):
            target_answer = " ".join(str(c) for c in reference_contexts)
        else:
            return MetricResult(
                name="context_recall",
                value=None,
                available=False,
                details={"reason": "missing_reference_answer"},
            )

    raw_retrieved = retrieved_contexts or []
    ret_texts: list[str] = []
    for item in raw_retrieved:
        if hasattr(item, "text"):
            ret_texts.append(getattr(item, "text", "") or "")
        else:
            ret_texts.append(str(item))

    facts = extract_reference_facts(target_answer)
    if not facts:
        return MetricResult(
            name="context_recall",
            value=None,
            available=False,
            details={"reason": "missing_reference_answer"},
        )

    joined_context = " ".join(_normalize(t) for t in ret_texts)
    context_tokens = set(joined_context.split())

    if not joined_context.strip():
        return MetricResult(
            name="context_recall",
            value=0.0,
            available=True,
            details={
                "supported_facts": 0,
                "total_facts": len(facts),
                "facts": facts,
                "reason": "empty_retrieved_context",
            },
        )

    supported_facts = 0
    fact_details = []

    for fact in facts:
        fact_norm = _normalize(fact)
        tokens = [t for t in fact_norm.split() if len(t) >= 3 and t not in STOPWORDS]
        if not tokens:
            tokens = [t for t in fact_norm.split() if t]

        is_supported = False
        if fact_norm in joined_context:
            is_supported = True
        elif tokens:
            overlap = [t for t in tokens if t in context_tokens]
            if len(tokens) == 1:
                is_supported = (len(overlap) == 1)
            elif len(tokens) == 2:
                is_supported = (len(overlap) >= 1)
            else:
                is_supported = (len(overlap) / len(tokens) >= 0.55) or (len(overlap) >= 3)

        if is_supported:
            supported_facts += 1
        fact_details.append({"fact": fact, "supported": is_supported})

    score = round(supported_facts / max(1, len(facts)), 4)
    return MetricResult(
        name="context_recall",
        value=score,
        available=True,
        details={
            "supported_facts": supported_facts,
            "total_facts": len(facts),
            "fact_details": fact_details,
        },
    )


def context_precision(
    retrieved_contexts: list[str] | list[RetrievedChunk] | None,
    reference_answer: str | list[str] | None = None,
    relevant_document_ids: list[str] | None = None,
    question: str | None = None,
    reference_contexts: list[str] | None = None,
    is_direct: bool = False,
) -> MetricResult:
    """Calculate ranking-aware Context Precision across retrieved chunks using ground-truth labels or reference answer."""
    if is_direct:
        return MetricResult(
            name="context_precision",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )

    # Handle backwards compatibility if reference_contexts was passed as 2nd positional arg
    if isinstance(reference_answer, list) and reference_contexts is None:
        reference_contexts = reference_answer
        reference_answer = " ".join(reference_contexts)

    has_labels = bool(relevant_document_ids and any(str(r).strip() for r in relevant_document_ids))
    has_ref_answer = bool(reference_answer and str(reference_answer).strip())
    has_ref_contexts = bool(reference_contexts and any(str(c).strip() for c in reference_contexts))

    if not (has_labels or has_ref_answer or has_ref_contexts):
        return MetricResult(
            name="context_precision",
            value=None,
            available=False,
            details={"reason": "missing_ground_truth_and_reference_answer"},
        )

    raw_retrieved = retrieved_contexts or []
    if not raw_retrieved:
        return MetricResult(
            name="context_precision",
            value=0.0,
            available=True,
            details={"reason": "no_retrieved_chunks", "total_retrieved": 0},
        )

    clean_relevant = [str(r).strip() for r in (relevant_document_ids or []) if r and str(r).strip()]
    raw_relevant_set = set(clean_relevant)

    ref_norm = _normalize(str(reference_answer or ""))
    ref_terms = [t for t in ref_norm.split() if len(t) >= 3 and t not in STOPWORDS]

    q_norm = _normalize(str(question or ""))
    q_terms = [t for t in q_norm.split() if len(t) >= 3 and t not in STOPWORDS]

    ref_ctx_terms: list[str] = []
    if reference_contexts:
        joined_ref_ctx = " ".join(_normalize(c) for c in reference_contexts)
        ref_ctx_terms = [t for t in joined_ref_ctx.split() if len(t) >= 3 and t not in STOPWORDS]

    relevance_vector: list[int] = []

    for rank_idx, chunk in enumerate(raw_retrieved, start=1):
        is_rel = False
        c_doc_id = getattr(chunk, "document_id", "") or ""
        c_doc_name = getattr(chunk, "document_name", "") or ""
        c_chunk_id = getattr(chunk, "chunk_id", "") or ""
        c_text = getattr(chunk, "text", str(chunk)) or ""
        c_norm = _normalize(c_text)
        c_tokens = set(c_norm.split())

        # 1. Match against ground truth document/chunk IDs
        if raw_relevant_set:
            for r in raw_relevant_set:
                r_norm = _normalize_doc_ref(r)
                if (
                    c_doc_id == r
                    or c_doc_name == r
                    or c_chunk_id == r
                    or _normalize_doc_ref(c_doc_id) == r_norm
                    or _normalize_doc_ref(c_doc_name) == r_norm
                    or _normalize_doc_ref(c_chunk_id) == r_norm
                ):
                    is_rel = True
                    break

        # 2. Match against reference answer / reference context content
        if not is_rel and (ref_terms or ref_ctx_terms):
            ref_overlap = len(set(ref_terms) & c_tokens) if ref_terms else 0
            ref_ctx_overlap = len(set(ref_ctx_terms) & c_tokens) if ref_ctx_terms else 0
            q_overlap = len(set(q_terms) & c_tokens) if q_terms else 0

            if ref_terms:
                if len(ref_terms) <= 2:
                    is_rel = (ref_overlap >= 1)
                else:
                    is_rel = (ref_overlap >= 2) or (ref_overlap / len(ref_terms) >= 0.25) or (ref_overlap >= 1 and q_overlap >= 1)

            if not is_rel and ref_ctx_overlap >= 2:
                is_rel = True

        relevance_vector.append(1 if is_rel else 0)

    total_relevant = sum(relevance_vector)
    K = len(relevance_vector)

    if total_relevant == 0:
        return MetricResult(
            name="context_precision",
            value=0.0,
            available=True,
            details={
                "precision_at_k": [0.0] * K,
                "relevance_vector": relevance_vector,
                "relevant_retrieved": 0,
                "total_retrieved": K,
            },
        )

    # Compute ranking-aware Average Precision: sum(Precision@k * rel(k)) / total_relevant
    running_rel = 0
    precisions_at_k: list[float] = []
    ap_numerator = 0.0

    for k, rel in enumerate(relevance_vector, start=1):
        running_rel += rel
        p_at_k = running_rel / k
        precisions_at_k.append(round(p_at_k, 4))
        if rel == 1:
            ap_numerator += p_at_k

    average_precision = round(ap_numerator / total_relevant, 4)

    return MetricResult(
        name="context_precision",
        value=average_precision,
        available=True,
        details={
            "precision_at_k": precisions_at_k,
            "relevance_vector": relevance_vector,
            "relevant_retrieved": total_relevant,
            "total_retrieved": K,
        },
    )


def faithfulness(
    prediction: str,
    retrieved_contexts: list[str],
    is_direct: bool = False,
    is_failed: bool = False,
) -> MetricResult:
    """Measure whether claims in answer are grounded in retrieved evidence without hallucination."""
    if is_direct:
        return MetricResult(
            name="faithfulness",
            value=None,
            available=False,
            details={"reason": "not_applicable_for_direct_llm"},
        )
    if is_failed:
        return MetricResult(name="faithfulness", value=None, available=False, details={"reason": "generation_failed"})
    if not retrieved_contexts:
        return MetricResult(name="faithfulness", value=0.0, available=True, details={"reason": "no_retrieved_contexts"})

    cleaned = _clean_offline_disclaimers(prediction)
    pred_clean = _normalize(cleaned)
    if not pred_clean:
        return MetricResult(name="faithfulness", value=0.0)

    context_tokens = set(" ".join(_normalize(t) for t in retrieved_contexts).split())
    content_tokens = [t for t in pred_clean.split() if len(t) > 3 and t not in STOPWORDS]
    if not content_tokens:
        return MetricResult(name="faithfulness", value=1.0)

    grounded = sum(1 for t in content_tokens if t in context_tokens)
    score = round(min(1.0, grounded / len(content_tokens)), 4)
    return MetricResult(name="faithfulness", value=score)


def answer_relevancy(question: str, prediction: str, is_failed: bool = False) -> MetricResult:
    """Measure whether the generated answer addresses the user's specific query."""
    if is_failed:
        return MetricResult(name="answer_relevancy", value=None, available=False, details={"reason": "generation_failed"})

    cleaned = _clean_offline_disclaimers(prediction)
    q_tokens = set(t for t in _normalize(question).split() if len(t) > 3 and t not in STOPWORDS)
    if not q_tokens:
        return MetricResult(name="answer_relevancy", value=1.0)
    pred_tokens = set(_normalize(cleaned).split())
    overlap = len(q_tokens & pred_tokens)
    score = round(min(1.0, overlap / len(q_tokens)), 4)
    return MetricResult(name="answer_relevancy", value=score)


def latency_summary(latencies_ms: list[float]) -> dict[str, float | None]:
    valid = [lat for lat in latencies_ms if lat is not None and lat >= 0]
    if not valid:
        return {"mean": None, "median": None, "p95": None}
    values = sorted(valid)
    p95_index = min(len(values) - 1, int(round(len(values) * 0.95)) - 1 if len(values) > 1 else 0)
    return {
        "mean": round(sum(values) / len(values), 2),
        "median": round(median(values), 2),
        "p95": round(values[p95_index], 2) if len(values) >= 20 else None,
    }


def metric_bundle(
    answer: AnswerResult,
    reference_answer: str | None,
    relevant_document_ids: list[str] | None,
    reference_contexts: list[str] | None,
    question: str | None = None,
) -> dict[str, MetricResult]:
    """Calculate the complete evaluation matrix for an approach's answer result."""
    strat = getattr(answer, "selected_strategy", None)
    is_direct = (strat == RetrievalStrategy.DIRECT or strat == "DIRECT")
    errors = getattr(answer, "errors", []) or []
    metadata = getattr(answer, "metadata", {}) or {}
    is_failed = bool(errors) or (metadata.get("generation_status") == "failed")

    retrieved_sources = getattr(answer, "retrieved_sources", []) or []
    retrieved_texts = [getattr(chunk, "text", "") for chunk in retrieved_sources]
    graph_evidence = getattr(answer, "graph_evidence", None)
    if graph_evidence and getattr(graph_evidence, "relationships", None):
        for rel in graph_evidence.relationships:
            src_name = getattr(rel, "source_entity_name", "")
            rel_type = getattr(rel, "relationship_type", "")
            tgt_name = getattr(rel, "target_entity_name", "")
            src_text = getattr(rel, "source_text", "")
            rel_str = f"{src_name} {rel_type} {tgt_name}"
            if src_text:
                rel_str += f" {src_text}"
            retrieved_texts.append(rel_str)

    ans_text = getattr(answer, "answer", "")

    return {
        "answer_correctness": answer_correctness(ans_text, reference_answer, is_failed=is_failed),
        "faithfulness": faithfulness(ans_text, retrieved_texts, is_direct=is_direct, is_failed=is_failed),
        "answer_relevancy": answer_relevancy(question or "", ans_text, is_failed=is_failed),
        "completeness": completeness(ans_text, reference_answer, is_failed=is_failed),
        "citation_correctness": citation_correctness(answer, is_direct=is_direct, is_failed=is_failed),
        "context_precision": context_precision(
            retrieved_sources or retrieved_texts,
            reference_answer=reference_answer,
            relevant_document_ids=relevant_document_ids,
            question=question,
            reference_contexts=reference_contexts,
            is_direct=is_direct,
        ),
        "context_recall": context_recall(
            retrieved_texts,
            reference_answer=reference_answer,
            reference_contexts=reference_contexts,
            is_direct=is_direct,
        ),
        "retrieval_recall_at_k": retrieval_recall_at_k(retrieved_sources, relevant_document_ids, is_direct=is_direct),
        "hit_rate_at_k": hit_rate_at_k(retrieved_sources, relevant_document_ids, is_direct=is_direct),
        "mrr": mean_reciprocal_rank(retrieved_sources, relevant_document_ids, is_direct=is_direct),
        "exact_match": exact_match(ans_text, reference_answer, is_failed=is_failed),
        "token_f1": token_f1(ans_text, reference_answer, is_failed=is_failed),
    }




