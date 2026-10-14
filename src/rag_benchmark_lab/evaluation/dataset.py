from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from ..schemas.evaluation import EvaluationExample


def _normalize_entry(payload: dict) -> EvaluationExample:
    return EvaluationExample(
        query_id=str(payload.get("query_id") or payload.get("id") or hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:16]),
        question=str(payload.get("question") or payload.get("prompt") or "").strip(),
        expected_answer=payload.get("expected_answer"),
        reference_contexts=list(payload.get("reference_contexts") or []),
        relevant_document_ids=list(payload.get("relevant_document_ids") or []),
        query_type=payload.get("query_type"),
        difficulty=payload.get("difficulty"),
        expected_reasoning_hops=payload.get("expected_reasoning_hops"),
        metadata={k: v for k, v in payload.items() if k not in {"query_id", "id", "question", "prompt", "expected_answer", "reference_contexts", "relevant_document_ids", "query_type", "difficulty", "expected_reasoning_hops"}},
    )


def validate_example(example: EvaluationExample) -> None:
    if not example.question:
        raise ValueError("Each evaluation example must contain a question")
    if example.expected_reasoning_hops is not None and example.expected_reasoning_hops < 0:
        raise ValueError("expected_reasoning_hops must be non-negative when provided")


def load_dataset(path: str | Path) -> list[EvaluationExample]:
    resolved = Path(path)
    text = resolved.read_text(encoding="utf-8")
    examples: list[EvaluationExample] = []
    if resolved.suffix.lower() == ".jsonl":
        for line in text.splitlines():
            if not line.strip():
                continue
            example = _normalize_entry(json.loads(line))
            validate_example(example)
            examples.append(example)
        return examples
    payload = json.loads(text)
    items = payload if isinstance(payload, list) else payload.get("examples", [])
    for item in items:
        example = _normalize_entry(item)
        validate_example(example)
        examples.append(example)
    return examples


def dataset_hash(examples: list[EvaluationExample]) -> str:
    payload = [asdict(example) for example in examples]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def sample_dataset() -> list[EvaluationExample]:
    return [
        EvaluationExample(
            query_id="q1",
            question="What is the primary purpose of the document ingestion pipeline?",
            expected_answer="It validates, chunks, embeds, and indexes documents while preserving provenance.",
            relevant_document_ids=["sample-doc-1"],
            query_type="factual",
            difficulty="easy",
            expected_reasoning_hops=1,
            metadata={"synthetic": True},
        ),
        EvaluationExample(
            query_id="q2",
            question="Which retrieval strategy is designed to help when facts are distributed across documents?",
            expected_answer="Hybrid RAG can combine vector and graph evidence for distributed facts.",
            relevant_document_ids=["sample-doc-1", "sample-doc-2"],
            query_type="relationship",
            difficulty="medium",
            expected_reasoning_hops=2,
            metadata={"synthetic": True},
        ),
        EvaluationExample(
            query_id="q3",
            question="Which organization is mentioned as introducing the example technology?",
            expected_answer=None,
            query_type="ambiguous",
            difficulty="medium",
            expected_reasoning_hops=2,
            metadata={"synthetic": True},
        ),
    ]

