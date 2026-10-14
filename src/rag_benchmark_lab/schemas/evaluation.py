from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class EvaluationExample:
    query_id: str
    question: str
    expected_answer: str | None = None
    reference_contexts: list[str] = field(default_factory=list)
    relevant_document_ids: list[str] = field(default_factory=list)
    query_type: str | None = None
    difficulty: str | None = None
    expected_reasoning_hops: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MetricResult:
    name: str
    value: float | None
    available: bool = True
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class EvaluationRunSummary:
    run_id: str
    dataset_id: str
    dataset_hash: str
    strategy: str
    total_queries: int
    passed_queries: int
    failed_queries: int
    metrics: dict[str, MetricResult] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
