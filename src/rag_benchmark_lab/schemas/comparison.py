from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class ComparisonStrategyResultRecord:
    result_id: str
    run_id: str
    strategy: str  # "VECTOR", "GRAPH", "HYBRID"
    strategy_label: str  # "Vector RAG", "GraphRAG", "Hybrid RAG"
    status: str  # "success", "partial", "failed"
    answer: str
    error_message: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    model_name: str | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    evidence_sources: list[dict[str, Any]] = field(default_factory=list)
    graph_evidence: dict[str, Any] | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass(slots=True)
class ComparisonRunRecord:
    run_id: str
    user_id: str
    kb_id: str
    query: str
    status: str  # "completed", "partial", "failed"
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: dict[str, Any] = field(default_factory=dict)
    strategies: list[str] = field(default_factory=list)
    results: list[ComparisonStrategyResultRecord] = field(default_factory=list)
    kb_name: str | None = None
