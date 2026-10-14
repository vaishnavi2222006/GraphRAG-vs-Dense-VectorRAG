from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .documents import Citation, RetrievedChunk, RetrievalStrategy


@dataclass(slots=True)
class RoutingDecision:
    selected_strategy: RetrievalStrategy
    rationale: str
    confidence: float
    considered_strategies: list[RetrievalStrategy] = field(default_factory=list)
    overridden: bool = False


@dataclass(slots=True)
class RetrievalBundle:
    strategy: RetrievalStrategy
    chunks: list[RetrievedChunk] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
