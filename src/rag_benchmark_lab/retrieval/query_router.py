from __future__ import annotations

import re

from ..schemas.documents import RetrievalStrategy
from ..schemas.retrieval import RoutingDecision


RELATIONSHIP_TERMS = {
    "connected",
    "relation",
    "relationship",
    "between",
    "linked",
    "associated",
    "which",
    "how",
    "why",
    "multi-hop",
    "compare",
    "compare",
    "through",
}


def route_query(question: str, manual_strategy: RetrievalStrategy | None = None) -> RoutingDecision:
    if manual_strategy is not None:
        return RoutingDecision(
            selected_strategy=manual_strategy,
            rationale=f"Manual override selected {manual_strategy.value}",
            confidence=1.0,
            considered_strategies=[RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID],
            overridden=True,
        )

    if not question or not question.strip():
        return RoutingDecision(
            selected_strategy=RetrievalStrategy.VECTOR,
            rationale="Empty query defaults to VECTOR retrieval with low confidence",
            confidence=0.1,
            considered_strategies=[RetrievalStrategy.VECTOR],
            overridden=False,
        )

    lowered = question.lower()
    entities = len(re.findall(r"\b[A-Z][a-zA-Z0-9_]+\b", question))
    relationship_signals = sum(1 for term in RELATIONSHIP_TERMS if term in lowered)

    if "compare" in lowered or "benchmark" in lowered:
        return RoutingDecision(RetrievalStrategy.ALL, "Comparison query benefits from all strategies", 0.77, [RetrievalStrategy.ALL, RetrievalStrategy.HYBRID])
    if entities >= 2 or relationship_signals >= 2:
        return RoutingDecision(RetrievalStrategy.HYBRID, "Multiple entities or relationship cues suggest multi-hop evidence", 0.71, [RetrievalStrategy.HYBRID, RetrievalStrategy.GRAPH, RetrievalStrategy.VECTOR])
    if relationship_signals >= 1:
        return RoutingDecision(RetrievalStrategy.GRAPH, "Relationship-oriented wording detected", 0.62, [RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID])
    return RoutingDecision(RetrievalStrategy.VECTOR, "Simple factual query without strong graph cues", 0.66, [RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID])

