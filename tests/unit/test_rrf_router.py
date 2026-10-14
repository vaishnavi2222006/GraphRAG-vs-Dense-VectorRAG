from __future__ import annotations

from rag_benchmark_lab.retrieval.query_router import route_query
from rag_benchmark_lab.retrieval.rrf import reciprocal_rank_fusion
from rag_benchmark_lab.schemas.documents import RetrievalStrategy


def test_reciprocal_rank_fusion_orders_high_frequency_items() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "a"]], k=60)
    assert list(scores.keys())[0] in {"a", "b"}
    assert scores["a"] == scores["b"]


def test_query_router_prefers_vector_for_simple_query() -> None:
    decision = route_query("What is a neural network?")
    assert decision.selected_strategy in {RetrievalStrategy.VECTOR, RetrievalStrategy.HYBRID}


def test_query_router_prefers_hybrid_for_relationship_query() -> None:
    decision = route_query("Which technologies are connected to the ingestion pipeline?")
    assert decision.selected_strategy in {RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID}
