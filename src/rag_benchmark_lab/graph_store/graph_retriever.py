from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas.documents import GraphEvidence, RetrievedChunk
from ..utils.errors import ServiceUnavailableError


def _query_terms(question: str) -> list[str]:
    terms = [term for term in re.findall(r"[A-Za-z][A-Za-z0-9_\-]{2,}", question) if term.lower() not in {"what", "which", "when", "where", "why", "how", "does", "do", "is", "are", "the", "and", "with", "that", "from", "into", "about", "connected"}]
    return terms[:8]


@dataclass(slots=True)
class GraphRetrievalResult:
    evidence: GraphEvidence
    source_chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    warnings: list[str]


class GraphRetriever:
    def __init__(self, neo4j_store, max_hops: int = 2, max_results: int = 20) -> None:
        self.neo4j_store = neo4j_store
        self.max_hops = max_hops
        self.max_results = max_results

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> GraphRetrievalResult:
        from time import perf_counter

        start = perf_counter()
        warnings: list[str] = []
        try:
            terms = _query_terms(question)
            entities = self.neo4j_store.search_entities(terms, limit=self.max_results)
            if document_ids:
                entities = [entity for entity in entities if entity.document_id in document_ids or entity.document_id is None]
            if not entities:
                warnings.append("no_graph_entities_found")
                evidence = GraphEvidence(entities=[], relationships=[], source_chunks=[])
                return GraphRetrievalResult(evidence=evidence, source_chunks=[], retrieval_latency_ms=(perf_counter() - start) * 1000.0, warnings=warnings)
            evidence = self.neo4j_store.traverse([entity.entity_id for entity in entities], max_hops=self.max_hops, limit=self.max_results)
            source_chunks = evidence.source_chunks
            if not source_chunks:
                warnings.append("no_graph_source_chunks_found")
            return GraphRetrievalResult(evidence=evidence, source_chunks=source_chunks, retrieval_latency_ms=(perf_counter() - start) * 1000.0, warnings=warnings)
        except Exception as exc:
            warnings.append(f"graph_retrieval_failed:{exc.__class__.__name__}")
            return GraphRetrievalResult(
                evidence=GraphEvidence(entities=[], relationships=[], source_chunks=[]),
                source_chunks=[],
                retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                warnings=warnings,
            )

