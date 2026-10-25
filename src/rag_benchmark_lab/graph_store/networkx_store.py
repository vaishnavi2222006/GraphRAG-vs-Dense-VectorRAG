"""NetworkX-based in-memory graph store — a drop-in replacement for Neo4jStore.

This store satisfies the same interface as Neo4jStore but requires no Docker or
external database. All data is kept in a NetworkX DiGraph and synced to the
SQLite registry (which is already used for chunks/entities/relationships), so
nothing is lost across Streamlit restarts.
"""
from __future__ import annotations

from dataclasses import asdict
from typing import Iterable

import networkx as nx

from ..schemas.documents import EntityRecord, GraphEvidence, RelationshipRecord, RetrievedChunk


class NetworkXStore:
    """Lightweight graph store backed by NetworkX DiGraph.

    The store is stateful for the lifetime of the Streamlit process (held in
    st.cache_resource via build_context). Graph data is also written to the
    SQLite DocumentRegistry so it survives across Streamlit restarts.
    """

    def __init__(self) -> None:
        self._graph: nx.DiGraph = nx.DiGraph()
        # entity_id -> EntityRecord (dict form)
        self._entities: dict[str, dict] = {}
        # relationship_id -> RelationshipRecord (dict form)
        self._relationships: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Health
    # ------------------------------------------------------------------

    def health(self) -> bool:
        return True

    def close(self) -> None:
        """Close connection (no-op for NetworkX in-memory store)."""
        pass

    # ------------------------------------------------------------------
    # Schema (no-op for NetworkX)
    # ------------------------------------------------------------------

    def ensure_schema(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def upsert_entities(self, entities: list[EntityRecord]) -> None:
        for entity in entities:
            d = asdict(entity)
            self._entities[entity.entity_id] = d
            self._graph.add_node(entity.entity_id, **d)

    def upsert_relationships(self, relationships: list[RelationshipRecord]) -> None:
        for rel in relationships:
            d = asdict(rel)
            self._relationships[rel.relationship_id] = d
            # Ensure nodes exist even if entities weren't added yet
            if rel.source_entity_id not in self._graph:
                self._graph.add_node(rel.source_entity_id, canonical_name=rel.source_entity_name)
            if rel.target_entity_id not in self._graph:
                self._graph.add_node(rel.target_entity_id, canonical_name=rel.target_entity_name)
            self._graph.add_edge(
                rel.source_entity_id,
                rel.target_entity_id,
                **d,
            )

    def upsert_document(self, document_id: str, document_name: str) -> None:
        # No separate Document node needed; documents tracked in SQLite registry
        pass

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> None:
        # Remove all entities and relationships for this document
        entity_ids_to_remove = [
            eid for eid, edata in self._entities.items()
            if edata.get("document_id") == document_id
        ]
        rel_ids_to_remove = [
            rid for rid, rdata in self._relationships.items()
            if rdata.get("source_document_id") == document_id
        ]
        for eid in entity_ids_to_remove:
            self._entities.pop(eid, None)
            if self._graph.has_node(eid):
                self._graph.remove_node(eid)
        for rid in rel_ids_to_remove:
            self._relationships.pop(rid, None)

    # ------------------------------------------------------------------
    # Search / Traversal
    # ------------------------------------------------------------------

    def search_entities(self, terms: list[str], limit: int = 10) -> list[EntityRecord]:
        lower_terms = [t.lower() for t in terms if t]
        matched: list[tuple[float, EntityRecord]] = []
        for entity_id, data in self._entities.items():
            name = (data.get("canonical_name") or "").lower()
            mention = (data.get("original_mention") or "").lower()
            if any(term in name or term in mention for term in lower_terms):
                record = EntityRecord(
                    entity_id=entity_id,
                    canonical_name=data.get("canonical_name", ""),
                    original_mention=data.get("original_mention", data.get("canonical_name", "")),
                    entity_type=data.get("entity_type", "Concept"),
                    confidence=data.get("confidence"),
                    document_id=data.get("document_id"),
                    chunk_id=data.get("chunk_id"),
                    page_number=data.get("page_number"),
                    extraction_method=data.get("extraction_method", "llm"),
                )
                matched.append((data.get("confidence") or 0.0, record))
        matched.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in matched[:limit]]

    def traverse(self, entity_ids: list[str], max_hops: int = 2, limit: int = 20) -> GraphEvidence:
        safe_hops = max(1, min(int(max_hops), 5))
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()

        # BFS/DFS up to max_hops from each seed
        frontier: set[str] = set(entity_ids) & set(self._graph.nodes)
        for _ in range(safe_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour in list(self._graph.successors(node)) + list(self._graph.predecessors(node)):
                    if neighbour not in visited_nodes:
                        next_frontier.add(neighbour)
            visited_nodes |= frontier
            frontier = next_frontier
            if len(visited_nodes) >= limit:
                break
        visited_nodes |= frontier

        entities: list[EntityRecord] = []
        for nid in visited_nodes:
            if nid not in self._entities:
                continue
            data = self._entities[nid]
            entities.append(EntityRecord(
                entity_id=nid,
                canonical_name=data.get("canonical_name", ""),
                original_mention=data.get("original_mention", data.get("canonical_name", "")),
                entity_type=data.get("entity_type", "Concept"),
                confidence=data.get("confidence"),
                document_id=data.get("document_id"),
                chunk_id=data.get("chunk_id"),
                page_number=data.get("page_number"),
                extraction_method=data.get("extraction_method", "llm"),
            ))

        relationships: list[RelationshipRecord] = []
        source_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()
        for u, v, edge_data in self._graph.edges(data=True):
            if u not in visited_nodes or v not in visited_nodes:
                continue
            rid = edge_data.get("relationship_id")
            if rid in visited_edges:
                continue
            visited_edges.add(rid)
            rdata = self._relationships.get(rid) or edge_data
            relationships.append(RelationshipRecord(
                relationship_id=rid or f"{u}__{v}",
                source_entity_id=u,
                source_entity_name=rdata.get("source_entity_name", ""),
                target_entity_id=v,
                target_entity_name=rdata.get("target_entity_name", ""),
                relationship_type=rdata.get("relationship_type", "RELATED_TO"),
                confidence=rdata.get("confidence"),
                source_document_id=rdata.get("source_document_id"),
                source_chunk_id=rdata.get("source_chunk_id"),
                page_number=rdata.get("page_number"),
                extraction_method=rdata.get("extraction_method", "llm"),
                source_text=rdata.get("source_text"),
            ))
            chunk_id = rdata.get("source_chunk_id")
            if chunk_id and chunk_id not in seen_chunk_ids:
                seen_chunk_ids.add(chunk_id)
                source_chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=str(rdata.get("source_document_id") or ""),
                    document_name=str(rdata.get("source_document_id") or ""),
                    chunk_index=0,
                    text=str(rdata.get("source_text") or ""),
                    score=float(rdata.get("confidence") or 0.0),
                    page_number=rdata.get("page_number"),
                    metadata={"relationship_type": rdata.get("relationship_type", "RELATED_TO")},
                ))

        return GraphEvidence(
            entities=entities,
            relationships=relationships,
            source_chunks=source_chunks,
        )

    # ------------------------------------------------------------------
    # Query / Count / Visualization helpers matching Neo4jStore
    # ------------------------------------------------------------------

    def count_entities(self) -> int:
        return len(self._entities)

    def count_relationships(self) -> int:
        return len(self._relationships)

    def count_documents(self) -> int:
        doc_ids = {e.get("document_id") for e in self._entities.values() if e.get("document_id")}
        return len(doc_ids)

    def list_entities(self, entity_type: str | None = None, search: str | None = None, limit: int = 150) -> list[dict]:
        results: list[dict] = []
        lower_search = search.lower().strip() if search else None
        for eid, data in self._entities.items():
            if entity_type and entity_type != "All":
                if data.get("entity_type") != entity_type:
                    continue
            if lower_search:
                canonical = (data.get("canonical_name") or "").lower()
                if lower_search not in canonical:
                    continue
            results.append(data)
            if len(results) >= limit:
                break
        return results

    def list_relationships(self, relationship_type: str | None = None, entity_id: str | None = None, limit: int = 250) -> list[dict]:
        results: list[dict] = []
        for rid, data in self._relationships.items():
            if relationship_type and relationship_type != "All":
                if data.get("relationship_type") != relationship_type:
                    continue
            if entity_id:
                if data.get("source_entity_id") != entity_id and data.get("target_entity_id") != entity_id:
                    continue
            results.append(data)
            if len(results) >= limit:
                break
        return results

    def clear_graph(self) -> None:
        self._graph.clear()
        self._entities.clear()
        self._relationships.clear()

    # ------------------------------------------------------------------
    # Restore from SQLite registry (called on startup)
    # ------------------------------------------------------------------

    def restore_from_registry(self, registry) -> None:
        """Re-hydrate the in-memory graph from the SQLite registry so data
        persists across Streamlit restarts."""
        for row in registry.list_entities():
            eid = row["entity_id"]
            d = dict(row)
            self._entities[eid] = d
            self._graph.add_node(eid, **d)

        for row in registry.list_relationships():
            rid = row["relationship_id"]
            d = dict(row)
            self._relationships[rid] = d
            src = row.get("source_entity_id", "")
            tgt = row.get("target_entity_id", "")
            if src and tgt:
                if src not in self._graph:
                    self._graph.add_node(src, canonical_name=row.get("source_entity_name", ""))
                if tgt not in self._graph:
                    self._graph.add_node(tgt, canonical_name=row.get("target_entity_name", ""))
                self._graph.add_edge(src, tgt, **d)
