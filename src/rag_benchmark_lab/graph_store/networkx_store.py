"""NetworkX-based in-memory graph store — a drop-in replacement for Neo4jStore.

This store satisfies the same interface as Neo4jStore but requires no Docker or
external database. All data is kept in a NetworkX DiGraph and synced to the
SQLite registry (which is already used for chunks/entities/relationships), so
nothing is lost across Streamlit restarts.
"""
from __future__ import annotations

from dataclasses import asdict
import re
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
        # chunk_id -> ChunkRecord (dict form)
        self._chunks: dict[str, dict] = {}

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
            eid = entity.entity_id
            if eid in self._entities:
                existing = self._entities[eid]
                doc_ids = set(existing.get("document_ids") or [])
                if existing.get("document_id"):
                    doc_ids.add(existing["document_id"])
                if entity.document_id:
                    doc_ids.add(entity.document_id)
                chunk_ids = set(existing.get("chunk_ids") or [])
                if existing.get("chunk_id"):
                    chunk_ids.add(existing["chunk_id"])
                if entity.chunk_id:
                    chunk_ids.add(entity.chunk_id)
                existing.update(d)
                existing["document_ids"] = list(doc_ids)
                existing["chunk_ids"] = list(chunk_ids)
                if self._graph.has_node(eid):
                    self._graph.nodes[eid].update(existing)
            else:
                d["document_ids"] = [entity.document_id] if entity.document_id else []
                d["chunk_ids"] = [entity.chunk_id] if entity.chunk_id else []
                self._entities[eid] = d
                self._graph.add_node(eid, **d)

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

    def upsert_chunks(self, chunks: list[object]) -> None:
        for chunk in chunks:
            cid = getattr(chunk, "chunk_id", None) or (chunk.get("chunk_id") if isinstance(chunk, dict) else None)
            if cid:
                d = asdict(chunk) if hasattr(chunk, "__dataclass_fields__") else dict(chunk)
                self._chunks[cid] = d

    def get_chunk(self, chunk_id: str) -> dict | None:
        return self._chunks.get(chunk_id)

    def upsert_document(self, document_id: str, document_name: str) -> None:
        # No separate Document node needed; documents tracked in SQLite registry
        pass

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    def delete_document(self, document_id: str) -> None:
        entity_ids_to_remove = []
        for eid, edata in self._entities.items():
            doc_ids = list(edata.get("document_ids") or [])
            if document_id in doc_ids:
                doc_ids.remove(document_id)
                edata["document_ids"] = doc_ids
            if edata.get("document_id") == document_id:
                edata["document_id"] = doc_ids[0] if doc_ids else None
            if not doc_ids and not edata.get("document_id"):
                entity_ids_to_remove.append(eid)
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
        chunk_ids_to_remove = [
            cid for cid, cdata in self._chunks.items()
            if cdata.get("document_id") == document_id
        ]
        for cid in chunk_ids_to_remove:
            self._chunks.pop(cid, None)

    # ------------------------------------------------------------------
    # Search / Traversal
    # ------------------------------------------------------------------

    def search_entities(self, terms: list[str], limit: int = 10, document_ids: list[str] | None = None) -> list[EntityRecord]:
        if document_ids is not None and len(document_ids) == 0:
            return []
        doc_id_set = set(document_ids) if document_ids is not None else None

        lower_terms = [t.lower() for t in terms if t]
        matched: list[tuple[float, EntityRecord]] = []
        for entity_id, data in self._entities.items():
            if doc_id_set is not None:
                ent_docs = set(data.get("document_ids") or [])
                if data.get("document_id"):
                    ent_docs.add(data["document_id"])
                if not (ent_docs & doc_id_set):
                    continue

            name = (data.get("canonical_name") or "").lower()
            mention = (data.get("original_mention") or "").lower()
            score = 0.0
            hits = 0
            for term in lower_terms:
                if term == name or term == mention:
                    score += 5.0
                    hits += 1
                elif re.search(rf"\b{re.escape(term)}\b", name) or re.search(rf"\b{re.escape(term)}\b", mention):
                    score += 3.0
                    hits += 1
                elif len(term) >= 4 and (term in name or term in mention):
                    score += 1.5
                    hits += 1
                elif len(name) >= 4 and (name in term or mention in term):
                    score += 1.5
                    hits += 1
            if hits > 0:
                score += (data.get("confidence") or 0.5)
                # Assign active document_id from the filtered set if available
                assigned_doc_id = data.get("document_id")
                if doc_id_set is not None and assigned_doc_id not in doc_id_set:
                    intersect = set(data.get("document_ids") or []) & doc_id_set
                    if intersect:
                        assigned_doc_id = next(iter(intersect))

                record = EntityRecord(
                    entity_id=entity_id,
                    canonical_name=data.get("canonical_name", ""),
                    original_mention=data.get("original_mention", data.get("canonical_name", "")),
                    entity_type=data.get("entity_type", "Concept"),
                    confidence=data.get("confidence"),
                    document_id=assigned_doc_id,
                    chunk_id=data.get("chunk_id"),
                    page_number=data.get("page_number"),
                    extraction_method=data.get("extraction_method", "llm"),
                )
                matched.append((score, record))
        matched.sort(key=lambda x: x[0], reverse=True)
        return [record for _, record in matched[:limit]]

    def traverse(self, entity_ids: list[str], max_hops: int = 2, limit: int = 20, document_ids: list[str] | None = None) -> GraphEvidence:
        if document_ids is not None and len(document_ids) == 0:
            return GraphEvidence(entities=[], relationships=[], source_chunks=[])
        doc_id_set = set(document_ids) if document_ids is not None else None

        safe_hops = max(1, min(int(max_hops), 5))
        visited_nodes: set[str] = set()
        visited_edges: set[str] = set()

        # BFS/DFS up to max_hops from each seed, restricted to valid nodes
        valid_seed_nodes = set(entity_ids) & set(self._graph.nodes)
        if doc_id_set is not None:
            filtered_seeds = set()
            for nid in valid_seed_nodes:
                ndata = self._entities.get(nid, {})
                ndocs = set(ndata.get("document_ids") or [])
                if ndata.get("document_id"):
                    ndocs.add(ndata["document_id"])
                if ndocs & doc_id_set:
                    filtered_seeds.add(nid)
            valid_seed_nodes = filtered_seeds

        frontier: set[str] = valid_seed_nodes
        for _ in range(safe_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for neighbour in list(self._graph.successors(node)) + list(self._graph.predecessors(node)):
                    if neighbour not in visited_nodes:
                        if doc_id_set is not None:
                            ndata = self._entities.get(neighbour, {})
                            ndocs = set(ndata.get("document_ids") or [])
                            if ndata.get("document_id"):
                                ndocs.add(ndata["document_id"])
                            # Check if connected edge belongs to doc_id_set
                            edge_in_scope = False
                            for u, v in [(node, neighbour), (neighbour, node)]:
                                if self._graph.has_edge(u, v):
                                    ed = self._graph.get_edge_data(u, v) or {}
                                    if ed.get("source_document_id") in doc_id_set:
                                        edge_in_scope = True
                                        break
                            if not edge_in_scope and not (ndocs & doc_id_set):
                                continue
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
            if doc_id_set is not None:
                ndocs = set(data.get("document_ids") or [])
                if data.get("document_id"):
                    ndocs.add(data["document_id"])
                if not (ndocs & doc_id_set):
                    continue
            assigned_doc_id = data.get("document_id")
            if doc_id_set is not None and assigned_doc_id not in doc_id_set:
                intersect = set(data.get("document_ids") or []) & doc_id_set
                if intersect:
                    assigned_doc_id = next(iter(intersect))

            entities.append(EntityRecord(
                entity_id=nid,
                canonical_name=data.get("canonical_name", ""),
                original_mention=data.get("original_mention", data.get("canonical_name", "")),
                entity_type=data.get("entity_type", "Concept"),
                confidence=data.get("confidence"),
                document_id=assigned_doc_id,
                chunk_id=data.get("chunk_id"),
                page_number=data.get("page_number"),
                extraction_method=data.get("extraction_method", "llm"),
            ))

        relationships: list[RelationshipRecord] = []
        source_chunks: list[RetrievedChunk] = []
        seen_chunk_ids: set[str] = set()

        # 1. Collect chunks from traversed relationships
        for u, v, edge_data in self._graph.edges(data=True):
            if u not in visited_nodes or v not in visited_nodes:
                continue
            rid = edge_data.get("relationship_id")
            if rid in visited_edges:
                continue
            visited_edges.add(rid)
            rdata = self._relationships.get(rid) or edge_data
            if doc_id_set is not None and rdata.get("source_document_id") not in doc_id_set:
                continue

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
                src_doc_id = str(rdata.get("source_document_id") or "")
                src_entity_data = self._entities.get(u) or self._entities.get(v) or {}
                src_doc_name = src_entity_data.get("document_name") or src_doc_id
                chunk_text = str(rdata.get("source_text") or "")
                chunk_idx = 0
                pg_num = rdata.get("page_number")
                if chunk_id in self._chunks:
                    c_info = self._chunks[chunk_id]
                    chunk_text = getattr(c_info, "text", None) or (c_info.get("text") if isinstance(c_info, dict) else "") or chunk_text
                    src_doc_name = getattr(c_info, "document_name", None) or (c_info.get("document_name") if isinstance(c_info, dict) else "") or src_doc_name
                    chunk_idx = getattr(c_info, "chunk_index", 0) if hasattr(c_info, "chunk_index") else (c_info.get("chunk_index", 0) if isinstance(c_info, dict) else 0)
                    pg_num = getattr(c_info, "page_number", pg_num) if hasattr(c_info, "page_number") else (c_info.get("page_number", pg_num) if isinstance(c_info, dict) else pg_num)

                source_chunks.append(RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=src_doc_id,
                    document_name=src_doc_name,
                    chunk_index=chunk_idx,
                    text=chunk_text,
                    score=float(rdata.get("confidence") or 0.8),
                    page_number=pg_num,
                    metadata={"relationship_type": rdata.get("relationship_type", "RELATED_TO")},
                ))

        # 2. Collect supporting chunks from visited entities
        for nid in visited_nodes:
            if nid not in self._entities:
                continue
            edata = self._entities[nid]
            all_cids = list(edata.get("chunk_ids") or [])
            if edata.get("chunk_id") and edata.get("chunk_id") not in all_cids:
                all_cids.append(edata["chunk_id"])

            src_doc_id = str(edata.get("document_id") or "")
            if doc_id_set is not None and src_doc_id not in doc_id_set:
                intersect = set(edata.get("document_ids") or []) & doc_id_set
                if intersect:
                    src_doc_id = next(iter(intersect))
                else:
                    continue

            src_doc_name = edata.get("document_name") or src_doc_id
            for chunk_id in all_cids:
                if chunk_id and chunk_id not in seen_chunk_ids:
                    seen_chunk_ids.add(chunk_id)
                    chunk_text = str(edata.get("canonical_name") or "")
                    chunk_idx = 0
                    pg_num = edata.get("page_number")
                    if chunk_id in self._chunks:
                        c_info = self._chunks[chunk_id]
                        chunk_text = getattr(c_info, "text", None) or (c_info.get("text") if isinstance(c_info, dict) else "") or chunk_text
                        src_doc_name = getattr(c_info, "document_name", None) or (c_info.get("document_name") if isinstance(c_info, dict) else "") or src_doc_name
                        chunk_idx = getattr(c_info, "chunk_index", 0) if hasattr(c_info, "chunk_index") else (c_info.get("chunk_index", 0) if isinstance(c_info, dict) else 0)
                        pg_num = getattr(c_info, "page_number", pg_num) if hasattr(c_info, "page_number") else (c_info.get("page_number", pg_num) if isinstance(c_info, dict) else pg_num)

                    source_chunks.append(RetrievedChunk(
                        chunk_id=chunk_id,
                        document_id=src_doc_id,
                        document_name=src_doc_name,
                        chunk_index=chunk_idx,
                        text=chunk_text,
                        score=float(edata.get("confidence") or 0.7),
                        page_number=pg_num,
                        metadata={"entity_name": edata.get("canonical_name", ""), "entity_type": edata.get("entity_type", "Concept")},
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
        self._chunks.clear()

    # ------------------------------------------------------------------
    # Restore from SQLite registry (called on startup)
    # ------------------------------------------------------------------

    def restore_from_registry(self, registry) -> None:
        """Re-hydrate the in-memory graph from the SQLite registry so data
        persists across Streamlit restarts."""
        if hasattr(registry, "list_chunks"):
            try:
                for row in registry.list_chunks():
                    self._chunks[row["chunk_id"]] = dict(row)
            except Exception:
                pass

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
