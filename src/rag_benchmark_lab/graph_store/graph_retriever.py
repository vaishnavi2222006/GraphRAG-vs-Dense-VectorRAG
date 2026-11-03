from __future__ import annotations

import re
from dataclasses import dataclass

from ..schemas.documents import GraphEvidence, RetrievedChunk
from ..utils.errors import ServiceUnavailableError


def _query_terms(question: str) -> list[str]:
    # Stopwords to exclude from single term search to prevent false substring matches
    stopwords = {
        "what", "which", "when", "where", "why", "how", "does", "do", "did", "is", "are", "was", "were",
        "the", "and", "with", "that", "from", "into", "about", "connected", "this", "these", "those",
        "explain", "describe", "discuss", "compare", "between", "among", "have", "has", "had", "for", "not",
        "of", "in", "on", "at", "to", "by", "an", "a", "or", "as", "if", "be", "so", "up", "out", "can",
        "may", "its", "it", "their", "his", "her", "them", "they", "our", "my", "your", "we", "you", "me",
        "him", "us", "who", "whom", "whose", "tell", "give", "show", "find", "use", "used", "using", "also",
        "all", "any", "some", "more", "most", "than", "then", "very", "just", "now", "such", "only", "same",
        "other", "there", "here", "each", "both", "few", "over", "under", "been", "will", "would", "could",
        "should", "document", "documents", "corpus", "uploaded", "available"
    }

    terms: list[str] = []
    seen: set[str] = set()

    # 1. Capitalized noun phrases / multi-word entities (e.g., "Dr. Elena Vasquez", "SunCore Systems")
    for match in re.finditer(r"\b([A-Z][a-zA-Z0-9_\-]+(?:\s+[A-Z][a-zA-Z0-9_\-]+)+)\b", question):
        phrase = match.group(1).strip()
        low = phrase.lower()
        if low not in seen and len(phrase) > 3:
            seen.add(low)
            terms.append(phrase)

    # 2. 2-gram and 3-gram contiguous content words
    words = [w.strip(".,?!\"'()[]{}:;-_/\\") for w in question.split()]
    clean_words = [w for w in words if w]
    for i in range(len(clean_words) - 1):
        w1, w2 = clean_words[i].lower(), clean_words[i + 1].lower()
        if w1 not in stopwords and w2 not in stopwords:
            bigram = f"{clean_words[i]} {clean_words[i+1]}"
            if bigram.lower() not in seen:
                seen.add(bigram.lower())
                terms.append(bigram)

    # 3. Individual significant terms (words, numbers with units like 1.6, 2023, LCOE)
    for term in re.findall(r"[A-Za-z0-9_\-]{2,}", question):
        low = term.lower()
        if low in stopwords or low in seen:
            continue
        # Filter out 2-character lowercase noise
        if len(term) < 3 and not (term.isupper() or term.isdigit()):
            continue
        seen.add(low)
        terms.append(term)

    return terms[:15]


@dataclass(slots=True)
class GraphRetrievalResult:
    evidence: GraphEvidence
    source_chunks: list[RetrievedChunk]
    retrieval_latency_ms: float
    warnings: list[str]


def _stem_word(w: str) -> str:
    word = w.lower().strip(".,?!\"'()[]{}:;-_/\\")
    if len(word) <= 3:
        return word
    for suffix in ("ing", "ies", "es", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[:-3] + "y"
            return word[:-len(suffix)]
    return word


def _tokenize_text(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9]+(?:'[a-z]+)?\b", text.lower())


def _score_graph_chunk(
    chunk: RetrievedChunk,
    question: str,
    terms: list[str],
    entities: list,
    relationships: list,
) -> float:
    chunk_text_lower = chunk.text.lower()
    chunk_words = _tokenize_text(chunk.text)
    chunk_stems = set(_stem_word(w) for w in chunk_words)
    score = 0.0

    # Term overlap with stemming and subword support
    if terms:
        matched_terms = 0
        for t in terms:
            t_words = _tokenize_text(t)
            if len(t_words) > 1:
                t_stems = [_stem_word(w) for w in t_words]
                found = any(
                    chunk_words[j : j + len(t_words)] == t_words
                    for j in range(len(chunk_words) - len(t_words) + 1)
                )
                if not found:
                    found = any(
                        [_stem_word(chunk_words[j + k]) for k in range(len(t_words))] == t_stems
                        for j in range(len(chunk_words) - len(t_words) + 1)
                    )
                if found:
                    matched_terms += 1
            else:
                st = _stem_word(t)
                if st in chunk_stems:
                    matched_terms += 1
        score += (matched_terms / len(terms)) * 0.50

    # Entity presence
    if entities:
        ent_hits = sum(1 for e in entities if e.canonical_name.lower() in chunk_text_lower)
        score += (ent_hits / len(entities)) * 0.30

    # Full phrase / number bonus
    numbers_in_q = re.findall(r"\b\d+(?:\.\d+)?\b", question)
    if numbers_in_q:
        num_hits = sum(1 for num in numbers_in_q if num in chunk_words)
        score += (num_hits / len(numbers_in_q)) * 0.15

    # Relationship text match bonus
    for rel in relationships:
        if rel.source_chunk_id == chunk.chunk_id or (rel.source_text and rel.source_text.lower() in chunk_text_lower):
            score += 0.15
            break

    # Blend with traversal/entity confidence
    score += min(float(chunk.score or 0.5), 1.0) * 0.10

    # Discount chunks that are primarily question lists rather than factual evidence
    lines = [line.strip() for line in chunk.text.split("\n") if line.strip()]
    q_lines = sum(
        1 for line in lines
        if line.endswith("?")
        or line.lower().startswith((
            "what ", "which ", "who ", "how ", "where ", "when ", "why ",
            "can a ", "suggested test", "test question"
        ))
    )
    if lines and (q_lines / len(lines)) > 0.25:
        score *= max(0.15, 1.0 - (q_lines / len(lines)))

    admin_patterns = [
        r"\bupload instructions\b",
        r"\bcreate a (?:new )?corpus\b",
        r"\bupload all \d+ (?:docx|pdf|txt|files)\b",
        r"\bsuggested test questions\b",
        r"\bexpected facts for manual validation\b",
        r"\bwait until processing\b",
        r"\bfictional (?:15-document|rag demo) corpus\b",
    ]
    admin_matches = sum(1 for p in admin_patterns if re.search(p, chunk.text, re.IGNORECASE))
    if admin_matches >= 2:
        score *= 0.20
    elif admin_matches == 1:
        score *= 0.50

    return round(min(1.0, max(0.02, score)), 4)


class GraphRetriever:
    def __init__(self, neo4j_store, max_hops: int = 2, max_results: int = 20, registry=None) -> None:
        self.neo4j_store = neo4j_store
        self.max_hops = max_hops
        self.max_results = max_results
        self.registry = registry

    def retrieve(self, question: str, document_ids: list[str] | None = None) -> GraphRetrievalResult:
        from time import perf_counter

        start = perf_counter()
        warnings: list[str] = []

        # Strict corpus isolation for empty corpus scope
        if document_ids is not None and len(document_ids) == 0:
            evidence = GraphEvidence(entities=[], relationships=[], source_chunks=[])
            return GraphRetrievalResult(
                evidence=evidence,
                source_chunks=[],
                retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                warnings=["empty_corpus_scope"],
            )

        doc_id_set = set(document_ids) if document_ids is not None else None

        try:
            terms = _query_terms(question)
            try:
                entities = self.neo4j_store.search_entities(terms, limit=self.max_results, document_ids=document_ids)
            except TypeError:
                # Store does not accept document_ids parameter
                entities = self.neo4j_store.search_entities(terms, limit=self.max_results)

            # Strict corpus filtering for seed entities
            if doc_id_set is not None:
                entities = [
                    entity for entity in entities
                    if entity.document_id in doc_id_set or (
                        hasattr(entity, "metadata") and any(d in doc_id_set for d in entity.metadata.get("document_ids", []))
                    )
                ]

            if not entities:
                warnings.append("no_graph_entities_found")
                evidence = GraphEvidence(entities=[], relationships=[], source_chunks=[])
                return GraphRetrievalResult(
                    evidence=evidence,
                    source_chunks=[],
                    retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                    warnings=warnings,
                )

            try:
                evidence = self.neo4j_store.traverse(
                    [entity.entity_id for entity in entities],
                    max_hops=self.max_hops,
                    limit=self.max_results,
                    document_ids=document_ids,
                )
            except TypeError:
                evidence = self.neo4j_store.traverse(
                    [entity.entity_id for entity in entities],
                    max_hops=self.max_hops,
                    limit=self.max_results,
                )

            source_chunks = evidence.source_chunks

            # Strict corpus filtering for traversed chunks and relationships
            if doc_id_set is not None:
                source_chunks = [
                    chunk for chunk in source_chunks
                    if chunk.document_id in doc_id_set
                ]
                traversed_relationships = [
                    rel for rel in evidence.relationships
                    if not rel.source_document_id or rel.source_document_id in doc_id_set
                ]
                traversed_entities = [
                    ent for ent in evidence.entities
                    if not ent.document_id or ent.document_id in doc_id_set
                ]
            else:
                traversed_relationships = evidence.relationships
                traversed_entities = evidence.entities

            # Hydrate full chunk text from registry or graph store where available
            for chunk in source_chunks:
                try:
                    record = None
                    if self.registry is not None:
                        record = self.registry.get_chunk(chunk.chunk_id)
                    if not record and hasattr(self.neo4j_store, "get_chunk"):
                        record = self.neo4j_store.get_chunk(chunk.chunk_id)
                    if record:
                        chunk_text = record.get("text") if isinstance(record, dict) else getattr(record, "text", None)
                        if chunk_text:
                            chunk.text = chunk_text
                            c_idx = record.get("chunk_index") if isinstance(record, dict) else getattr(record, "chunk_index", None)
                            if c_idx is not None:
                                chunk.chunk_index = c_idx
                            c_doc = record.get("document_name") if isinstance(record, dict) else getattr(record, "document_name", None)
                            if c_doc:
                                chunk.document_name = c_doc
                            c_pg = record.get("page_number") if isinstance(record, dict) else getattr(record, "page_number", None)
                            if c_pg is not None:
                                chunk.page_number = c_pg
                except Exception:
                    pass

            # Score and rank source_chunks by query and relational relevance
            for chunk in source_chunks:
                chunk.score = _score_graph_chunk(chunk, question, terms, traversed_entities, traversed_relationships)

            source_chunks.sort(key=lambda c: c.score, reverse=True)
            # Prioritize chunks with actual evidence match
            scored_chunks = [c for c in source_chunks if c.score >= 0.15]
            if scored_chunks:
                source_chunks = scored_chunks[: self.max_results]
            else:
                source_chunks = source_chunks[: self.max_results]

            evidence = GraphEvidence(
                entities=traversed_entities,
                relationships=traversed_relationships,
                source_chunks=source_chunks,
            )

            if not source_chunks:
                warnings.append("no_graph_source_chunks_found")

            return GraphRetrievalResult(
                evidence=evidence,
                source_chunks=source_chunks,
                retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                warnings=warnings,
            )
        except Exception as exc:
            warnings.append(f"graph_retrieval_failed:{exc.__class__.__name__}")
            return GraphRetrievalResult(
                evidence=GraphEvidence(entities=[], relationships=[], source_chunks=[]),
                source_chunks=[],
                retrieval_latency_ms=(perf_counter() - start) * 1000.0,
                warnings=warnings,
            )

