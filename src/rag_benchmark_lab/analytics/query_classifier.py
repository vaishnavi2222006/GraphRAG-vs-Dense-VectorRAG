"""Query Classification Service for RAG Intelligence.

Classifies queries into distinct semantic and structural categories:
- Factual / Simple Lookup
- Semantic Search
- Multi-hop Reasoning
- Entity Relationship
- Comparative
- Aggregation
- Procedural / Step-by-Step
- Analytical / Complex Reasoning
- Contextual
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import re
import sqlite3
from typing import Any


QUERY_CATEGORIES = [
    "Factual / Simple Lookup",
    "Semantic Search",
    "Multi-hop Reasoning",
    "Entity Relationship",
    "Comparative",
    "Aggregation",
    "Procedural / Step-by-Step",
    "Analytical / Complex Reasoning",
    "Contextual",
]


CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Factual / Simple Lookup": "Direct, precise fact retrieval such as names, addresses, IDs, phone numbers, or specific numbers.",
    "Semantic Search": "Broad semantic search across descriptive text, conceptual explanations, or corpus overviews.",
    "Multi-hop Reasoning": "Requires linking facts across multiple entities, hops, or documents to reach an answer.",
    "Entity Relationship": "Explicitly investigates connections, links, associations, or interactions between entities.",
    "Comparative": "Compares or contrasts two or more entities, methods, protocols, or outcomes.",
    "Aggregation": "Counts, totals, enumerations, or summaries across multiple records or chunks.",
    "Procedural / Step-by-Step": "Protocols, workflows, sequences, or step-by-step instructions for tasks or events.",
    "Analytical / Complex Reasoning": "In-depth causal reasoning, 'why' questions, impact analysis, or evaluative reasoning.",
    "Contextual": "Follow-up questions or context-dependent queries relying on previous conversational history.",
}


CATEGORY_COLOR_MAP: dict[str, str] = {
    "Factual / Simple Lookup": "#3B82F6",         # Blue
    "Semantic Search": "#6366F1",                 # Indigo
    "Multi-hop Reasoning": "#8B5CF6",             # Purple
    "Entity Relationship": "#EC4899",             # Pink
    "Comparative": "#F59E0B",                     # Amber
    "Aggregation": "#10B981",                     # Emerald
    "Procedural / Step-by-Step": "#06B6D4",       # Cyan
    "Analytical / Complex Reasoning": "#EF4444",   # Red
    "Contextual": "#64748B",                      # Slate
}


@dataclass(slots=True)
class QueryClassification:
    query_text: str
    category: str
    confidence: float
    signals: list[str] = field(default_factory=list)
    query_hash: str = ""

    def __post_init__(self) -> None:
        if not self.query_hash:
            norm = self.query_text.strip().lower()
            self.query_hash = hashlib.sha256(norm.encode("utf-8")).hexdigest()[:16]


class QueryClassifier:
    """Deterministic, linguistic query classifier with SQLite caching."""

    def __init__(self, sqlite_conn_factory=None) -> None:
        self._conn_factory = sqlite_conn_factory
        self._memory_cache: dict[str, QueryClassification] = {}
        if self._conn_factory:
            self._init_db()

    def _init_db(self) -> None:
        try:
            with self._conn_factory() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS query_classifications (
                        query_hash TEXT PRIMARY KEY,
                        query_text TEXT NOT NULL,
                        category TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        signals_json TEXT NOT NULL DEFAULT '[]',
                        created_at TEXT NOT NULL
                    )
                """)
                conn.execute("CREATE INDEX IF NOT EXISTS idx_qc_cat ON query_classifications(category)")
        except Exception:
            pass

    def classify(self, query: str) -> QueryClassification:
        """Classify a single query, using cached results if available."""
        if not query or not query.strip():
            return QueryClassification(
                query_text="",
                category="Factual / Simple Lookup",
                confidence=0.5,
                signals=["empty_query"],
            )

        clean_q = query.strip()
        q_hash = hashlib.sha256(clean_q.lower().encode("utf-8")).hexdigest()[:16]

        if q_hash in self._memory_cache:
            return self._memory_cache[q_hash]

        if self._conn_factory:
            try:
                with self._conn_factory() as conn:
                    row = conn.execute(
                        "SELECT category, confidence, signals_json FROM query_classifications WHERE query_hash = ?",
                        (q_hash,),
                    ).fetchone()
                    if row:
                        classification = QueryClassification(
                            query_text=clean_q,
                            category=row[0],
                            confidence=float(row[1]),
                            signals=json.loads(row[2] or "[]"),
                            query_hash=q_hash,
                        )
                        self._memory_cache[q_hash] = classification
                        return classification
            except Exception:
                pass

        # Perform fresh rule-based classification
        classification = self._detect_category(clean_q, q_hash)
        self._memory_cache[q_hash] = classification

        # Persist to SQLite
        if self._conn_factory:
            try:
                with self._conn_factory() as conn:
                    now = datetime.now(timezone.utc).isoformat()
                    conn.execute(
                        """INSERT INTO query_classifications
                        (query_hash, query_text, category, confidence, signals_json, created_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ON CONFLICT(query_hash) DO UPDATE SET
                            category=excluded.category,
                            confidence=excluded.confidence,
                            signals_json=excluded.signals_json""",
                        (
                            q_hash,
                            clean_q,
                            classification.category,
                            classification.confidence,
                            json.dumps(classification.signals),
                            now,
                        ),
                    )
            except Exception:
                pass

        return classification

    def _detect_category(self, query: str, q_hash: str) -> QueryClassification:
        q_lower = query.lower()
        signals: list[str] = []

        # 1. Comparative Queries
        comp_keywords = [
            "compare", "comparison", "versus", " vs ", "difference between",
            "differ", "compared to", "differ from", "contrast", "advantages of",
            "tradeoff", "trade-off", "which is better",
        ]
        comp_matches = [kw for kw in comp_keywords if kw in q_lower]
        if comp_matches:
            signals.extend([f"comp_term:{m}" for m in comp_matches])
            return QueryClassification(query, "Comparative", 0.92, signals, q_hash)

        # 2. Analytical / Complex Reasoning (Why / Causal / Impact questions)
        analytical_lead_patterns = [
            r"^\s*why\b", r"\bexplain why\b", r"\bimpact of\b", r"\bcause of\b",
            r"\breason for\b", r"\broot cause\b", r"\bhow come\b",
        ]
        for pat in analytical_lead_patterns:
            if re.search(pat, q_lower):
                signals.append(f"analytical_pat:{pat}")
                return QueryClassification(query, "Analytical / Complex Reasoning", 0.90, signals, q_hash)

        # 3. Multi-hop Reasoning Queries
        multihop_keywords = [
            "trace", "journey", "across multiple documents", "from initial",
            "through follow-up", "multi-hop", "multihop", "chain of",
            "who manages", "which team maintains", "path from", "connecting path",
        ]
        multihop_matches = [kw for kw in multihop_keywords if kw in q_lower]
        if multihop_matches or ("who manages" in q_lower and "team" in q_lower):
            signals.extend([f"multihop_term:{m}" for m in multihop_matches])
            return QueryClassification(query, "Multi-hop Reasoning", 0.90, signals, q_hash)

        # 4. Contextual Queries (Earlier / follow-up reference)
        contextual_lead_patterns = [
            r"\bbased on (that|it|this|them|prior)\b", r"\bwhat about\b",
            r"\bas mentioned\b", r"\bin that case\b", r"\bearlier\b",
        ]
        for pat in contextual_lead_patterns:
            if re.search(pat, q_lower):
                signals.append(f"context_pat:{pat}")
                return QueryClassification(query, "Contextual", 0.88, signals, q_hash)

        # 5. Procedural / Step-by-Step Queries
        proc_keywords = [
            "step", "steps", "procedure", "process", "workflow", "protocol",
            "how to", "guideline", "guidelines", "instructions", "sequence",
            "stages", "triage protocol", "admission protocol", "discharge process",
        ]
        proc_matches = [kw for kw in proc_keywords if re.search(rf"\b{re.escape(kw)}\b", q_lower)]
        if proc_matches:
            signals.extend([f"proc_term:{m}" for m in proc_matches])
            return QueryClassification(query, "Procedural / Step-by-Step", 0.88, signals, q_hash)

        # 6. Entity Relationship Queries
        rel_keywords = [
            "relationship", "relationships", "relation", "relations",
            "connected", "connection", "connections", "associated with",
            "associated medical", "doctors and their", "doctor and specialty",
            "interact", "interaction", "interactions", "dependencies",
            "which doctors work in", "which department belongs",
        ]
        rel_matches = [kw for kw in rel_keywords if kw in q_lower]
        if rel_matches:
            signals.extend([f"rel_term:{m}" for m in rel_matches])
            return QueryClassification(query, "Entity Relationship", 0.86, signals, q_hash)

        # 7. Factual / Simple Lookup Queries (Check high-frequency lookup patterns)
        factual_patterns = [
            r"\bwhere is\b",
            r"\bwhat is the ([\w\-]+\s+)?(name|address|location|phone|capacity|bed capacity|diagnosis|contact extension|extension)\b",
            r"\bwhat (is|was) the ([\w\-]+\s+)?(name|address|location|diagnosis)\b",
            r"\b(hospital name|bed capacity|patient diagnosis)\b",
            r"\bwho is\b",
            r"\bwhen was\b",
            r"\bp-\d+\b",
            r"\bexact identifier\b",
            r"\bwhich database\b",
            r"\bcontact extension\b",
            r"\bwhich contact extension\b",
        ]
        for pat in factual_patterns:
            if re.search(pat, q_lower):
                signals.append(f"factual_pat:{pat}")
        if signals:
            return QueryClassification(query, "Factual / Simple Lookup", 0.88, signals, q_hash)

        # 8. Aggregation Queries
        agg_patterns = [
            r"\bhow many\b", r"\btotal number\b", r"\btotal count\b",
            r"\bcount of\b", r"\blist all\b", r"\ball of the\b",
            r"\bsum of\b", r"\baverage\b", r"\bpercentage of\b",
        ]
        for pat in agg_patterns:
            if re.search(pat, q_lower):
                signals.append(f"agg_pat:{pat}")
        if signals:
            return QueryClassification(query, "Aggregation", 0.85, signals, q_hash)

        # 9. Other Analytical / Complex Reasoning Queries
        other_analytical = [
            r"\bhow does\b", r"\bhow can\b", r"\bevaluate\b", r"\banalyze\b", r"\banalysis\b",
        ]
        for pat in other_analytical:
            if re.search(pat, q_lower):
                signals.append(f"analytical_pat:{pat}")
                return QueryClassification(query, "Analytical / Complex Reasoning", 0.82, signals, q_hash)

        # Short direct queries fallback to Factual
        if len(query.split()) <= 4:
            signals.append("short_direct_query")
            return QueryClassification(query, "Factual / Simple Lookup", 0.75, signals, q_hash)

        # 9. Default: Semantic Search Queries
        signals.append("broad_thematic_semantic_query")
        return QueryClassification(query, "Semantic Search", 0.72, signals, q_hash)
