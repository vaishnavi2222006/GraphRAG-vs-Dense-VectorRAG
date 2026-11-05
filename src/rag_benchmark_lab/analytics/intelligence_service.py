"""RAG Intelligence and Query Analytics Service.

Aggregates historical queries from Chat History, Comparison Runs, and Evaluation Runs,
classifies queries into semantic/structural categories, and calculates empirical
performance metrics across Vector RAG, GraphRAG, and Hybrid RAG architectures.

Strictly separates Actual Evaluation scores from Estimated/Provisional scores,
normalizes across 4 evaluation pillars (Answer Quality, Retrieval Quality, Performance, Efficiency),
and dynamically re-weights missing metrics without penalizing evaluated approaches.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import statistics
from typing import Any

from rag_benchmark_lab.analytics.query_classifier import (
    CATEGORY_COLOR_MAP,
    CATEGORY_DESCRIPTIONS,
    QUERY_CATEGORIES,
    QueryClassification,
    QueryClassifier,
)
from rag_benchmark_lab.services.cost_service import CostService
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.user_registry import UserRegistry


@dataclass(slots=True)
class ScoringWeights:
    """Configurable weights for the overall weighted performance score."""
    answer_quality: float = 0.35
    retrieval_quality: float = 0.35
    latency: float = 0.15
    cost: float = 0.15

    def normalized(self) -> ScoringWeights:
        tot = self.answer_quality + self.retrieval_quality + self.latency + self.cost
        if tot <= 0:
            return ScoringWeights(0.35, 0.35, 0.15, 0.15)
        return ScoringWeights(
            round(self.answer_quality / tot, 4),
            round(self.retrieval_quality / tot, 4),
            round(self.latency / tot, 4),
            round(self.cost / tot, 4),
        )


@dataclass(slots=True)
class ApproachRecord:
    strategy: str  # "VECTOR", "GRAPH", "HYBRID"
    status: str = "success"
    # Separate actual vs provisional scores
    actual_quality_score: float | None = None
    provisional_quality_score: float | None = None
    quality_score: float | None = None
    is_provisional: bool = False
    score_source: str = "none"  # "Actual Evaluation", "Estimated / Provisional", "Telemetry Only"
    # 4 Evaluation Pillars
    answer_quality_score: float | None = None
    retrieval_quality_score: float | None = None
    latency_score: float | None = None
    cost_score: float | None = None
    overall_weighted_score: float | None = None
    # Performance & Latency Telemetry
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    # Efficiency & Token Telemetry
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_inr: float | None = None
    # Answer and Evidence
    answer: str = ""
    citations: list[dict[str, Any]] = field(default_factory=list)
    graph_entities: list[dict[str, Any]] = field(default_factory=list)
    graph_relationships: list[dict[str, Any]] = field(default_factory=list)
    # Metrics separation
    metrics: dict[str, float] = field(default_factory=dict)  # Genuine measured metrics
    actual_metrics: dict[str, float] = field(default_factory=dict)
    provisional_metrics: dict[str, float] = field(default_factory=dict)
    run_id: str = ""
    source: str = "evaluation"


@dataclass(slots=True)
class QueryIntelligenceRecord:
    query_id: str
    query_text: str
    category: str
    category_confidence: float
    signals: list[str]
    approaches: dict[str, ApproachRecord] = field(default_factory=dict)
    best_approach: str = "Insufficient data"
    recommendation_reason: str = ""
    margin_summary: str = ""
    score_source: str = "Actual Evaluation"
    is_provisional: bool = False
    source: str = "evaluation"
    created_at: str = ""


@dataclass(slots=True)
class CategoryIntelligence:
    category_name: str
    query_count: int
    actual_query_count: int
    provisional_query_count: int
    color: str
    description: str
    best_approach: str  # Overall recommended architecture
    best_answer_quality_arch: str
    best_retrieval_quality_arch: str
    fastest_arch: str
    most_cost_efficient_arch: str
    confidence_level: str  # "High", "Medium", "Low", "Insufficient data"
    data_sufficiency_reason: str
    empirical_reason: str
    margin_summary: str
    tradeoffs: str
    strengths: list[str]
    approach_stats: dict[str, dict[str, Any]]  # Per strategy summary across 4 pillars
    win_counts: dict[str, int]
    query_records: list[QueryIntelligenceRecord] = field(default_factory=list)

    @property
    def best_overall_architecture(self) -> str:
        return self.best_approach


@dataclass(slots=True)
class IntelligenceSummary:
    total_queries: int
    actual_evaluated_queries: int
    provisional_queries: int
    classified_categories_count: int
    category_distribution: dict[str, int]
    best_overall_architecture: str | None
    architecture_wins: dict[str, int]
    mean_benchmark_latency_ms: float | None
    scoring_weights: ScoringWeights
    categories: dict[str, CategoryIntelligence]
    all_queries: list[QueryIntelligenceRecord]


class IntelligenceService:
    """Service to extract, classify, evaluate, and recommend RAG architectures."""

    def __init__(
        self,
        doc_registry: DocumentRegistry,
        user_registry: UserRegistry,
        classifier: QueryClassifier | None = None,
        cost_service: CostService | None = None,
    ) -> None:
        self.doc_registry = doc_registry
        self.user_registry = user_registry
        self.classifier = classifier or QueryClassifier()
        self.cost_service = cost_service or CostService()

    def get_intelligence_summary(
        self,
        kb_id: str | None = None,
        weights: ScoringWeights | None = None,
        score_mode: str = "all",  # "all", "actual_only", "provisional_only"
    ) -> IntelligenceSummary:
        """Aggregate all queries, classify them, compute weighted scores, and produce RAG analytics."""
        active_weights = (weights or ScoringWeights()).normalized()

        # 1. Collect queries from evaluations, comparisons, and chat
        raw_queries_map: dict[str, dict[str, Any]] = {}

        self._load_from_evaluations(raw_queries_map)
        self._load_from_comparisons(raw_queries_map, kb_id=kb_id)
        self._load_from_chat_messages(raw_queries_map)

        # 2. Build QueryIntelligenceRecord for each query with pillar scores and weighted overall score
        query_records: list[QueryIntelligenceRecord] = []
        actual_query_count = 0
        provisional_query_count = 0

        for norm_q, q_data in raw_queries_map.items():
            query_text = q_data["query_text"]
            classification = self.classifier.classify(query_text)
            approaches: dict[str, ApproachRecord] = q_data["approaches"]

            # Filter approaches based on score_mode if requested
            filtered_approaches: dict[str, ApproachRecord] = {}
            for strat, app in approaches.items():
                if score_mode == "actual_only" and app.is_provisional and app.actual_quality_score is None:
                    continue
                if score_mode == "provisional_only" and not app.is_provisional:
                    continue
                filtered_approaches[strat] = app

            if not filtered_approaches:
                continue

            # Compute pillar scores and weighted score for each approach
            self._compute_approach_pillars(filtered_approaches, active_weights)

            # Determine best approach for this query
            best_strat, reason, margin = self._determine_query_best_approach(filtered_approaches)

            has_actual = any(not a.is_provisional for a in filtered_approaches.values())
            is_query_prov = all(a.is_provisional for a in filtered_approaches.values())
            if has_actual:
                actual_query_count += 1
                q_score_src = "Actual Evaluation"
            else:
                provisional_query_count += 1
                q_score_src = "Estimated / Provisional"

            record = QueryIntelligenceRecord(
                query_id=q_data.get("query_id", classification.query_hash),
                query_text=query_text,
                category=classification.category,
                category_confidence=classification.confidence,
                signals=classification.signals,
                approaches=filtered_approaches,
                best_approach=best_strat,
                recommendation_reason=reason,
                margin_summary=margin,
                score_source=q_score_src,
                is_provisional=is_query_prov,
                source=q_data.get("source", "benchmark"),
                created_at=q_data.get("created_at", datetime.now(timezone.utc).isoformat()),
            )
            query_records.append(record)

        # 3. Aggregate per Category
        category_distribution: dict[str, int] = {cat: 0 for cat in QUERY_CATEGORIES}
        category_records: dict[str, list[QueryIntelligenceRecord]] = {cat: [] for cat in QUERY_CATEGORIES}

        for rec in query_records:
            if rec.category in category_records:
                category_records[rec.category].append(rec)
                category_distribution[rec.category] += 1
            else:
                category_records.setdefault(rec.category, []).append(rec)
                category_distribution[rec.category] = category_distribution.get(rec.category, 0) + 1

        # 4. Compute Category Intelligence & Recommendations
        categories_dict: dict[str, CategoryIntelligence] = {}
        architecture_overall_wins: dict[str, int] = {"VECTOR": 0, "GRAPH": 0, "HYBRID": 0}

        for cat in QUERY_CATEGORIES:
            recs = category_records.get(cat, [])
            cat_intel = self._build_category_intelligence(cat, recs, active_weights)
            categories_dict[cat] = cat_intel

            winner = cat_intel.best_approach
            if winner in architecture_overall_wins:
                architecture_overall_wins[winner] += 1

        # 5. Compute Overall Best Architecture
        best_overall = None
        max_wins = -1
        for strat, wins in architecture_overall_wins.items():
            if wins > max_wins and wins > 0:
                max_wins = wins
                best_overall = strat

        # Calculate mean latency
        all_lats = [
            app.total_latency_ms
            for rec in query_records
            for app in rec.approaches.values()
            if app.total_latency_ms is not None
        ]
        mean_lat = round(statistics.mean(all_lats), 1) if all_lats else None

        active_cats_count = sum(1 for cnt in category_distribution.values() if cnt > 0)

        return IntelligenceSummary(
            total_queries=len(query_records),
            actual_evaluated_queries=actual_query_count,
            provisional_queries=provisional_query_count,
            classified_categories_count=active_cats_count,
            category_distribution=category_distribution,
            best_overall_architecture=best_overall,
            architecture_wins=architecture_overall_wins,
            mean_benchmark_latency_ms=mean_lat,
            scoring_weights=active_weights,
            categories=categories_dict,
            all_queries=query_records,
        )

    def _compute_provisional_score(
        self,
        status: str,
        citations: list,
        graph_entities: list,
        graph_relationships: list,
        answer: str,
        latency_ms: float | None,
    ) -> tuple[float | None, dict[str, float]]:
        """Calculate an evidence-based provisional quality score for comparison runs."""
        if status != "success" or not answer or len(answer.strip()) < 15:
            return None, {}
        c_cnt = len(citations or [])
        cit_score = min(1.0, c_cnt / 4.0) if c_cnt > 0 else 0.4
        g_cnt = len(graph_entities or []) + len(graph_relationships or [])
        graph_bonus = min(0.15, (g_cnt / 20.0) * 0.15) if g_cnt > 0 else 0.0
        ans_len = len(answer.strip())
        rel_score = round(min(1.0, 0.60 + min(0.35, (ans_len / 400.0) * 0.35)), 4)
        faith_score = round(min(1.0, 0.50 + min(0.35, (c_cnt / 5.0) * 0.35) + graph_bonus), 4)
        lat_penalty = min(0.05, (latency_ms or 0.0) / 60000.0)

        score = round(max(0.1, min(1.0, 0.40 * faith_score + 0.35 * rel_score + 0.25 * cit_score - lat_penalty)), 4)
        metrics = {
            "faithfulness": faith_score,
            "answer_relevancy": rel_score,
            "citation_correctness": round(cit_score, 4),
            "context_precision": round(min(1.0, 0.5 + min(0.4, (c_cnt / 5.0) * 0.4) + graph_bonus), 4),
        }
        return score, metrics

    def _load_from_evaluations(self, queries_map: dict[str, dict[str, Any]]) -> None:
        """Extract actual historical evaluation results from DocumentRegistry."""
        try:
            results = self.doc_registry.list_evaluation_results()
        except Exception:
            return

        for r in results:
            payload = r
            if "payload_json" in r and isinstance(r["payload_json"], (str, dict)):
                p = r["payload_json"]
                if isinstance(p, str):
                    try:
                        payload = {**r, **json.loads(p)}
                    except Exception:
                        payload = r
                elif isinstance(p, dict):
                    payload = {**r, **p}

            question = payload.get("question")
            strategy = (payload.get("strategy") or "").upper()
            if not question or strategy not in {"VECTOR", "GRAPH", "HYBRID"}:
                continue

            norm_q = question.strip().lower()
            if norm_q not in queries_map:
                queries_map[norm_q] = {
                    "query_id": payload.get("query_id", ""),
                    "query_text": question.strip(),
                    "approaches": {},
                    "source": "evaluation",
                    "created_at": payload.get("timestamp", ""),
                }

            # Check if this evaluation result failed (quota exhaustion, timeout, etc.)
            is_err_run = bool(payload.get("errors")) or (
                payload.get("composite_score") is not None and payload.get("composite_score") < 0
            )

            existing = queries_map[norm_q]["approaches"].get(strategy)
            # If we already have a successful evaluated approach, don't overwrite with a failed error run
            if is_err_run and existing and existing.status == "success" and existing.actual_quality_score is not None:
                continue

            # Parse genuine measured metrics
            mets = payload.get("metrics") or {}
            metric_vals: dict[str, float] = {}
            for m_key, m_val in mets.items():
                if isinstance(m_val, dict) and m_val.get("available", True):
                    v = m_val.get("value")
                    if v is not None:
                        metric_vals[m_key] = float(v)
                elif isinstance(m_val, (int, float)):
                    metric_vals[m_key] = float(m_val)

            # Compute actual composite quality score from ground-truth / verified metrics
            composite = payload.get("composite_score")
            if is_err_run:
                composite = None
            elif composite is not None and composite > 0:
                composite = round(float(composite), 4)
            else:
                core_scores = [
                    metric_vals[k]
                    for k in (
                        "answer_correctness",
                        "faithfulness",
                        "answer_relevancy",
                        "token_f1",
                        "exact_match",
                        "context_precision",
                        "context_recall",
                        "retrieval_recall_at_k",
                        "mrr",
                    )
                    if k in metric_vals and metric_vals[k] is not None
                ]
                composite = round(sum(core_scores) / len(core_scores), 4) if core_scores else None

            tot_lat = payload.get("total_latency_ms")
            ret_lat = payload.get("retrieval_latency_ms")
            gen_lat = payload.get("generation_latency_ms")
            answer = payload.get("answer", "")
            citations = payload.get("citations") or []
            graph_ev = payload.get("graph_evidence") or {}
            graph_entities = graph_ev.get("entities", []) if isinstance(graph_ev, dict) else []
            graph_relationships = graph_ev.get("relationships", []) if isinstance(graph_ev, dict) else []

            # If graph evidence is a list of relationship strings, convert to dict format
            if isinstance(graph_relationships, list) and graph_relationships and isinstance(graph_relationships[0], str):
                graph_relationships = [{"relationship": s} for s in graph_relationships]

            approach = ApproachRecord(
                strategy=strategy,
                status="failed" if is_err_run else "success",
                actual_quality_score=composite,
                provisional_quality_score=None,
                quality_score=composite,
                is_provisional=False,
                score_source="Actual Evaluation" if composite is not None else "Telemetry Only",
                retrieval_latency_ms=round(ret_lat, 1) if ret_lat is not None else None,
                generation_latency_ms=round(gen_lat, 1) if gen_lat is not None else None,
                total_latency_ms=round(tot_lat, 1) if tot_lat is not None else None,
                answer=answer,
                citations=citations,
                graph_entities=graph_entities,
                graph_relationships=graph_relationships,
                metrics=metric_vals,
                actual_metrics=metric_vals,
                provisional_metrics={},
                run_id=payload.get("run_id", ""),
                source="evaluation",
            )

            if existing and existing.actual_quality_score is not None and composite is not None:
                # Average multiple valid evaluations
                existing.actual_quality_score = round((existing.actual_quality_score + composite) / 2.0, 4)
                existing.quality_score = existing.actual_quality_score
                for mk, mv in metric_vals.items():
                    existing.actual_metrics[mk] = round((existing.actual_metrics.get(mk, mv) + mv) / 2.0, 4)
                    existing.metrics[mk] = existing.actual_metrics[mk]
            elif not existing or (existing.actual_quality_score is None and composite is not None):
                queries_map[norm_q]["approaches"][strategy] = approach

    def _load_from_comparisons(self, queries_map: dict[str, dict[str, Any]], kb_id: str | None = None) -> None:
        """Extract multi-strategy comparison results from UserRegistry, keeping provisional scores separate."""
        try:
            runs = self.user_registry.list_comparison_runs(kb_id=kb_id)
        except Exception:
            return

        for run in runs:
            try:
                full_run = self.user_registry.get_comparison_run(run.run_id)
            except Exception:
                continue
            if not full_run or not full_run.query:
                continue

            norm_q = full_run.query.strip().lower()
            if norm_q not in queries_map:
                queries_map[norm_q] = {
                    "query_id": full_run.run_id,
                    "query_text": full_run.query.strip(),
                    "approaches": {},
                    "source": "comparison",
                    "created_at": full_run.created_at,
                }

            for res in getattr(full_run, "results", []):
                strat = (res.strategy or "").upper()
                if strat not in {"VECTOR", "GRAPH", "HYBRID"}:
                    continue

                # Calculate cost if tokens available
                p_tok = res.prompt_tokens
                c_tok = res.completion_tokens
                t_tok = (p_tok + c_tok) if (p_tok is not None and c_tok is not None) else None
                cost_usd = None
                cost_inr = None
                if p_tok is not None and c_tok is not None:
                    cost_est = self.cost_service.calculate_cost(
                        model_name=res.model_name or "qwen/qwen3.8-27b",
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                    )
                    cost_usd = cost_est.cost_usd
                    cost_inr = cost_est.cost_inr

                # Citations & Graph Evidence
                citations = res.citations or []
                graph_entities = []
                graph_relationships = []
                if res.graph_evidence:
                    if isinstance(res.graph_evidence, dict):
                        graph_entities = res.graph_evidence.get("entities", [])
                        graph_relationships = res.graph_evidence.get("relationships", [])

                # Calculate provisional evidence and quality score
                prov_score, prov_metrics = self._compute_provisional_score(
                    status=res.status or "success",
                    citations=citations,
                    graph_entities=graph_entities,
                    graph_relationships=graph_relationships,
                    answer=res.answer or "",
                    latency_ms=res.total_latency_ms,
                )

                existing = queries_map[norm_q]["approaches"].get(strat)
                if existing:
                    # Enrich telemetry without overwriting actual evaluation scores
                    if existing.prompt_tokens is None and p_tok is not None:
                        existing.prompt_tokens = p_tok
                        existing.completion_tokens = c_tok
                        existing.total_tokens = t_tok
                        existing.cost_usd = cost_usd
                        existing.cost_inr = cost_inr
                    if not existing.citations and citations:
                        existing.citations = citations
                    if not existing.graph_entities and graph_entities:
                        existing.graph_entities = graph_entities
                        existing.graph_relationships = graph_relationships
                    # If this query had NO actual evaluation, record provisional score separately
                    if existing.actual_quality_score is None:
                        existing.provisional_quality_score = prov_score
                        existing.quality_score = prov_score
                        existing.provisional_metrics = prov_metrics
                        existing.is_provisional = True
                        existing.score_source = "Estimated / Provisional"
                else:
                    queries_map[norm_q]["approaches"][strat] = ApproachRecord(
                        strategy=strat,
                        status=res.status or "success",
                        actual_quality_score=None,
                        provisional_quality_score=prov_score,
                        quality_score=prov_score,
                        is_provisional=True,
                        score_source="Estimated / Provisional",
                        retrieval_latency_ms=round(res.retrieval_latency_ms, 1) if res.retrieval_latency_ms is not None else None,
                        generation_latency_ms=round(res.generation_latency_ms, 1) if res.generation_latency_ms is not None else None,
                        total_latency_ms=round(res.total_latency_ms, 1) if res.total_latency_ms is not None else None,
                        prompt_tokens=p_tok,
                        completion_tokens=c_tok,
                        total_tokens=t_tok,
                        cost_usd=cost_usd,
                        cost_inr=cost_inr,
                        answer=res.answer or "",
                        citations=citations,
                        graph_entities=graph_entities,
                        graph_relationships=graph_relationships,
                        metrics={},  # Actual metrics remain empty for comparison runs
                        actual_metrics={},
                        provisional_metrics=prov_metrics,
                        run_id=run.run_id,
                        source="comparison",
                    )

    def _load_from_chat_messages(self, queries_map: dict[str, dict[str, Any]]) -> None:
        """Extract queries and assistant telemetry from chat conversations."""
        try:
            conversations = self.user_registry.list_conversations()
        except Exception:
            return

        for conv in conversations:
            try:
                msgs = self.user_registry.list_messages(conv.conversation_id)
            except Exception:
                continue

            last_user_q: str | None = None
            for m in msgs:
                if m.role == "user" and m.content:
                    last_user_q = m.content.strip()
                elif m.role == "assistant" and last_user_q:
                    norm_q = last_user_q.lower()
                    strat = (getattr(m, "retrieval_strategy", None) or "").upper()
                    if strat in {"VECTOR", "GRAPH", "HYBRID"}:
                        if norm_q not in queries_map:
                            queries_map[norm_q] = {
                                "query_id": m.message_id,
                                "query_text": last_user_q,
                                "approaches": {},
                                "source": "chat",
                                "created_at": m.created_at,
                            }

                        existing = queries_map[norm_q]["approaches"].get(strat)
                        if not existing:
                            queries_map[norm_q]["approaches"][strat] = ApproachRecord(
                                strategy=strat,
                                status="success",
                                actual_quality_score=None,
                                provisional_quality_score=None,
                                quality_score=None,
                                is_provisional=False,
                                score_source="Telemetry Only",
                                retrieval_latency_ms=round(m.retrieval_latency_ms, 1) if getattr(m, "retrieval_latency_ms", None) is not None else None,
                                generation_latency_ms=round(m.generation_latency_ms, 1) if getattr(m, "generation_latency_ms", None) is not None else None,
                                total_latency_ms=round(m.total_latency_ms, 1) if getattr(m, "total_latency_ms", None) is not None else None,
                                prompt_tokens=getattr(m, "prompt_tokens", None),
                                completion_tokens=getattr(m, "completion_tokens", None),
                                total_tokens=getattr(m, "total_tokens", None),
                                cost_usd=getattr(m, "cost_usd", None),
                                cost_inr=getattr(m, "cost_inr", None),
                                answer=m.content,
                                run_id=conv.conversation_id,
                                source="chat",
                            )
                    last_user_q = None

    def _compute_approach_pillars(
        self, approaches: dict[str, ApproachRecord], weights: ScoringWeights
    ) -> None:
        """Compute the 4 normalized evaluation pillar scores and the overall weighted score for each approach."""
        # 1. Determine min/max latency and cost among valid approaches for relative normalization
        valid_apps = [a for a in approaches.values() if a.status == "success"]
        lats = [a.total_latency_ms for a in valid_apps if a.total_latency_ms is not None and a.total_latency_ms > 0]
        min_lat = min(lats) if lats else 500.0
        max_lat = max(lats) if lats else 15000.0

        costs = [a.cost_usd for a in valid_apps if a.cost_usd is not None and a.cost_usd >= 0]
        min_cost = min(costs) if costs else 0.00001
        max_cost = max(costs) if costs else 0.005

        w = weights.normalized()

        for a in approaches.values():
            if a.status != "success":
                continue

            # A. Answer Quality Score
            # Use actual metrics if available; fallback to provisional metrics only if provisional
            m_dict = a.actual_metrics if not a.is_provisional else {**a.actual_metrics, **a.provisional_metrics}
            ans_keys = ("answer_correctness", "faithfulness", "answer_relevancy")
            ans_vals = [m_dict[k] for k in ans_keys if k in m_dict and m_dict[k] is not None]
            if ans_vals:
                a.answer_quality_score = round(sum(ans_vals) / len(ans_vals), 4)
            elif a.quality_score is not None:
                a.answer_quality_score = a.quality_score
            else:
                a.answer_quality_score = None

            # B. Retrieval Quality Score
            ret_keys = ("context_precision", "context_recall", "retrieval_recall_at_k")
            ret_vals = [m_dict[k] for k in ret_keys if k in m_dict and m_dict[k] is not None]
            a.retrieval_quality_score = round(sum(ret_vals) / len(ret_vals), 4) if ret_vals else None

            # C. Latency Score (lower latency -> higher score)
            if a.total_latency_ms is not None and a.total_latency_ms > 0:
                if max_lat > min_lat:
                    a.latency_score = round(max(0.0, min(1.0, 1.0 - (a.total_latency_ms - min_lat) / (max_lat - min_lat))), 4)
                else:
                    a.latency_score = round(max(0.0, min(1.0, 1.0 - a.total_latency_ms / 20000.0)), 4)
            else:
                a.latency_score = None

            # D. Cost Score (lower cost -> higher score)
            if a.cost_usd is not None and a.cost_usd >= 0:
                if max_cost > min_cost:
                    a.cost_score = round(max(0.0, min(1.0, 1.0 - (a.cost_usd - min_cost) / (max_cost - min_cost))), 4)
                else:
                    a.cost_score = round(max(0.0, min(1.0, 1.0 - a.cost_usd / 0.005)), 4)
            elif a.total_tokens is not None and a.total_tokens > 0:
                a.cost_score = round(max(0.0, min(1.0, 1.0 - a.total_tokens / 10000.0)), 4)
            else:
                a.cost_score = None

            # E. Overall Weighted Score (with dynamic missing-metric re-weighting)
            active_pairs = []
            if a.answer_quality_score is not None:
                active_pairs.append((a.answer_quality_score, w.answer_quality))
            if a.retrieval_quality_score is not None:
                active_pairs.append((a.retrieval_quality_score, w.retrieval_quality))
            if a.latency_score is not None:
                active_pairs.append((a.latency_score, w.latency))
            if a.cost_score is not None:
                active_pairs.append((a.cost_score, w.cost))

            if active_pairs:
                tot_active_w = sum(p[1] for p in active_pairs)
                if tot_active_w > 0:
                    a.overall_weighted_score = round(
                        max(0.0, min(1.0, sum(v * (wt / tot_active_w) for v, wt in active_pairs))), 4
                    )
            else:
                a.overall_weighted_score = None

    def _determine_query_best_approach(
        self, approaches: dict[str, ApproachRecord]
    ) -> tuple[str, str, str]:
        """Determine the empirical winning architecture for a single query using weighted performance scores."""
        if not approaches:
            return "Insufficient data", "No execution results available for this query.", ""

        valid_apps = [a for a in approaches.values() if a.status == "success"]
        if not valid_apps:
            return "Failed", "All evaluated approaches failed execution.", ""

        if len(valid_apps) == 1:
            only = valid_apps[0]
            score_txt = f"{only.overall_weighted_score:.3f}" if only.overall_weighted_score is not None else "N/A"
            return (
                only.strategy,
                f"{only.strategy} was the only approach executed for this query (Weighted Score: {score_txt}).",
                f"Single approach evaluated ({only.score_source})",
            )

        # Multi-approach comparison: rank by overall weighted score descending
        scored_apps = [a for a in valid_apps if a.overall_weighted_score is not None]
        if scored_apps:
            scored_apps.sort(key=lambda a: a.overall_weighted_score or 0.0, reverse=True)
            top = scored_apps[0]
            if len(scored_apps) >= 2:
                runner_up = scored_apps[1]
                diff = (top.overall_weighted_score or 0.0) - (runner_up.overall_weighted_score or 0.0)
                lat_diff = abs((top.total_latency_ms or 0.0) - (runner_up.total_latency_ms or 0.0))

                if diff < 0.015 and lat_diff < 150:
                    return (
                        "Tie",
                        f"Tie between {top.strategy} ({top.overall_weighted_score:.3f}) and {runner_up.strategy} ({runner_up.overall_weighted_score:.3f}) with nearly identical performance.",
                        f"Score diff: {diff:.3f}",
                    )

                margin_pct = (diff / max(runner_up.overall_weighted_score or 0.01, 0.01)) * 100
                src_txt = f"[{top.score_source}]" if top.score_source == runner_up.score_source else f"[{top.score_source} vs {runner_up.score_source}]"
                return (
                    top.strategy,
                    f"{top.strategy} achieved higher overall weighted score ({top.overall_weighted_score:.3f} vs {runner_up.overall_weighted_score:.3f}) {src_txt}.",
                    f"+{margin_pct:.1f}% weighted score vs {runner_up.strategy}",
                )
            else:
                return (
                    top.strategy,
                    f"{top.strategy} achieved overall weighted score of {top.overall_weighted_score:.3f}.",
                    f"Weighted score: {top.overall_weighted_score:.3f}",
                )

        # Fallback to latency if scores unavailable
        sorted_by_lat = sorted(valid_apps, key=lambda a: a.total_latency_ms or 99999.0)
        fastest = sorted_by_lat[0]
        return (
            fastest.strategy,
            f"{fastest.strategy} executed fastest ({fastest.total_latency_ms:,.0f} ms).",
            f"Fastest execution",
        )

    def _build_category_intelligence(
        self,
        category: str,
        records: list[QueryIntelligenceRecord],
        weights: ScoringWeights,
    ) -> CategoryIntelligence:
        """Aggregate records into category-level benchmarks and recommendations across all 4 evaluation pillars."""
        desc = CATEGORY_DESCRIPTIONS.get(category, "")
        color = CATEGORY_COLOR_MAP.get(category, "#6366F1")
        query_count = len(records)

        actual_cnt = sum(1 for r in records if not r.is_provisional)
        prov_cnt = sum(1 for r in records if r.is_provisional)

        # Collect stats per approach across 4 pillars
        stats: dict[str, dict[str, Any]] = {}
        win_counts: dict[str, int] = {"VECTOR": 0, "GRAPH": 0, "HYBRID": 0}

        for strat in ("VECTOR", "GRAPH", "HYBRID"):
            overall_scores = []
            ans_scores = []
            ret_scores = []
            faith_scores = []
            rel_scores = []
            corr_scores = []
            ctx_prec_scores = []
            ctx_rec_scores = []
            ret_rec_scores = []
            tot_lats = []
            ret_lats = []
            gen_lats = []
            tokens = []
            costs = []

            for r in records:
                app = r.approaches.get(strat)
                if not app or app.status != "success":
                    continue
                if app.overall_weighted_score is not None:
                    overall_scores.append(app.overall_weighted_score)
                if app.answer_quality_score is not None:
                    ans_scores.append(app.answer_quality_score)
                if app.retrieval_quality_score is not None:
                    ret_scores.append(app.retrieval_quality_score)
                if app.total_latency_ms is not None:
                    tot_lats.append(app.total_latency_ms)
                if app.retrieval_latency_ms is not None:
                    ret_lats.append(app.retrieval_latency_ms)
                if app.generation_latency_ms is not None:
                    gen_lats.append(app.generation_latency_ms)
                if app.total_tokens is not None:
                    tokens.append(app.total_tokens)
                if app.cost_usd is not None:
                    costs.append(app.cost_usd)

                m = app.actual_metrics if not app.is_provisional else {**app.actual_metrics, **app.provisional_metrics}
                if "faithfulness" in m:
                    faith_scores.append(m["faithfulness"])
                if "answer_relevancy" in m:
                    rel_scores.append(m["answer_relevancy"])
                if "answer_correctness" in m:
                    corr_scores.append(m["answer_correctness"])
                if "context_precision" in m:
                    ctx_prec_scores.append(m["context_precision"])
                if "context_recall" in m:
                    ctx_rec_scores.append(m["context_recall"])
                if "retrieval_recall_at_k" in m:
                    ret_rec_scores.append(m["retrieval_recall_at_k"])

            stats[strat] = {
                "overall_weighted_score": round(statistics.mean(overall_scores), 4) if overall_scores else None,
                "answer_quality_score": round(statistics.mean(ans_scores), 4) if ans_scores else None,
                "retrieval_quality_score": round(statistics.mean(ret_scores), 4) if ret_scores else None,
                "faithfulness": round(statistics.mean(faith_scores), 4) if faith_scores else None,
                "answer_relevancy": round(statistics.mean(rel_scores), 4) if rel_scores else None,
                "answer_correctness": round(statistics.mean(corr_scores), 4) if corr_scores else None,
                "context_precision": round(statistics.mean(ctx_prec_scores), 4) if ctx_prec_scores else None,
                "context_recall": round(statistics.mean(ctx_rec_scores), 4) if ctx_rec_scores else None,
                "retrieval_recall_at_k": round(statistics.mean(ret_rec_scores), 4) if ret_rec_scores else None,
                "retrieval_latency_ms": round(statistics.mean(ret_lats), 1) if ret_lats else None,
                "generation_latency_ms": round(statistics.mean(gen_lats), 1) if gen_lats else None,
                "total_latency_ms": round(statistics.mean(tot_lats), 1) if tot_lats else None,
                "tokens": round(statistics.mean(tokens), 0) if tokens else None,
                "cost_usd": round(statistics.mean(costs), 6) if costs else None,
                "sample_count": len(tot_lats),
            }

        # Win counts
        for r in records:
            if r.best_approach in win_counts:
                win_counts[r.best_approach] += 1

        total_comparisons = sum(
            1 for r in records if len([a for a in r.approaches.values() if a.status == "success"]) >= 2
        )

        # Confidence level calculation
        if query_count == 0:
            confidence = "Insufficient data"
            suff_reason = "No historical queries recorded for this category yet."
        elif total_comparisons == 0:
            confidence = "Insufficient data"
            suff_reason = f"{query_count} query/queries detected, but no multi-strategy comparative runs exist."
        elif actual_cnt >= 3:
            confidence = "High"
            suff_reason = f"{actual_cnt} query/queries with verified actual evaluation benchmarks."
        elif actual_cnt >= 1 or total_comparisons >= 3:
            confidence = "Medium"
            suff_reason = f"{actual_cnt} actual and {prov_cnt} provisional comparative benchmark(s)."
        else:
            confidence = "Low"
            suff_reason = f"Sparse sample size ({query_count} query/queries). Interpret results as provisional."

        # Handle zero data case
        if query_count == 0 or total_comparisons == 0:
            return CategoryIntelligence(
                category_name=category,
                query_count=query_count,
                actual_query_count=actual_cnt,
                provisional_query_count=prov_cnt,
                color=color,
                description=desc,
                best_approach="Insufficient data",
                best_answer_quality_arch="Insufficient data",
                best_retrieval_quality_arch="Insufficient data",
                fastest_arch="Insufficient data",
                most_cost_efficient_arch="Insufficient data",
                confidence_level=confidence,
                data_sufficiency_reason=suff_reason,
                empirical_reason="Execute 3-way evaluations or comparisons in this category to establish empirical recommendations.",
                margin_summary="Awaiting comparative benchmark",
                tradeoffs="Comparative telemetry insufficient for definitive ranking.",
                strengths=["Data collection pending"],
                approach_stats=stats,
                win_counts=win_counts,
                query_records=records,
            )

        # Identify sub-winners across the 4 pillars
        # 1. Best Overall Architecture
        candidates = []
        for strat in ("VECTOR", "GRAPH", "HYBRID"):
            w = win_counts[strat]
            ov = stats[strat]["overall_weighted_score"] or 0.0
            candidates.append((ov, w, strat))
        candidates.sort(reverse=True)
        top_strat = candidates[0][2]
        runner_strat = candidates[1][2]

        # 2. Best Answer Quality
        ans_cand = [(stats[s]["answer_quality_score"] or 0.0, s) for s in ("VECTOR", "GRAPH", "HYBRID") if stats[s]["answer_quality_score"] is not None]
        best_ans = max(ans_cand, key=lambda x: x[0])[1] if ans_cand else "Insufficient data"

        # 3. Best Retrieval Quality
        ret_cand = [(stats[s]["retrieval_quality_score"] or 0.0, s) for s in ("VECTOR", "GRAPH", "HYBRID") if stats[s]["retrieval_quality_score"] is not None]
        best_ret = max(ret_cand, key=lambda x: x[0])[1] if ret_cand else "Insufficient data"

        # 4. Fastest Architecture
        lat_cand = [(stats[s]["total_latency_ms"], s) for s in ("VECTOR", "GRAPH", "HYBRID") if stats[s]["total_latency_ms"] is not None]
        fastest = min(lat_cand, key=lambda x: x[0])[1] if lat_cand else "Insufficient data"

        # 5. Most Cost-Efficient
        cost_cand = [(stats[s]["cost_usd"] if stats[s]["cost_usd"] is not None else (stats[s]["tokens"] or 999999), s) for s in ("VECTOR", "GRAPH", "HYBRID")]
        cost_cand_valid = [c for c in cost_cand if c[0] is not None]
        most_cost = min(cost_cand_valid, key=lambda x: x[0])[1] if cost_cand_valid else "Insufficient data"

        top_stats = stats[top_strat]
        runner_stats = stats[runner_strat]

        top_score = top_stats["overall_weighted_score"]
        runner_score = runner_stats["overall_weighted_score"]
        top_lat = top_stats["total_latency_ms"]
        runner_lat = runner_stats["total_latency_ms"]

        strengths: list[str] = []
        margin_parts: list[str] = []

        if top_score is not None and runner_score is not None and top_score > runner_score:
            pct = ((top_score - runner_score) / max(runner_score, 0.01)) * 100
            margin_parts.append(f"+{pct:.1f}% weighted score")
            strengths.append(f"Leading overall weighted score ({top_score:.3f} vs {runner_score:.3f})")
        if top_lat is not None and runner_lat is not None and top_lat < runner_lat:
            diff_ms = runner_lat - top_lat
            margin_parts.append(f"-{diff_ms:.0f}ms latency")
            strengths.append(f"Faster end-to-end response time (-{diff_ms:.0f}ms)")
        elif top_lat is not None and runner_lat is not None and top_lat > runner_lat:
            strengths.append(f"Delivers higher quality despite a {(top_lat - runner_lat):.0f}ms latency overhead")

        if win_counts[top_strat] > 0:
            strengths.append(f"Won {win_counts[top_strat]} of {total_comparisons} comparative query evaluations")

        margin_summary = ", ".join(margin_parts) if margin_parts else f"Leading approach with {win_counts[top_strat]} wins"
        empirical_reason = (
            f"{top_strat} demonstrated superior overall performance across {query_count} benchmarked "
            f"queries in this category, leading in {win_counts[top_strat]} comparative runs."
        )

        tradeoffs = (
            f"{top_strat} optimal for {category.lower()} queries. "
            f"Average latency is {top_lat or 'N/A'}ms with estimated token cost of "
            f"${top_stats['cost_usd'] or 0.0:.5f} USD."
        )

        return CategoryIntelligence(
            category_name=category,
            query_count=query_count,
            actual_query_count=actual_cnt,
            provisional_query_count=prov_cnt,
            color=color,
            description=desc,
            best_approach=top_strat,
            best_answer_quality_arch=best_ans,
            best_retrieval_quality_arch=best_ret,
            fastest_arch=fastest,
            most_cost_efficient_arch=most_cost,
            confidence_level=confidence,
            data_sufficiency_reason=suff_reason,
            empirical_reason=empirical_reason,
            margin_summary=margin_summary,
            tradeoffs=tradeoffs,
            strengths=strengths,
            approach_stats=stats,
            win_counts=win_counts,
            query_records=records,
        )
