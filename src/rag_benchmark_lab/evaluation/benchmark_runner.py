from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import uuid

from ..schemas.documents import RetrievalStrategy
from ..schemas.evaluation import EvaluationExample, EvaluationRunSummary, MetricResult, RecommendationResult
from ..services.document_registry import DocumentRegistry
from ..services.query_service import QueryService
from .dataset import dataset_hash
from .metrics import latency_summary, metric_bundle
from .recommendation import evaluate_approach, rank_benchmark_approaches, recommend_best_approach
from .result_store import ResultStore


class BenchmarkRunner:
    def __init__(self, query_service: QueryService, result_store: ResultStore) -> None:
        self.query_service = query_service
        self.result_store = result_store

    def run(
        self,
        examples: list[EvaluationExample],
        strategies: list[RetrievalStrategy] | None = None,
        document_ids: list[str] | None = None,
    ) -> dict:
        """Execute four-way benchmark evaluation across the supplied test questions."""
        strategies = strategies or [
            RetrievalStrategy.VECTOR,
            RetrievalStrategy.GRAPH,
            RetrievalStrategy.HYBRID,
            RetrievalStrategy.DIRECT,
        ]
        run_id = str(uuid.uuid4())
        llm_svc = getattr(self.query_service, "llm_service", None)
        mock_mode_flag = getattr(llm_svc, "mock_mode", False)
        model_name = getattr(llm_svc, "model_name", "unknown")

        per_strategy: dict[str, list[float]] = {strategy.value: [] for strategy in strategies}
        per_strategy_metrics: dict[str, dict[str, list[float]]] = {strategy.value: defaultdict(list) for strategy in strategies}
        rows: list[dict] = []
        query_summaries: list[dict] = []

        daily_quota_exhausted = False
        quota_reason = ""

        for example in examples:
            has_ground_truth = bool(example.expected_answer and str(example.expected_answer).strip())
            approach_evals = {}
            query_row_map = {}

            for strategy in strategies:
                skip_reason = quota_reason if daily_quota_exhausted else None
                call_kwargs = {}
                if skip_reason:
                    call_kwargs["skip_generation_reason"] = skip_reason

                try:
                    answer = self.query_service.answer(
                        example.question,
                        manual_strategy=strategy,
                        document_ids=document_ids,
                        **call_kwargs,
                    )
                except TypeError:
                    answer = self.query_service.answer(
                        example.question,
                        manual_strategy=strategy,
                        document_ids=document_ids,
                    )

                errors = getattr(answer, "errors", []) or []
                warnings = getattr(answer, "warnings", []) or []
                retrieved_sources = getattr(answer, "retrieved_sources", []) or []
                citations = getattr(answer, "citations", []) or []
                graph_evidence = getattr(answer, "graph_evidence", None)
                sel_strategy = getattr(answer, "selected_strategy", strategy)
                sel_strat_val = getattr(sel_strategy, "value", str(sel_strategy))
                ans_model_name = getattr(answer, "model_name", None) or model_name

                # Check if this strategy triggered quota exhaustion
                for err in errors:
                    if "QuotaExceededError" in err or "quota" in err.lower() or "resource_exhausted" in err.lower() or "RateLimitError" in err:
                        daily_quota_exhausted = True
                        prov_name = getattr(getattr(self.query_service, "llm_service", None), "provider_name", "LLM")
                        quota_reason = (
                            f"{prov_name}'s available quota or rate limit has been reached for the configured model. "
                            "Subsequent generations will be gracefully skipped."
                        )
                        break

                metrics = metric_bundle(
                    answer,
                    example.expected_answer,
                    example.relevant_document_ids,
                    example.reference_contexts,
                    question=example.question,
                )

                if getattr(answer, "total_latency_ms", None) is not None:
                    per_strategy[strategy.value].append(answer.total_latency_ms)

                for m_name, m_val in metrics.items():
                    if m_val.value is not None:
                        per_strategy_metrics[strategy.value][m_name].append(m_val.value)

                app_eval = evaluate_approach(strategy.value, answer, metrics, has_ground_truth)
                approach_evals[strategy.value] = app_eval

                graph_paths_count = None
                if strategy.value in (RetrievalStrategy.GRAPH.value, RetrievalStrategy.HYBRID.value):
                    if graph_evidence and getattr(graph_evidence, "relationships", None):
                        graph_paths_count = len(graph_evidence.relationships)
                    else:
                        graph_paths_count = 0

                citations_data = [
                    {
                        "document_id": getattr(c, "document_id", ""),
                        "document_name": getattr(c, "document_name", ""),
                        "chunk_id": getattr(c, "chunk_id", ""),
                        "chunk_index": getattr(c, "chunk_index", 0),
                        "score": getattr(c, "score", None),
                    }
                    for c in citations
                ]

                row = {
                    "run_id": run_id,
                    "query_id": example.query_id,
                    "strategy": strategy.value,
                    "question": example.question,
                    "answer": getattr(answer, "answer", ""),
                    "expected_answer": example.expected_answer,
                    "relevant_document_ids": example.relevant_document_ids,
                    "selected_strategy": sel_strat_val,
                    "model_name": ans_model_name,
                    "retrieval_latency_ms": getattr(answer, "retrieval_latency_ms", None),
                    "generation_latency_ms": getattr(answer, "generation_latency_ms", None),
                    "total_latency_ms": getattr(answer, "total_latency_ms", None),
                    "retrieved_chunk_count": len(retrieved_sources) if strategy.value != RetrievalStrategy.DIRECT.value else None,
                    "graph_path_count": graph_paths_count,
                    "citations": citations_data,
                    "metrics": {name: asdict(metric) for name, metric in metrics.items()},
                    "mock_mode": mock_mode_flag,
                    "warnings": warnings,
                    "errors": errors,
                    "status": app_eval.status_label,
                    "composite_score": app_eval.composite_score,
                    "strengths": app_eval.strengths,
                    "weaknesses": app_eval.weaknesses,
                    "is_ground_truth_based": has_ground_truth,
                }

                query_row_map[strategy.value] = row

            # Determine query-level recommendation
            query_rec = recommend_best_approach(approach_evals, has_ground_truth, question=example.question)
            for strat_val, row in query_row_map.items():
                row["recommendation"] = asdict(query_rec)
                rows.append(row)
                self.result_store.save_result(f"{run_id}:{example.query_id}:{strat_val}", run_id, row)

            query_summaries.append(
                {
                    "query_id": example.query_id,
                    "question": example.question,
                    "expected_answer": example.expected_answer,
                    "recommendation": asdict(query_rec),
                    "approaches": {s: asdict(eval_) for s, eval_ in approach_evals.items()},
                }
            )

        aggregates = {
            strategy: {
                "latency": latency_summary(latencies),
                "queries": len(examples),
                "metrics": {
                    m_name: round(sum(m_vals) / len(m_vals), 4) if m_vals else None
                    for m_name, m_vals in per_strategy_metrics[strategy].items()
                },
            }
            for strategy, latencies in per_strategy.items()
        }

        # Compute dataset-level aggregate ranking and executive summary
        ranking_summary = rank_benchmark_approaches(rows, query_summaries)

        summary_payload = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_id": "benchmark_dataset",
            "dataset_hash": dataset_hash(examples),
            "strategies": [strategy.value for strategy in strategies],
            "total_queries": len(examples),
            "mock_mode": mock_mode_flag,
            "model_name": model_name,
            "document_ids": document_ids or [],
            "overall_best": ranking_summary.get("overall_best"),
            "overall_reason": ranking_summary.get("overall_reason"),
            "winner_counts": ranking_summary.get("winner_counts", {}),
            "ties_count": ranking_summary.get("ties_count", 0),
            "inconclusive_count": ranking_summary.get("inconclusive_count", 0),
            "failed_count": ranking_summary.get("failed_count", 0),
        }
        self.result_store.save_run(run_id, summary_payload)

        return {
            "run_id": run_id,
            "summary": summary_payload,
            "rows": rows,
            "aggregates": aggregates,
            "query_summaries": query_summaries,
            "recommendation_summary": ranking_summary,
        }


