from __future__ import annotations

from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from statistics import mean
import uuid

from ..schemas.documents import RetrievalStrategy
from ..schemas.evaluation import EvaluationExample, EvaluationRunSummary, MetricResult
from ..services.document_registry import DocumentRegistry
from ..services.query_service import QueryService
from .dataset import dataset_hash
from .metrics import latency_summary, metric_bundle
from .result_store import ResultStore


class BenchmarkRunner:
    def __init__(self, query_service: QueryService, result_store: ResultStore) -> None:
        self.query_service = query_service
        self.result_store = result_store

    def run(self, examples: list[EvaluationExample], strategies: list[RetrievalStrategy] | None = None) -> dict:
        strategies = strategies or [RetrievalStrategy.VECTOR, RetrievalStrategy.GRAPH, RetrievalStrategy.HYBRID]
        run_id = str(uuid.uuid4())
        summary_payload = {
            "run_id": run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_id": "uploaded_dataset",
            "dataset_hash": dataset_hash(examples),
            "strategies": [strategy.value for strategy in strategies],
            "total_queries": len(examples),
        }
        self.result_store.save_run(run_id, summary_payload)

        per_strategy: dict[str, list[float]] = {strategy.value: [] for strategy in strategies}
        rows: list[dict] = []
        for example in examples:
            for strategy in strategies:
                answer = self.query_service.answer(example.question, manual_strategy=strategy)
                metrics = metric_bundle(answer, example.expected_answer, example.relevant_document_ids, example.reference_contexts)
                per_strategy[strategy.value].append(answer.total_latency_ms or 0.0)
                row = {
                    "run_id": run_id,
                    "query_id": example.query_id,
                    "strategy": strategy.value,
                    "question": example.question,
                    "answer": answer.answer,
                    "selected_strategy": answer.selected_strategy.value,
                    "retrieval_latency_ms": answer.retrieval_latency_ms,
                    "generation_latency_ms": answer.generation_latency_ms,
                    "total_latency_ms": answer.total_latency_ms,
                    "metrics": {name: asdict(metric) for name, metric in metrics.items()},
                    "warnings": answer.warnings,
                    "errors": answer.errors,
                }
                rows.append(row)
                self.result_store.save_result(f"{run_id}:{example.query_id}:{strategy.value}", run_id, row)

        aggregates = {
            strategy: {
                "latency": latency_summary(latencies),
                "queries": len(examples),
            }
            for strategy, latencies in per_strategy.items()
        }
        return {"run_id": run_id, "summary": summary_payload, "rows": rows, "aggregates": aggregates}

