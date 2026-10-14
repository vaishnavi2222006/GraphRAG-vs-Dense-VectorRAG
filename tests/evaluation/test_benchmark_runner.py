from __future__ import annotations

from dataclasses import dataclass

from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import sample_dataset
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.schemas.documents import RetrievedChunk, RetrievalStrategy
from rag_benchmark_lab.services.document_registry import DocumentRegistry


@dataclass
class FakeQueryService:
    def answer(self, question: str, manual_strategy=None, document_ids=None):
        chunk = RetrievedChunk(chunk_id="c1", document_id="sample-doc-1", document_name="demo", chunk_index=0, text="answer evidence", score=0.8)
        return type(
            "Answer",
            (),
            {
                "answer": f"{manual_strategy.value} answer",
                "selected_strategy": manual_strategy,
                "citations": [],
                "retrieved_sources": [chunk],
                "retrieval_latency_ms": 1.0,
                "generation_latency_ms": 1.0,
                "total_latency_ms": 2.0,
                "warnings": [],
                "errors": [],
            },
        )()


def test_benchmark_runner_persists_results(tmp_path):
    registry = DocumentRegistry(tmp_path / "registry.sqlite3")
    runner = BenchmarkRunner(FakeQueryService(), ResultStore(registry))
    result = runner.run(sample_dataset(), strategies=[RetrievalStrategy.VECTOR])
    assert result["rows"]
    assert registry.list_evaluation_runs()
