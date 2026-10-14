from __future__ import annotations

from pathlib import Path

from rag_benchmark_lab.app import build_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import load_dataset


def main() -> None:
    context = build_context()
    dataset_path = Path("data/sample_evaluation/sample_evaluation.jsonl")
    examples = load_dataset(dataset_path)
    runner = BenchmarkRunner(context.query_service, context.result_store)
    result = runner.run(examples)
    print(result["run_id"])
    print(result["aggregates"])


if __name__ == "__main__":
    main()
