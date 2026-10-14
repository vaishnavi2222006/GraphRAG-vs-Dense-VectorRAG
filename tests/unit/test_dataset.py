from __future__ import annotations

from pathlib import Path

from rag_benchmark_lab.evaluation.dataset import load_dataset, sample_dataset, validate_example


def test_sample_dataset_contains_examples() -> None:
    assert len(sample_dataset()) >= 3


def test_load_dataset_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "dataset.jsonl"
    path.write_text('{"query_id":"1","question":"What is X?"}\n', encoding="utf-8")
    examples = load_dataset(path)
    assert examples[0].query_id == "1"


def test_validate_example_rejects_empty_question() -> None:
    from rag_benchmark_lab.schemas.evaluation import EvaluationExample

    example = EvaluationExample(query_id="1", question="")
    try:
        validate_example(example)
    except ValueError:
        assert True
    else:
        raise AssertionError("Expected ValueError")
