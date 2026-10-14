from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from ..services.document_registry import DocumentRegistry


class ResultStore:
    def __init__(self, registry: DocumentRegistry) -> None:
        self.registry = registry

    def save_run(self, run_id: str, payload: dict[str, Any]) -> None:
        self.registry.save_evaluation_run(run_id, payload)

    def save_result(self, result_id: str, run_id: str, payload: dict[str, Any]) -> None:
        self.registry.save_evaluation_result(result_id, run_id, payload)

    def list_runs(self) -> list[dict[str, Any]]:
        return self.registry.list_evaluation_runs()

    def list_results(self, run_id: str | None = None) -> list[dict[str, Any]]:
        return self.registry.list_evaluation_results(run_id)

