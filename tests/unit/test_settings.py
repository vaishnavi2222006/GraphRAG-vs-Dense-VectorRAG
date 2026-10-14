from __future__ import annotations

import pytest

from rag_benchmark_lab.config.settings import AppSettings


def test_settings_validation_rejects_invalid_chunk_overlap(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CHUNK_SIZE", "100")
    monkeypatch.setenv("CHUNK_OVERLAP", "100")
    with pytest.raises(ValueError):
        AppSettings.from_env()


def test_settings_to_dict_masks_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEO4J_PASSWORD", "secret")
    settings = AppSettings.from_env()
    payload = settings.to_dict()
    assert payload["neo4j_password"] == "***"
