from __future__ import annotations

try:
    from ragas import evaluate  # type: ignore
except Exception:  # pragma: no cover
    evaluate = None


def ragas_available() -> bool:
    return evaluate is not None

