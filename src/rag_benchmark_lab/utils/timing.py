from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter


@dataclass(slots=True)
class TimerResult:
    name: str
    elapsed_ms: float


@contextmanager
def timed(name: str):
    start = perf_counter()
    result = {"name": name, "elapsed_ms": 0.0}
    try:
        yield result
    finally:
        result["elapsed_ms"] = (perf_counter() - start) * 1000.0


def elapsed_ms(start_seconds: float) -> float:
    return (perf_counter() - start_seconds) * 1000.0
