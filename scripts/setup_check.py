from __future__ import annotations

from rag_benchmark_lab.app import build_context


def main() -> None:
    context = build_context()
    print(context.settings.to_dict())
    print(context.registry.counts())
    for health in context.health_service.check():
        print(f"{health.name}: {health.healthy} {health.details or ''}")


if __name__ == "__main__":
    main()
