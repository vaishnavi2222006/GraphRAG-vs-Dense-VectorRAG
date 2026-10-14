# Implementation Status

Current phase: foundation, ingestion, retrieval, evaluation, dashboard scaffold, and validation completed.

Completed:
- Core settings and environment example
- Document ingestion pipeline with PDF/TXT/DOCX support
- Sentence Transformer embedding service with mock fallback
- Qdrant integration wrapper
- Neo4j graph storage wrapper and graph retrieval
- Dense, graph, and hybrid retrieval services
- Query routing and answer generation
- Durable SQLite registry and evaluation result storage
- Streamlit dashboard pages
- Dependency installation in a lightweight local environment
- Unit and evaluation tests passing
- Integration tests skipping cleanly when Qdrant and Neo4j are unavailable
- Streamlit app startup verification in headless mode

Pending:
- Docker Compose validation
- Sample ingestion and benchmark execution against live Qdrant and Neo4j
- Optional Sentence Transformers installation via `pip install -e .[ml]`

Notes:
- Ollama is not installed in the current environment, so the default mock LLM mode remains enabled until a local model is configured.
- Qdrant and Neo4j containers are not running yet.
