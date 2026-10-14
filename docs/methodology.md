# Experimental Methodology

This project compares Dense Vector RAG, GraphRAG, and Hybrid RAG over the same corpus.

## Procedure

1. Ingest documents and preserve document/chunk provenance.
2. Generate dense embeddings and store chunks in Qdrant.
3. Extract entities and relationships from chunks and store them in Neo4j.
4. Run each evaluation query under the chosen strategy or routing mode.
5. Persist answers, retrieval evidence, latencies, and metric outputs.
6. Repeat runs with the same configuration for reproducibility.

## Controls

- Same corpus for all strategies.
- Same chunking settings.
- Same answer generation model and prompt instructions where possible.
- Fixed evaluation dataset during comparison runs.
- Strategy choice recorded explicitly for every query.

## Interpretation

Results should be compared using both quality metrics and operational metrics.
GraphRAG may improve relationship-heavy queries, while Vector RAG can remain superior for direct factual lookup. Hybrid RAG should be judged on its balance between evidence coverage and latency.
