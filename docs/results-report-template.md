# Results Report Template

## Run Metadata

- Run ID: {{run_id}}
- Dataset ID: {{dataset_id}}
- Dataset Hash: {{dataset_hash}}
- Timestamp: {{timestamp}}
- Model: {{model_name}}

## Aggregate Results

{{aggregate_results_table}}

## Per-Query Results

{{per_query_results_table}}

## Observations

- Notes on query categories where GraphRAG outperformed Vector RAG
- Notes on latency and retrieval overhead
- Notes on partial failures or unavailable services

## Limitations

- Synthetic data should not be treated as a substitute for a real domain corpus.
- Mock generation mode is for functional testing only.
