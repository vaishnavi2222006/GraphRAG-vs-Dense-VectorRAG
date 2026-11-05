"""Unit tests for the Strategy Comparison Matrix in the Compare tab."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import pandas as pd
import pytest

from rag_benchmark_lab.schemas.comparison import ComparisonStrategyResultRecord
from rag_benchmark_lab.schemas.documents import RetrievalStrategy

# Import build_comparison_matrix_df dynamically from pages/4_Compare.py
PAGE_PATH = Path(__file__).resolve().parent.parent.parent / "pages" / "4_Compare.py"
spec = importlib.util.spec_from_file_location("compare_page", str(PAGE_PATH))
compare_mod = importlib.util.module_from_spec(spec)
# Note: To avoid running Streamlit page logic during test import, extract build_comparison_matrix_df
# We can read the AST or load the function
import ast

def _extract_build_function():
    source = PAGE_PATH.read_text(encoding="utf-8")
    parsed = ast.parse(source)
    # Find helper functions and build_comparison_matrix_df
    target_nodes = []
    for node in parsed.body:
        if isinstance(node, ast.FunctionDef) and node.name in (
            "_has_quota_err",
            "_has_auth_err",
            "build_comparison_matrix_df",
        ):
            target_nodes.append(node)
    
    code = compile(ast.Module(body=target_nodes, type_ignores=[]), "<compare_page_fns>", "exec")
    from rag_benchmark_lab.utils.ui import format_latency
    from rag_benchmark_lab.evaluation.metrics import faithfulness
    scope = {
        "RetrievalStrategy": RetrievalStrategy,
        "ComparisonStrategyResultRecord": ComparisonStrategyResultRecord,
        "format_latency": format_latency,
        "faithfulness": faithfulness,
        "pd": pd,
    }
    exec(code, scope)
    return scope["build_comparison_matrix_df"]

build_comparison_matrix_df = _extract_build_function()


def test_build_comparison_matrix_df_populated():
    """Verify that build_comparison_matrix_df generates the 4-column matrix with all required parameters."""
    vector_record = ComparisonStrategyResultRecord(
        result_id="res-v1",
        run_id="cmp-1",
        strategy=RetrievalStrategy.VECTOR.value,
        strategy_label="Vector RAG",
        status="success",
        answer="MediCore Hospital has 120 beds.",
        retrieval_latency_ms=120.5,
        generation_latency_ms=850.0,
        total_latency_ms=970.5,
        citations=[{"document_name": "Doc1", "chunk_index": 0}],
        evidence_sources=[{"text": "MediCore Hospital has 120 beds."}],
    )

    graph_record = ComparisonStrategyResultRecord(
        result_id="res-g1",
        run_id="cmp-1",
        strategy=RetrievalStrategy.GRAPH.value,
        strategy_label="GraphRAG",
        status="success",
        answer="MediCore Hospital is located at 18 Lakeview Road.",
        retrieval_latency_ms=95.0,
        generation_latency_ms=910.0,
        total_latency_ms=1005.0,
        citations=[{"document_name": "Doc2", "chunk_index": 1}],
        evidence_sources=[{"text": "MediCore Hospital is located at 18 Lakeview Road."}],
        graph_evidence={
            "relationships": [
                {"source_entity_name": "MediCore", "relationship_type": "LOCATED_AT", "target_entity_name": "Lakeview"}
            ]
        },
    )

    hybrid_record = ComparisonStrategyResultRecord(
        result_id="res-h1",
        run_id="cmp-1",
        strategy=RetrievalStrategy.HYBRID.value,
        strategy_label="Hybrid RAG",
        status="success",
        answer="MediCore Hospital has 120 beds and is located at 18 Lakeview Road.",
        retrieval_latency_ms=150.0,
        generation_latency_ms=800.0,
        total_latency_ms=950.0,
        citations=[{"document_name": "Doc1", "chunk_index": 0}, {"document_name": "Doc2", "chunk_index": 1}],
        evidence_sources=[{"text": "MediCore Hospital has 120 beds."}, {"text": "MediCore Hospital is located at 18 Lakeview Road."}],
        graph_evidence={
            "relationships": [
                {"source_entity_name": "MediCore", "relationship_type": "LOCATED_AT", "target_entity_name": "Lakeview"}
            ]
        },
    )

    strat_map = {
        RetrievalStrategy.VECTOR.value: vector_record,
        RetrievalStrategy.GRAPH.value: graph_record,
        RetrievalStrategy.HYBRID.value: hybrid_record,
    }

    df = build_comparison_matrix_df(strat_map, query_text="What is the hospital address and bed capacity?")

    assert isinstance(df, pd.DataFrame)
    assert list(df.columns) == ["Parameter", "Vector RAG", "GraphRAG", "Hybrid RAG"]

    param_series = df["Parameter"].tolist()
    assert "Retrieval strategy" in param_series
    assert "Retrieval latency" in param_series
    assert "Generation latency" in param_series
    assert "Total latency" in param_series
    assert "Passages retrieved" in param_series
    assert "Citations" in param_series
    assert "Graph paths" in param_series
    assert "Answer quality" in param_series
    assert "Execution status" in param_series

    # Check Graph paths values
    graph_row = df[df["Parameter"] == "Graph paths"].iloc[0]
    assert graph_row["Vector RAG"] == "N/A (Vector)"
    assert graph_row["GraphRAG"] == "1"
    assert graph_row["Hybrid RAG"] == "1"

    # Check Execution status
    stat_row = df[df["Parameter"] == "Execution status"].iloc[0]
    assert stat_row["Vector RAG"] == "Success"
    assert stat_row["GraphRAG"] == "Success"
    assert stat_row["Hybrid RAG"] == "Success"

    # Check Retrieval strategy descriptions
    strat_row = df[df["Parameter"] == "Retrieval strategy"].iloc[0]
    assert "Qdrant" in strat_row["Vector RAG"]
    assert "Neo4j" in strat_row["GraphRAG"]
    assert "RRF" in strat_row["Hybrid RAG"]


def test_build_comparison_matrix_df_missing_graceful():
    """Verify that build_comparison_matrix_df handles missing strategies gracefully without failing."""
    # Empty strat map
    df_empty = build_comparison_matrix_df({})
    assert list(df_empty.columns) == ["Parameter", "Vector RAG", "GraphRAG", "Hybrid RAG"]
    assert len(df_empty) == 9

    # Vector RAG with error
    failed_record = ComparisonStrategyResultRecord(
        result_id="res-fail",
        run_id="cmp-2",
        strategy=RetrievalStrategy.VECTOR.value,
        strategy_label="Vector RAG",
        status="failed",
        answer="",
        errors=["429 resource_exhausted quota exceeded"],
        error_message="quota exceeded",
    )
    df_fail = build_comparison_matrix_df({RetrievalStrategy.VECTOR.value: failed_record})
    stat_row = df_fail[df_fail["Parameter"] == "Execution status"].iloc[0]
    assert stat_row["Vector RAG"] == "Quota Exceeded"
    assert stat_row["GraphRAG"] == "N/A"
