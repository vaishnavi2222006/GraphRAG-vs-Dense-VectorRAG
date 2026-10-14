from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Comparison - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Side-by-Side Comparison")
question = st.text_area("Question", height=120)

if st.button("Compare strategies") and question.strip():
    results = context.query_service.compare(question)
    columns = st.columns(3)
    for column, strategy in zip(columns, ["VECTOR", "GRAPH", "HYBRID"], strict=True):
        with column:
            result = results[strategy]
            st.subheader(strategy)
            st.write(result.answer)
            st.write(f"Total latency: {result.total_latency_ms:.2f} ms")
            st.write(f"Retrieved chunks: {len(result.retrieved_sources)}")
