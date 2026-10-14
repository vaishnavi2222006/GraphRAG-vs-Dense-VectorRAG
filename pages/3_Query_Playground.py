from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.schemas.documents import RetrievalStrategy


st.set_page_config(page_title="Query Playground - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Query Playground")
question = st.text_area("Question", height=120)
strategy = st.selectbox("Retrieval strategy", ["AUTO", "VECTOR", "GRAPH", "HYBRID"])
top_k = st.number_input("Top K", min_value=1, max_value=20, value=context.settings.default_top_k)

if st.button("Run query") and question.strip():
    manual = None if strategy == "AUTO" else RetrievalStrategy[strategy]
    result = context.query_service.answer(question, manual_strategy=manual, top_k=int(top_k))
    st.subheader("Answer")
    st.write(result.answer)
    st.write(f"Strategy: {result.selected_strategy.value} | Routing: {result.routing_rationale}")
    st.write(f"Retrieval latency: {result.retrieval_latency_ms:.2f} ms | Generation latency: {result.generation_latency_ms:.2f} ms | Total: {result.total_latency_ms:.2f} ms")
    st.subheader("Citations")
    st.dataframe(pd.DataFrame([citation.__dict__ for citation in result.citations]), use_container_width=True)
    st.subheader("Retrieved Chunks")
    st.dataframe(pd.DataFrame([chunk.__dict__ for chunk in result.retrieved_sources]), use_container_width=True)
    if result.graph_evidence:
        st.subheader("Graph Evidence")
        st.write(f"Entities: {len(result.graph_evidence.entities)} | Relationships: {len(result.graph_evidence.relationships)}")
