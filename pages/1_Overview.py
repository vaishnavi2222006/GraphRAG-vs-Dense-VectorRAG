from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Overview - RAG Benchmark Lab", layout="wide")
context = get_context()
counts = context.registry.counts()
health = context.health_service.check()

st.title("Overview")
st.caption("Current corpus, indexing, and service status.")

cols = st.columns(5)
metrics = [
    ("Documents", counts["documents"]),
    ("Chunks", counts["chunks"]),
    ("Entities", counts["entities"]),
    ("Relationships", counts["relationships"]),
    ("Evaluation Runs", counts["evaluation_runs"]),
]
for column, (label, value) in zip(cols, metrics, strict=True):
    with column:
        st.metric(label, value)

st.subheader("Service Health")
for item in health:
    st.write(f"{item.name}: {'healthy' if item.healthy else 'unavailable'}{f' - {item.details}' if item.details else ''}")

st.subheader("Recent Ingestion Errors")
documents = context.registry.list_documents()
errors = [document for document in documents if document.get("error_message")]
if errors:
    st.dataframe(errors, use_container_width=True)
else:
    st.info("No ingestion errors recorded.")

st.subheader("Architecture Summary")
st.markdown(
    """
    - Documents are loaded into a durable SQLite registry.
    - Chunks are embedded and written to Qdrant.
    - Extracted entities and relationships are written to Neo4j.
    - Query routing chooses Vector RAG, GraphRAG, Hybrid RAG, or all three for evaluation.
    - Results and benchmark runs are persisted for later analysis.
    """
)
