from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Corpus Management - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Corpus Management")
uploaded_files = st.file_uploader("Upload PDF, TXT, or DOCX files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if uploaded_files and st.button("Ingest uploaded files"):
    for uploaded in uploaded_files:
        outcome = context.document_service.ingest_bytes(uploaded.name, uploaded.getvalue())
        st.write(f"{uploaded.name}: {outcome.document.status.value}")
        if outcome.errors:
            st.warning("; ".join(outcome.errors))

documents = context.registry.list_documents()
if documents:
    st.dataframe(pd.DataFrame(documents), use_container_width=True)
else:
    st.info("No documents in the corpus yet.")

selected_document_id = st.selectbox("Select a document for actions", [""] + [document["document_id"] for document in documents])
action_cols = st.columns(3)
with action_cols[0]:
    if st.button("Re-index selected") and selected_document_id:
        record = next(document for document in documents if document["document_id"] == selected_document_id)
        source_path = record.get("source_path")
        if source_path and Path(source_path).exists():
            outcome = context.document_service.ingest_path(source_path)
            st.success(f"Re-indexed {outcome.document.document_name}")
        else:
            st.error("Source file is not available for re-indexing.")
with action_cols[1]:
    if st.button("Delete selected") and selected_document_id:
        warnings = context.document_service.delete_document(selected_document_id)
        st.warning("; ".join(warnings) if warnings else "Document deleted.")
with action_cols[2]:
    if st.button("Clear corpus"):
        context.document_service.clear_corpus()
        st.success("Corpus cleared.")

if selected_document_id:
    st.subheader("Document Details")
    st.write(next(document for document in documents if document["document_id"] == selected_document_id))
    st.subheader("Chunks")
    chunks = context.registry.list_chunks(selected_document_id)
    if chunks:
        st.dataframe(pd.DataFrame(chunks), use_container_width=True)
    else:
        st.info("No chunks stored for this document.")
