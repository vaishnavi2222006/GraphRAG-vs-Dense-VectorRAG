from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Settings - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Settings and System Health")
st.json(context.settings.to_dict())
st.subheader("Service Health")
for item in context.health_service.check():
    st.write(f"{item.name}: {'healthy' if item.healthy else 'unavailable'}{f' - {item.details}' if item.details else ''}")
