from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context, render_overview_page

st.set_page_config(page_title="Research Overview - RAG Benchmark Lab", page_icon="🧪", layout="wide")
context = get_context()
render_overview_page(context)
