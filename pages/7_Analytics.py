from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Analytics - RAG Benchmark Lab", layout="wide")
context = get_context()
results = context.result_store.list_results()

st.title("Analytics")
if not results:
    st.info("No benchmark results available yet.")
else:
    frame = pd.DataFrame(results)
    if "total_latency_ms" in frame:
        st.plotly_chart(px.box(frame, x="strategy", y="total_latency_ms", title="Latency by Strategy"), use_container_width=True)
    if "metrics" in frame:
        st.write("Metric data stored per query; expand a run to inspect detailed metric payloads.")
    st.dataframe(frame, use_container_width=True)
