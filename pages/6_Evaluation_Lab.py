from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import load_dataset, sample_dataset
from rag_benchmark_lab.evaluation.result_store import ResultStore


st.set_page_config(page_title="Evaluation Lab - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Evaluation Lab")
sample_choice = st.selectbox("Dataset source", ["Built-in sample dataset", "Upload JSON/JSONL"])
dataset_examples = sample_dataset()

uploaded = None
if sample_choice == "Upload JSON/JSONL":
    uploaded = st.file_uploader("Upload evaluation dataset", type=["json", "jsonl"])
    if uploaded:
        temp_path = context.settings.app_data_dir / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        dataset_examples = load_dataset(temp_path)

st.write(f"Loaded {len(dataset_examples)} examples")
if st.button("Run benchmark"):
    runner = BenchmarkRunner(context.query_service, context.result_store)
    result = runner.run(dataset_examples)
    st.success(f"Benchmark run saved: {result['run_id']}")
    st.dataframe(pd.DataFrame(result["rows"]), use_container_width=True)
    st.json(result["aggregates"])

st.subheader("Stored Runs")
st.dataframe(pd.DataFrame(context.result_store.list_runs()), use_container_width=True)
