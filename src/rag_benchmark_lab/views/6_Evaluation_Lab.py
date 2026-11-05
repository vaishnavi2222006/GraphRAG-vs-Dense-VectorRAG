from __future__ import annotations

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.evaluation.benchmark_runner import BenchmarkRunner
from rag_benchmark_lab.evaluation.dataset import load_dataset, sample_dataset
from rag_benchmark_lab.schemas.documents import RetrievalStrategy
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_dev_mode_banner,
    render_lab_header,
    render_strategy_badge,
)

# page config handled in main app
inject_custom_css()
user_id = require_auth()
context = get_context()
user_name = get_current_user_name() or "User"

# Sidebar: KB selector and logout
with st.sidebar:
    st.markdown(f"**👤 {user_name}**")
    kbs = context.user_registry.list_knowledge_bases(user_id)
    active_kb_id = st.session_state.get("active_kb_id")
    if kbs:
        kb_options = {kb.kb_id: kb.name for kb in kbs}
        if active_kb_id not in kb_options:
            active_kb_id = list(kb_options.keys())[0]
            st.session_state["active_kb_id"] = active_kb_id
        selected_kb_id = st.selectbox(
            "📂 Knowledge Base",
            options=list(kb_options.keys()),
            format_func=lambda x: kb_options[x],
            index=list(kb_options.keys()).index(active_kb_id),
            key="eval_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("No knowledge bases yet.")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, key="eval_logout"):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="Benchmark Evaluation Lab",
    subtitle="Execute rigorous empirical benchmark experiments across Vector RAG, GraphRAG, and Hybrid RAG.",
    badge_text="EMPIRICAL BENCHMARK",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

# Step 1: Benchmark Dataset Selection
st.markdown("### 1. Dataset Selection & Validation")
ds_col1, ds_col2 = st.columns([1, 2])

with ds_col1:
    dataset_source = st.radio("Dataset Source", ["Built-in Sample Dataset", "Upload Custom JSON/JSONL"], index=0)

examples = []
if dataset_source == "Built-in Sample Dataset":
    examples = sample_dataset()
    st.success(f"Loaded built-in benchmark dataset with {len(examples)} validated query examples.")
else:
    uploaded = st.file_uploader("Upload JSON or JSONL benchmark dataset", type=["json", "jsonl"])
    if uploaded:
        temp_dir = context.settings.app_data_dir
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_path = temp_dir / uploaded.name
        temp_path.write_bytes(uploaded.getvalue())
        try:
            examples = load_dataset(temp_path)
            st.success(f"Uploaded and parsed {len(examples)} examples from `{uploaded.name}`.")
        except Exception as exc:
            st.error(f"Failed to validate uploaded dataset: {exc}")
    else:
        st.info("Upload a JSON or JSONL dataset to evaluate your custom queries.")

if examples:
    with st.expander(f"📋 Preview Benchmark Dataset ({len(examples)} queries)", expanded=False):
        df_prev = pd.DataFrame(
            [
                {
                    "Query ID": ex.query_id,
                    "Question": ex.question,
                    "Expected Answer": ex.expected_answer or "N/A",
                    "Query Type": ex.metadata.get("query_type", "factual"),
                    "Difficulty": ex.metadata.get("difficulty", "medium"),
                    "Hops": ex.metadata.get("expected_hops", 1),
                }
                for ex in examples
            ]
        )
        st.dataframe(df_prev, use_container_width=True)

st.markdown("---")

# Step 2: Experiment Configuration
st.markdown("### 2. Experiment Configuration")
cfg_col1, cfg_col2, cfg_col3 = st.columns(3)

with cfg_col1:
    selected_strategies = st.multiselect(
        "Retrieval Architectures to Benchmark",
        options=["VECTOR", "GRAPH", "HYBRID"],
        default=["VECTOR", "GRAPH", "HYBRID"],
        format_func=lambda s: f"{s} RAG",
        help="Select which retrieval pipelines will execute against the benchmark queries.",
    )

with cfg_col2:
    query_limit = st.slider("Max Queries to Run", min_value=1, max_value=max(1, len(examples)), value=min(5, len(examples)))

with cfg_col3:
    if context.settings.llm_provider == "groq":
        llm_engine_label = f"Groq ({context.settings.groq_model})"
    elif context.settings.llm_provider == "gemini":
        llm_engine_label = f"Google Gemini ({context.settings.gemini_model})"
    elif context.settings.llm_mock_mode:
        llm_engine_label = "Mock / Dev Synthesis"
    else:
        llm_engine_label = f"Ollama ({context.settings.ollama_model})"
    st.caption(f"• LLM Engine: **{llm_engine_label}**")
    st.caption("• Vector DB: **Qdrant (Live)**")
    st.caption("• Graph DB: **Neo4j (Live)**")

# Fetch document IDs for active corpus
kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb_id, user_id) if active_kb_id else []
is_corpus_empty = not kb_doc_ids

if is_corpus_empty:
    st.warning("⚠️ The selected Knowledge Base has no documents. Empty corpora cannot be evaluated.")

# Step 3: Run Benchmark
start_btn = st.button(
    "🧪 Launch Empirical Benchmark Run",
    type="primary",
    disabled=not examples or not selected_strategies or is_corpus_empty,
)

if start_btn:
    if is_corpus_empty:
        st.error("Cannot evaluate an empty knowledge base. Please upload documents first.")
        st.stop()

    active_examples = examples[:query_limit]
    active_strat_enums = [RetrievalStrategy[s] for s in selected_strategies]

    st.markdown("### 🚀 Benchmark Execution Progress")
    progress_bar = st.progress(0)
    status_text = st.empty()

    runner = BenchmarkRunner(context.query_service, context.result_store)

    with st.spinner("Executing benchmark across strategies..."):
        run_output = runner.run(
            active_examples,
            strategies=active_strat_enums,
            document_ids=kb_doc_ids,
        )

    progress_bar.progress(1.0)
    status_text.success(f"✅ Benchmark Run `{run_output['run_id'][:12]}...` completed and persisted!")

    st.markdown("---")

    # Display Aggregates
    st.markdown("### 📊 Benchmark Run Aggregates")
    aggs = run_output.get("aggregates", {})
    agg_table = []
    for strat, data in aggs.items():
        lat = data.get("latency", {})
        mets = data.get("metrics", {})
        agg_table.append(
            {
                "Strategy": strat,
                "Queries Evaluated": data.get("queries", 0),
                "Mean Latency (ms)": round(lat.get("mean") or 0, 1),
                "Median Latency (ms)": round(lat.get("median") or 0, 1),
                "Faithfulness": mets.get("faithfulness", "N/A"),
                "Answer Relevancy": mets.get("answer_relevancy", "N/A"),
                "Recall@K": mets.get("retrieval_recall_at_k", "N/A"),
                "Hit Rate@K": mets.get("hit_rate_at_k", "N/A"),
                "MRR": mets.get("mrr", "N/A"),
                "Context Precision": mets.get("context_precision", "N/A"),
                "Context Recall": mets.get("context_recall", "N/A"),
            }
        )
    st.dataframe(pd.DataFrame(agg_table), use_container_width=True)

    # Detailed Per-Query Table
    with st.expander("🔍 Detailed Per-Query Results", expanded=True):
        df_rows = []
        for r in run_output.get("rows", []):
            df_rows.append(
                {
                    "Query ID": r["query_id"],
                    "Strategy": r["strategy"],
                    "Question": r["question"][:60] + "...",
                    "Answer": r["answer"][:80] + "...",
                    "Total Latency (ms)": round(r["total_latency_ms"] or 0, 1),
                    "Faithfulness": r["metrics"].get("faithfulness", {}).get("value"),
                    "Answer Relevancy": r["metrics"].get("answer_relevancy", {}).get("value"),
                    "Recall@K": r["metrics"].get("retrieval_recall_at_k", {}).get("value"),
                }
            )
        st.dataframe(pd.DataFrame(df_rows), use_container_width=True)

st.markdown("---")

# Historical Benchmark Runs
st.markdown("### 📜 Persisted Experiment History")
runs = context.result_store.list_runs()
if runs:
    df_runs = []
    for run in runs:
        df_runs.append(
            {
                "Run ID": run["run_id"][:12] + "...",
                "Timestamp": run.get("timestamp", "")[:19].replace("T", " "),
                "Dataset": run.get("dataset_id", "default"),
                "Total Queries": run.get("total_queries", 0),
                "Strategies": ", ".join(run.get("strategies", [])),
                "Mode": "MOCK / DEV" if run.get("mock_mode") else "REAL LLM",
            }
        )
    st.dataframe(pd.DataFrame(df_runs), use_container_width=True)
else:
    st.info("No benchmark runs recorded in SQLite history yet.")
