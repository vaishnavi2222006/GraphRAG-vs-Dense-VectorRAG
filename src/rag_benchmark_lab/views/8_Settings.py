from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.ui import inject_custom_css, render_dev_mode_banner, render_lab_header

# page config handled in main app
inject_custom_css()
context = get_context()

render_lab_header(
    title="System Configuration & Diagnostics",
    subtitle="Configure database endpoints, retrieval hyperparameters, LLM inference backends, and monitor service health.",
    badge_text="SYSTEM ADMIN",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

settings = context.settings

# Section 1: Live Service Health Diagnostics
st.markdown("### 🔌 Active Service Health & Connectivity")
health_items = context.health_service.check()

cols = st.columns(len(health_items))
for idx, item in enumerate(health_items):
    with cols[idx]:
        if item.healthy:
            border_col = "#34D399"
            bg_col = "rgba(52, 211, 153, 0.08)"
            badge_txt = "ONLINE & HEALTHY"
            badge_col = "#34D399"
            icon_char = "✅"
        elif "mock" in item.name.lower() or "dev" in item.name.lower():
            border_col = "#FBBF24"
            bg_col = "rgba(251, 191, 36, 0.08)"
            badge_txt = "LOCAL DEV SYNTHESIS"
            badge_col = "#FCD34D"
            icon_char = "⚡"
        else:
            border_col = "#FB7185"
            bg_col = "rgba(244, 63, 94, 0.08)"
            badge_txt = "DISCONNECTED"
            badge_col = "#FB7185"
            icon_char = "❌"

        st.markdown(
            f"""
            <div style="background: {bg_col}; border: 1px solid rgba(148, 163, 184, 0.16); border-top: 4px solid {border_col}; border-radius: 12px; padding: 18px 16px; text-align: center; height: 100%; box-shadow: 0 4px 16px rgba(0,0,0,0.3);">
                <div style="font-size: 26px; margin-bottom: 6px;">{icon_char}</div>
                <div style="font-weight: 700; color: #F1F5F9; font-size: 14px; letter-spacing: -0.01em;">{item.name}</div>
                <div style="font-size: 11px; font-weight: 700; color: {badge_col}; margin-top: 6px; letter-spacing: 0.05em;">{badge_txt}</div>
                <div style="font-size: 11.5px; color: #94A3B8; margin-top: 8px; line-height: 1.4;">{item.details or ''}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

st.markdown("---")

# LLM Provider Selection & Status
st.markdown("### 🤖 Answer-Generation LLM Provider")
prov_col1, prov_col2 = st.columns([1, 1])

current_prov = st.session_state.get("override_llm_provider", settings.llm_provider)
with prov_col1:
    prov_options = ["groq", "gemini", "ollama"]
    prov_idx = prov_options.index(current_prov) if current_prov in prov_options else 0
    chosen_prov = st.radio(
        "Active LLM Backend",
        options=prov_options,
        index=prov_idx,
        format_func=lambda x: "Groq API (High-Speed Cloud)" if x == "groq" else ("Google Gemini (Cloud API)" if x == "gemini" else "Ollama (Local Inference)"),
        key="prov_radio",
    )
    if chosen_prov != current_prov:
        import os
        st.session_state["override_llm_provider"] = chosen_prov
        os.environ["LLM_PROVIDER"] = chosen_prov
        st.cache_resource.clear()
        st.rerun()

with prov_col2:
    import os
    if chosen_prov == "groq":
        has_key = bool(settings.groq_api_key or os.getenv("GROQ_API_KEY"))
        if has_key:
            st.success(f"✅ Groq is active. Model: `{settings.groq_model}`. API key is configured.")
        else:
            st.warning("⚠️ `GROQ_API_KEY` is not detected in environment.")
            key_input = st.text_input("Enter Groq API Key for current session:", type="password", key="session_groq_key")
            if st.button("Save Groq API Key to Session"):
                if key_input.strip():
                    os.environ["GROQ_API_KEY"] = key_input.strip()
                    st.cache_resource.clear()
                    st.success("Groq API key saved. Reloading...")
                    st.rerun()
    elif chosen_prov == "gemini":
        has_key = bool(settings.gemini_api_key or os.getenv("GEMINI_API_KEY"))
        if has_key:
            st.success(f"✅ Gemini is active. Model: `{settings.gemini_model}`. API key is configured.")
        else:
            st.warning("⚠️ `GEMINI_API_KEY` is not detected in environment.")
            key_input = st.text_input("Enter Gemini API Key for current session:", type="password", key="session_gemini_key")
            if st.button("Save Gemini API Key to Session"):
                if key_input.strip():
                    os.environ["GEMINI_API_KEY"] = key_input.strip()
                    st.cache_resource.clear()
                    st.success("API key saved. Reloading...")
                    st.rerun()
    else:
        st.info(f"Ollama is configured at `{settings.ollama_base_url}` with model `{settings.ollama_model}`.")

st.markdown("---")

# Section 2: Structured Parameter Groups
st.markdown("### ⚙️ Research Pipeline Hyperparameters")

group_tab1, group_tab2, group_tab3, group_tab4 = st.tabs([
    "Database & Storage",
    "Retrieval & RRF Fusion",
    "Inference & LLM",
    "Chunking & Preprocessing",
])

with group_tab1:
    st.markdown("#### Database Adapters & Persistent Stores")
    db_table = [
        {"Parameter": "Qdrant Vector DB URL", "Value": settings.qdrant_url, "Description": "HTTP endpoint for vector indexing and cosine search"},
        {"Parameter": "Qdrant Collection", "Value": settings.qdrant_collection, "Description": "Vector collection namespace holding embedded chunks"},
        {"Parameter": "Neo4j Knowledge Graph URI", "Value": settings.neo4j_uri, "Description": "Bolt connection string for graph database"},
        {"Parameter": "Neo4j Username", "Value": settings.neo4j_username, "Description": "Graph database authentication user"},
        {"Parameter": "SQLite Metadata Path", "Value": str(settings.sqlite_path), "Description": "Durable registry for document metadata and benchmark runs"},
    ]
    st.dataframe(pd.DataFrame(db_table), use_container_width=True)

with group_tab2:
    st.markdown("#### Retrieval & Multi-Hop Fusion Hyperparameters")
    ret_table = [
        {"Parameter": "Embedding Model", "Value": settings.embedding_model, "Description": "Sentence Transformers model for text representation"},
        {"Parameter": "Default Top-K", "Value": str(settings.default_top_k), "Description": "Number of top ranked passages retrieved for generation"},
        {"Parameter": "Graph Max Hops", "Value": str(settings.graph_max_hops), "Description": "Maximum multi-hop traversal radius in Cypher queries"},
        {"Parameter": "Graph Max Results", "Value": str(settings.graph_max_results), "Description": "Maximum entity candidates expanded per query"},
        {"Parameter": "RRF Constant (k)", "Value": str(settings.rrf_k), "Description": "Reciprocal Rank Fusion smoothing constant in score equation 1/(k + rank)"},
    ]
    st.dataframe(pd.DataFrame(ret_table), use_container_width=True)

with group_tab3:
    st.markdown("#### LLM Generation & Evaluation Engine")
    groq_key_display = "Configured (***)" if settings.groq_api_key else "Not Set (Set GROQ_API_KEY)"
    gemini_key_display = "Configured (***)" if settings.gemini_api_key else "Not Set (Set GEMINI_API_KEY)"
    llm_table = [
        {"Parameter": "Active LLM Provider", "Value": settings.llm_provider.upper(), "Description": "Selected answer generation provider (Groq, Gemini, or Ollama)"},
        {"Parameter": "Groq LLM Model", "Value": settings.groq_model, "Description": "High-speed cloud model used when Groq provider is selected"},
        {"Parameter": "Groq API Key", "Value": groq_key_display, "Description": "Status of GROQ_API_KEY environment variable (masked)"},
        {"Parameter": "Groq Timeout", "Value": f"{settings.groq_timeout_seconds} seconds", "Description": "Socket timeout for Groq API calls"},
        {"Parameter": "Google Gemini Model", "Value": settings.gemini_model, "Description": "Cloud model used when Gemini provider is selected"},
        {"Parameter": "Gemini API Key", "Value": gemini_key_display, "Description": "Status of GEMINI_API_KEY environment variable (masked)"},
        {"Parameter": "Gemini Timeout", "Value": f"{settings.gemini_timeout_seconds} seconds", "Description": "Socket timeout for Gemini API calls"},
        {"Parameter": "Ollama Base URL", "Value": settings.ollama_base_url, "Description": "Local inference REST API server"},
        {"Parameter": "Ollama LLM Model", "Value": settings.ollama_model, "Description": "Local foundational model used when Ollama provider is selected"},
        {"Parameter": "Ollama Timeout", "Value": f"{settings.ollama_timeout_seconds} seconds", "Description": "HTTP socket timeout for Ollama generation requests"},
        {"Parameter": "Answer Temperature", "Value": str(settings.answer_temperature), "Description": "Sampling temperature (0.0 for deterministic factual answers)"},
        {"Parameter": "Mock Mode Active", "Value": str(settings.llm_mock_mode), "Description": "Whether offline development synthesis is enabled for Ollama fallback"},
    ]
    st.dataframe(pd.DataFrame(llm_table), use_container_width=True)

with group_tab4:
    st.markdown("#### Ingestion & Text Preprocessing")
    ing_table = [
        {"Parameter": "Chunk Size", "Value": f"{settings.chunk_size} characters", "Description": "Sliding window token/character budget per text chunk"},
        {"Parameter": "Chunk Overlap", "Value": f"{settings.chunk_overlap} characters", "Description": "Boundary overlap between consecutive chunks to preserve context"},
        {"Parameter": "Max Upload Size", "Value": f"{settings.upload_size_limit_mb} MB", "Description": "Hard upload limit for user uploaded PDF, TXT, DOCX files"},
        {"Parameter": "Upload Directory", "Value": str(settings.upload_dir), "Description": "Local disk storage path for uploaded raw files"},
    ]
    st.dataframe(pd.DataFrame(ing_table), use_container_width=True)

st.markdown("---")

# Expandable Raw JSON for Developer Debugging
with st.expander("🛠️ Developer Raw Configuration JSON (Debug Inspection)"):
    st.json(settings.to_dict())
