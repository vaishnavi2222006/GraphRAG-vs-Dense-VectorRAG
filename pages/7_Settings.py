"""RAG Benchmark Lab — Settings & System Diagnostics Page."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import require_auth
from rag_benchmark_lab.utils.ui import (
    get_current_theme,
    inject_custom_css,
    render_html,
    render_topnav,
    toggle_theme,
)

st.set_page_config(
    page_title="Settings — RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Settings")

user_id = require_auth()
context = get_context()
settings = context.settings
current_theme = get_current_theme()

# ── Header ────────────────────────────────────────────────────────────────────
render_html(
    """<div style="margin-top:20px;margin-bottom:18px;">
<div style="display:inline-flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:var(--accent);text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  System Telemetry &amp; Config
</div>
<h1 style="font-size:28px;font-weight:800;letter-spacing:-0.03em;color:var(--text);margin:0 0 4px 0;">
  Settings &amp; Diagnostics
</h1>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Inspect LLM inference providers, database adapters, retrieval parameters, theme settings, and live system health.
</p>
</div>"""
)

# ── Service Health ────────────────────────────────────────────────────────────
render_html(
    """<div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:12px;">
      Live Component Health Diagnostics
    </div>"""
)
health_items = context.health_service.check(check_remote_llm=False)

h_cols = st.columns(len(health_items))
for idx, item in enumerate(health_items):
    with h_cols[idx]:
        if item.healthy:
            border_col = "var(--success)"
            bg_col = "var(--success-bg)"
            badge_txt = "ONLINE & HEALTHY"
            badge_col = "var(--success)"
            icon_char = "✓"
        elif "mock" in item.name.lower() or "dev" in item.name.lower():
            border_col = "var(--warning)"
            bg_col = "var(--warning-bg)"
            badge_txt = "DEV MODE"
            badge_col = "var(--warning)"
            icon_char = "⚡"
        else:
            border_col = "var(--error)"
            bg_col = "var(--error-bg)"
            badge_txt = "ATTENTION"
            badge_col = "var(--error)"
            icon_char = "!"

        render_html(
            f"""
            <div class="rag-card" style="border-top: 3px solid {border_col}; padding: 16px 14px; text-align: center; height: 100%;">
                <div style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;border-radius:8px;background:{bg_col};color:{badge_col};font-weight:800;font-size:16px;margin-bottom:8px;">
                  {icon_char}
                </div>
                <div style="font-weight: 700; color: var(--text); font-size: 13.5px;">{item.name}</div>
                <div style="font-size: 10.5px; font-weight: 700; color: {badge_col}; margin-top: 4px; letter-spacing: 0.04em;">{badge_txt}</div>
                <div style="font-size: 11px; color: var(--text-muted); margin-top: 6px; line-height: 1.4;">{item.details or ''}</div>
            </div>
            """
        )

st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)

# ── Appearance & Theme ────────────────────────────────────────────────────────
render_html(
    """<div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:12px;">
      Interface Appearance &amp; Accessibility
    </div>"""
)
thm_c1, thm_c2 = st.columns([2, 1], vertical_alignment="center")
with thm_c1:
    render_html(
        f"""
        <div class="rag-card" style="padding:16px 20px;">
          <div style="display:flex;align-items:center;gap:12px;">
            <div style="font-size:20px;">{"🌙" if current_theme == "dark" else "☀️"}</div>
            <div>
              <div style="font-size:14px;font-weight:700;color:var(--text);">Current Theme: <span style="color:var(--accent);text-transform:capitalize;">{current_theme} Mode</span></div>
              <div style="font-size:12px;color:var(--text-muted);margin-top:2px;">Persistent across browser reloads, page navigation, and reruns.</div>
            </div>
          </div>
        </div>
        """
    )
with thm_c2:
    btn_lbl = "Switch to Light Mode" if current_theme == "dark" else "Switch to Dark Mode"
    btn_icon = ":material/light_mode:" if current_theme == "dark" else ":material/dark_mode:"
    if st.button(btn_lbl, icon=btn_icon, use_container_width=True, key="settings_theme_toggle"):
        toggle_theme()

st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)

# ── LLM Provider Configuration ────────────────────────────────────────────────
render_html(
    """<div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:12px;">
      LLM Inference Provider
    </div>"""
)
p_col1, p_col2 = st.columns([1.2, 1.8])

with p_col1:
    with st.container(border=True):
        st.markdown("**Configured Provider**")
        st.caption("Google Gemini is the default provider. Alternative providers like local Ollama are supported via configuration.")

        curr_provider = settings.llm_provider
        st.markdown(f"Active Provider: **{curr_provider.upper()}**")

        if curr_provider == "gemini":
            is_configured = hasattr(context.llm_service, "is_configured") and context.llm_service.is_configured()
            if is_configured:
                st.success(f"Google Gemini configured (Model: `{settings.gemini_model}`)")
            else:
                st.error("Gemini API key is not configured. Please update `GEMINI_API_KEY` in your `.env` file and restart the application.")
        else:
            st.info(f"Ollama configured at `{settings.ollama_base_url}` with model `{settings.ollama_model}`.")

with p_col2:
    with st.container(border=True):
        st.markdown("**How to Update Gemini API Key**")
        st.markdown(
            """
1. Open `.env` in the project root:
   ```bash
   nano ~/Downloads/GraphRAG\\ vs.\\ Dense\\ Vector\\ RAG/.env
   ```
2. Update the `GEMINI_API_KEY` entry:
   ```bash
   GEMINI_API_KEY=your_new_key_here
   ```
3. Restart Streamlit:
   ```bash
   .venv/bin/streamlit run app.py
   ```
4. **Note on Quota**: Keys belonging to the same Google Cloud / AI Studio project generally share the project's quota. If daily quota was exhausted, creating a new key in the same project will not reset the limit.
            """
        )

st.markdown("<hr style='border-color:var(--border);margin:24px 0;'/>", unsafe_allow_html=True)

# ── Hyperparameters Tables ───────────────────────────────────────────────────
render_html(
    """<div style="font-size:18px;font-weight:800;color:var(--text);margin-bottom:12px;">
      Pipeline Hyperparameters &amp; Storage Endpoints
    </div>"""
)

tab_ret, tab_db, tab_llm = st.tabs(["Retrieval & Fusion", "Database & Indexing", "LLM Parameters"])

with tab_ret:
    ret_table = [
        {"Parameter": "Embedding Model", "Value": settings.embedding_model, "Description": "Sentence Transformers model for vector representations"},
        {"Parameter": "Default Top-K", "Value": str(settings.default_top_k), "Description": "Number of top-ranked evidence chunks retrieved"},
        {"Parameter": "Graph Max Hops", "Value": str(settings.graph_max_hops), "Description": "Maximum relational radius traversed in graph store"},
        {"Parameter": "Graph Max Results", "Value": str(settings.graph_max_results), "Description": "Maximum entity candidates expanded per query"},
        {"Parameter": "RRF Smoothing Constant (k)", "Value": str(settings.rrf_k), "Description": "Reciprocal Rank Fusion constant in score formula 1/(k + rank)"},
    ]
    st.dataframe(pd.DataFrame(ret_table), use_container_width=True, hide_index=True)

with tab_db:
    db_table = [
        {"Parameter": "Qdrant URL", "Value": settings.qdrant_url, "Description": "Vector database endpoint (fallback to in-memory if offline)"},
        {"Parameter": "Qdrant Collection", "Value": settings.qdrant_collection, "Description": "Vector collection namespace"},
        {"Parameter": "Neo4j URI", "Value": settings.neo4j_uri, "Description": "Graph store endpoint (fallback to NetworkX if offline)"},
        {"Parameter": "Neo4j Username", "Value": settings.neo4j_username, "Description": "Graph store username"},
        {"Parameter": "SQLite Path", "Value": str(settings.sqlite_path), "Description": "Persistent SQLite database for document and benchmark records"},
    ]
    st.dataframe(pd.DataFrame(db_table), use_container_width=True, hide_index=True)

with tab_llm:
    llm_key_status = "Configured (***)" if (hasattr(context.llm_service, "is_configured") and context.llm_service.is_configured()) else "Not Configured"
    llm_table = [
        {"Parameter": "Default LLM Provider", "Value": settings.llm_provider.upper(), "Description": "Active answer generation backend"},
        {"Parameter": "Gemini Model", "Value": settings.gemini_model, "Description": "Google Gemini model name"},
        {"Parameter": "Gemini API Key", "Value": llm_key_status, "Description": "API key configuration status (secrets masked)"},
        {"Parameter": "Gemini Timeout (s)", "Value": str(settings.gemini_timeout_seconds), "Description": "Timeout for Gemini API requests"},
        {"Parameter": "Ollama Base URL", "Value": settings.ollama_base_url, "Description": "Alternative local inference endpoint"},
        {"Parameter": "Ollama Model", "Value": settings.ollama_model, "Description": "Alternative local LLM model name"},
        {"Parameter": "Answer Temperature", "Value": str(settings.answer_temperature), "Description": "Sampling temperature for answer generation"},
    ]
    st.dataframe(pd.DataFrame(llm_table), use_container_width=True, hide_index=True)
