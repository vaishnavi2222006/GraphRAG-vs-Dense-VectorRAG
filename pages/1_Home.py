"""RAG Benchmark Lab — Home Workspace & Product Overview."""
from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, is_authenticated
from rag_benchmark_lab.utils.ui import (
    inject_custom_css,
    render_html,
    render_topnav,
)

st.set_page_config(
    page_title="RAG Benchmark Lab",
    layout="wide",
    initial_sidebar_state="collapsed",
)

inject_custom_css()
render_topnav(active_page="Home")

context = get_context()
is_auth = is_authenticated()
user_name = get_current_user_name() or "Researcher"
user_id = st.session_state.get("_rag_auth_user_id", "")

user_kbs = []
if is_auth and user_id:
    user_kbs = context.user_registry.list_knowledge_bases(user_id)

# Fetch real backend statistics (Strictly genuine data, never hardcoded)
real_counts = context.health_service.get_real_counts()
all_docs = context.registry.list_documents()
eval_runs = context.result_store.list_runs()

total_documents = len(all_docs)
total_chunks = real_counts.get("chunks", 0)
total_entities = real_counts.get("entities", 0)
total_relationships = real_counts.get("relationships", 0)
total_eval_runs = len(eval_runs)

# ── 1. Hero Section ──────────────────────────────────────────────────────────
HERO_HTML = """<div style="text-align:center;padding:40px 16px 16px;max-width:860px;margin:0 auto;">
<div style="display:inline-flex;align-items:center;gap:8px;background:var(--accent-light);border:1px solid var(--accent-border);border-radius:9999px;padding:4px 14px;margin-bottom:16px;">
  <span style="display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--accent);"></span>
  <span style="font-size:11.5px;font-weight:700;color:var(--accent);letter-spacing:0.04em;text-transform:uppercase;">Empirical AI Research Platform</span>
</div>
<h1 style="font-size:44px;font-weight:800;color:var(--text);margin:0 0 14px 0;letter-spacing:-0.035em;line-height:1.15;">
  RAG Benchmark Lab
</h1>
<p style="font-size:16px;color:var(--text-muted);margin:0 auto 24px;line-height:1.6;max-width:680px;">
  Compare, evaluate, and understand retrieval-augmented generation architectures.
  Benchmark dense vector semantic search, knowledge graph traversal, and hybrid rank fusion on your domain documents.
</p>
</div>"""
render_html(HERO_HTML)

# ── Hero Actions ─────────────────────────────────────────────────────────────
c_sp1, c_b1, c_b2, c_sp2 = st.columns([1.8, 1.4, 1.5, 1.8], vertical_alignment="center")
with c_b1:
    with st.container(key="hero_get_started_wrap"):
        if not is_auth:
            if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
                st.switch_page("pages/0_Login.py")
        elif not user_kbs:
            if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
                st.session_state["show_create_kb"] = True
                st.switch_page("pages/2_Documents.py")
        else:
            if st.button("Get Started", type="primary", use_container_width=True, key="home_hero_get_started"):
                st.switch_page("pages/3_Chat.py")

with c_b2:
    with st.container(key="hero_explore_wrap"):
        if st.button("Explore RAG Methods", use_container_width=True, key="home_hero_explore"):
            st.switch_page("pages/4_Compare.py")

render_html("<div style='height:28px;'></div>")

# ── 2. Live Platform Overview Cards ──────────────────────────────────────────
m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)

with m_col1:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--text);">{total_documents}</div>
          <div class="stat-label">Total Documents</div>
        </div>"""
    )
with m_col2:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--vector);">{total_chunks}</div>
          <div class="stat-label">Total Chunks</div>
        </div>"""
    )
with m_col3:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--graph);">{total_entities}</div>
          <div class="stat-label">Total Entities</div>
        </div>"""
    )
with m_col4:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--hybrid);">{total_relationships}</div>
          <div class="stat-label">Total Relationships</div>
        </div>"""
    )
with m_col5:
    render_html(
        f"""<div class="stat-card">
          <div class="stat-value" style="color:var(--accent);">{total_eval_runs}</div>
          <div class="stat-label">Evaluation Runs</div>
        </div>"""
    )

render_html("<div style='height:36px;'></div>")

# ── 3. Connected Architecture Overview Pipeline ──────────────────────────────
ARCH_WORKFLOW_HTML = """<div class="arch-card-wrap">
<div class="arch-header-wrap">
  <span class="arch-tag">SYSTEM ARCHITECTURE</span>
  <h3 class="arch-title">End-to-End RAG Processing &amp; Evaluation Pipeline</h3>
  <p style="font-size:13px;color:var(--text-muted);margin:6px 0 0 0;">
    From raw unstructured documents to verifiable multi-hop answers and empirical benchmark metrics.
  </p>
</div>

<div class="arch-pipeline">
  <!-- Step 1: Upload -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--accent-light);border:1px solid var(--accent-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
        <polyline points="17 8 12 3 7 8"></polyline>
        <line x1="12" y1="3" x2="12" y2="15"></line>
      </svg>
    </div>
    <div class="arch-node-name">1. Upload</div>
    <div class="arch-node-sub">PDF · DOCX · TXT</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 2: Extraction -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--vector-bg);border:1px solid var(--vector-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--vector)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
        <polyline points="14 2 14 8 20 8"></polyline>
        <line x1="16" y1="13" x2="8" y2="13"></line>
        <line x1="16" y1="17" x2="8" y2="17"></line>
      </svg>
    </div>
    <div class="arch-node-name">2. Extraction</div>
    <div class="arch-node-sub">Clean Text Stream</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 3: Chunking -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--graph-bg);border:1px solid var(--graph-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--graph)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <rect x="3" y="3" width="7" height="7"></rect>
        <rect x="14" y="3" width="7" height="7"></rect>
        <rect x="14" y="14" width="7" height="7"></rect>
        <rect x="3" y="14" width="7" height="7"></rect>
      </svg>
    </div>
    <div class="arch-node-name">3. Chunking</div>
    <div class="arch-node-sub">Semantic Passages</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 4: Embeddings -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--hybrid-bg);border:1px solid var(--hybrid-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--hybrid)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
    </div>
    <div class="arch-node-name">4. Embeddings</div>
    <div class="arch-node-sub">Dense Vectors</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 5: Vector DB / Graph -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--accent-light);border:1px solid var(--accent-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <ellipse cx="12" cy="5" rx="9" ry="3"></ellipse>
        <path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path>
        <path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path>
      </svg>
    </div>
    <div class="arch-node-name">5. Vector / Graph</div>
    <div class="arch-node-sub">Qdrant &amp; Neo4j</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 6: Retrieval -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--vector-bg);border:1px solid var(--vector-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--vector)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="11" cy="11" r="8"></circle>
        <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
      </svg>
    </div>
    <div class="arch-node-name">6. Retrieval</div>
    <div class="arch-node-sub">Vector, Graph, RRF</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 7: LLM Generation -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--graph-bg);border:1px solid var(--graph-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--graph)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
      </svg>
    </div>
    <div class="arch-node-name">7. Generation</div>
    <div class="arch-node-sub">Grounded Synthesis</div>
  </div>

  <div class="arch-arrow">
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"></polyline></svg>
  </div>

  <!-- Step 8: Evaluation -->
  <div class="arch-node">
    <div class="arch-node-icon" style="background:var(--success-bg);border:1px solid var(--success-border);">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--success)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M18 20V10"></path>
        <path d="M12 20V4"></path>
        <path d="M6 20v-6"></path>
      </svg>
    </div>
    <div class="arch-node-name">8. Evaluation</div>
    <div class="arch-node-sub">RAG Triad &amp; Speed</div>
  </div>
</div>
</div>"""
render_html(ARCH_WORKFLOW_HTML)

render_html("<div style='height:36px;'></div>")

# ── 4. Retrieval Architecture Strategies ──────────────────────────────────────
render_html(
    """<div style="margin-bottom:16px;">
<h2 style="font-size:22px;font-weight:800;color:var(--text);margin:0 0 4px 0;letter-spacing:-0.02em;">
  Retrieval Architectures Under Study
</h2>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Three foundational retrieval paradigms benchmarked against domain documents.
</p>
</div>"""
)

col_f1, col_f2, col_f3 = st.columns(3, gap="medium")

with col_f1:
    render_html(
        """<div class="retrieval-card">
<div class="retrieval-card-top">
  <div class="retrieval-card-title-group">
    <div class="retrieval-card-icon" style="background:var(--vector-bg);border:1px solid var(--vector-border);">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--vector)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="2" y1="12" x2="22" y2="12"></line>
        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"></path>
      </svg>
    </div>
    <span class="retrieval-card-title">Vector RAG</span>
  </div>
  <span class="badge badge-vector">Semantic Search</span>
</div>
<p class="retrieval-card-desc">
  Encodes text passages into dense mathematical embedding spaces. Performs cosine nearest-neighbor search via Qdrant to retrieve conceptually similar content with low latency.
</p>
</div>"""
    )
    if st.button("Compare Vector RAG", use_container_width=True, key="feat_btn_vector"):
        st.switch_page("pages/4_Compare.py")

with col_f2:
    render_html(
        """<div class="retrieval-card">
<div class="retrieval-card-top">
  <div class="retrieval-card-title-group">
    <div class="retrieval-card-icon" style="background:var(--graph-bg);border:1px solid var(--graph-border);">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--graph)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="18" cy="5" r="3"></circle>
        <circle cx="6" cy="12" r="3"></circle>
        <circle cx="18" cy="19" r="3"></circle>
        <line x1="8.59" y1="13.51" x2="15.42" y2="17.49"></line>
        <line x1="15.41" y1="6.51" x2="8.59" y2="10.49"></line>
      </svg>
    </div>
    <span class="retrieval-card-title">GraphRAG</span>
  </div>
  <span class="badge badge-graph">Relational Graph</span>
</div>
<p class="retrieval-card-desc">
  Extracts named entities and typed relationships into Neo4j property graphs. Traverses multi-hop entity chains to synthesize cross-document relational insights that vectors miss.
</p>
</div>"""
    )
    if st.button("Compare GraphRAG", use_container_width=True, key="feat_btn_graph"):
        st.switch_page("pages/4_Compare.py")

with col_f3:
    render_html(
        """<div class="retrieval-card">
<div class="retrieval-card-top">
  <div class="retrieval-card-title-group">
    <div class="retrieval-card-icon" style="background:var(--hybrid-bg);border:1px solid var(--hybrid-border);">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="var(--hybrid)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
        <polyline points="2 17 12 22 22 17"></polyline>
        <polyline points="2 12 12 17 22 12"></polyline>
      </svg>
    </div>
    <span class="retrieval-card-title">Hybrid RAG</span>
  </div>
  <span class="badge badge-hybrid">Rank Fusion</span>
</div>
<p class="retrieval-card-desc">
  Fuses dense vector semantic similarity with knowledge graph path traversal using Reciprocal Rank Fusion (RRF). Balances semantic recall with relational precision for high accuracy.
</p>
</div>"""
    )
    if st.button("Compare Hybrid RAG", use_container_width=True, key="feat_btn_hybrid"):
        st.switch_page("pages/4_Compare.py")

render_html("<div style='height:36px;'></div>")

# ── 5. How It Works ──────────────────────────────────────────────────────────
render_html(
    """<div style="margin-bottom:16px;">
<h2 style="font-size:22px;font-weight:800;color:var(--text);margin:0 0 4px 0;letter-spacing:-0.02em;">
  Experimental Workflow
</h2>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Three systematic stages from raw domain files to reproducible benchmark results.
</p>
</div>"""
)

c_s1, c_s2, c_s3 = st.columns(3, gap="medium")

with c_s1:
    render_html(
        """<div class="rag-card" style="min-height:140px;">
          <div style="font-size:16px;font-weight:800;color:var(--accent);margin-bottom:8px;">01 — Ingest &amp; Index</div>
          <div style="font-size:13.5px;color:var(--text-muted);line-height:1.5;">
            Upload PDF, DOCX, or TXT documents into an isolated knowledge base. Automatic chunking, vector embedding, and entity-relationship extraction.
          </div>
        </div>"""
    )

with c_s2:
    render_html(
        """<div class="rag-card" style="min-height:140px;">
          <div style="font-size:16px;font-weight:800;color:var(--vector);margin-bottom:8px;">02 — Query &amp; Inspect</div>
          <div style="font-size:13.5px;color:var(--text-muted);line-height:1.5;">
            Chat conversationally or run side-by-side queries. Inspect verifiable citations, similarity scores, and multi-hop graph paths.
          </div>
        </div>"""
    )

with c_s3:
    render_html(
        """<div class="rag-card" style="min-height:140px;">
          <div style="font-size:16px;font-weight:800;color:var(--success);margin-bottom:8px;">03 — Benchmark &amp; Score</div>
          <div style="font-size:13.5px;color:var(--text-muted);line-height:1.5;">
            Execute standardized benchmark suites to evaluate faithfulness, answer relevancy, context precision, and latency distributions.
          </div>
        </div>"""
    )

render_html("<div style='height:36px;'></div>")

# ── 6. Active Research Workspace Status ───────────────────────────────────────
render_html(
    """<div style="margin-bottom:16px;">
<h2 style="font-size:22px;font-weight:800;color:var(--text);margin:0 0 4px 0;letter-spacing:-0.02em;">
  Research Workspace
</h2>
<p style="font-size:14px;color:var(--text-muted);margin:0;">
  Current session status and rapid navigation for your active knowledge base.
</p>
</div>"""
)

if is_auth:
    if user_kbs:
        active_kb_id = st.session_state.get("active_kb_id", user_kbs[0].kb_id)
        active_kb = next((kb for kb in user_kbs if kb.kb_id == active_kb_id), user_kbs[0])

        kb_doc_ids = context.user_registry.get_kb_document_ids(active_kb.kb_id, user_id)
        docs = [d for d in all_docs if d["document_id"] in set(kb_doc_ids)]

        # Determine processing status
        if not docs:
            processing_status = "No documents"
            status_badge_html = '<span class="badge badge-warning">No documents</span>'
        elif any(d.get("status", "").lower() == "failed" for d in docs):
            status_badge_html = '<span class="badge badge-error">Attention required</span>'
        else:
            status_badge_html = '<span class="badge badge-success">Ready</span>'

        # Latest evaluation info
        if eval_runs:
            latest_run = eval_runs[0]
            eval_date = latest_run.get("timestamp", "")[:10]
            latest_eval_status = f"{len(eval_runs)} run(s) ({eval_date})"
        else:
            latest_eval_status = "No runs yet"

        render_html(
            f"""<div class="rag-card" style="margin-bottom:16px;">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:16px;flex-wrap:wrap;gap:8px;">
                <div>
                  <span style="font-size:11px;color:var(--text-muted);text-transform:uppercase;letter-spacing:0.06em;font-weight:600;">Active Knowledge Base</span>
                  <div style="font-size:19px;font-weight:800;color:var(--text);">{active_kb.name}</div>
                </div>
                <div>{status_badge_html}</div>
              </div>
            </div>"""
        )

        ws1, ws2, ws3, ws4 = st.columns(4)
        with ws1:
            render_html(
                f"""<div class="stat-card" style="padding:16px;">
                  <div class="stat-value">{len(docs)}</div>
                  <div class="stat-label">KB Documents</div>
                </div>"""
            )
        with ws2:
            render_html(
                f"""<div class="stat-card" style="padding:16px;">
                  <div class="stat-value" style="color:var(--vector);">{real_counts.get("chunks", 0)}</div>
                  <div class="stat-label">Indexed Chunks</div>
                </div>"""
            )
        with ws3:
            render_html(
                f"""<div class="stat-card" style="padding:16px;">
                  <div class="stat-value" style="color:var(--graph);">{real_counts.get("entities", 0)}</div>
                  <div class="stat-label">Graph Entities</div>
                </div>"""
            )
        with ws4:
            render_html(
                f"""<div class="stat-card" style="padding:16px;">
                  <div class="stat-value" style="font-size:20px;color:var(--accent);line-height:1.4;">{latest_eval_status}</div>
                  <div class="stat-label">Benchmark Status</div>
                </div>"""
            )

        render_html("<div style='height:16px;'></div>")

        c_act1, c_act2, c_act3 = st.columns(3)
        with c_act1:
            if st.button("Manage Documents", use_container_width=True, key="home_ws_manage_docs"):
                st.switch_page("pages/2_Documents.py")
        with c_act2:
            if st.button("Start Chat", type="primary", use_container_width=True, key="home_ws_start_chat"):
                st.switch_page("pages/3_Chat.py")
        with c_act3:
            if st.button("Run Evaluation", use_container_width=True, key="home_ws_run_eval"):
                st.switch_page("pages/5_Evaluation.py")

    else:
        render_html(
            """<div class="rag-card" style="text-align:center;padding:36px 20px;">
              <h4 style="font-size:17px;font-weight:800;color:var(--text);margin:0 0 6px 0;">No Knowledge Bases Created</h4>
              <p style="font-size:13.5px;color:var(--text-muted);max-width:480px;margin:0 auto 18px;line-height:1.5;">
                Create your first knowledge base in Documents to start uploading PDF, DOCX, or TXT documents.
              </p>
            </div>"""
        )
        c_k1, c_k2, c_k3 = st.columns([1, 1.2, 1])
        with c_k2:
            if st.button("Create Knowledge Base", type="primary", use_container_width=True, key="home_btn_create_kb"):
                st.session_state["show_create_kb"] = True
                st.switch_page("pages/2_Documents.py")

else:
    render_html(
        """<div class="rag-card" style="text-align:center;padding:36px 20px;">
          <h4 style="font-size:18px;font-weight:800;color:var(--text);margin:0 0 8px 0;">Get Started with RAG Benchmark Lab</h4>
          <p style="font-size:14px;color:var(--text-muted);max-width:540px;margin:0 auto 20px;line-height:1.6;">
            Sign in to upload research documents, build connected knowledge graphs, query using Vector RAG and GraphRAG, and evaluate empirical performance.
          </p>
        </div>"""
    )
    c_s_sp1, c_s_btn1, c_s_btn2, c_s_sp2 = st.columns([1.5, 1.2, 1.2, 1.5])
    with c_s_btn1:
        if st.button("Sign In", type="primary", use_container_width=True, key="home_btn_signin"):
            st.switch_page("pages/0_Login.py")
    with c_s_btn2:
        if st.button("Create Account", use_container_width=True, key="home_btn_create_acct"):
            st.switch_page("pages/0_Login.py")
