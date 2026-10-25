"""RAG Benchmark Lab — Unified UI utilities.

Dark-theme design system, native top-navigation bar, and shared styling components.
"""
from __future__ import annotations

import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# DESIGN-SYSTEM CSS
# ─────────────────────────────────────────────────────────────────────────────

_DESIGN_SYSTEM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root Tokens ─────────────────────────────────────────────────────────── */
:root {
  --bg:          #0B0D12;
  --nav-bg:      #10141D;
  --card:        #151A24;
  --card-hi:     #1B2230;
  --text:        #F5F7FA;
  --text-muted:  #A0AABC;
  --border:      #293244;
  --accent:      #8B7CFF;
  --accent2:     #39C6B5;
  --success:     #35C98A;
  --warning:     #F5B942;
  --error:       #FF6B78;
  --vector:      #60A5FA;
  --graph:       #A78BFA;
  --hybrid:      #39C6B5;
  --radius:      8px;
  --radius-lg:   12px;
}

/* ── Global Reset ────────────────────────────────────────────────────────── */
html, body, [class*="css"] {
  font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  color: var(--text) !important;
}

/* ── App Background ──────────────────────────────────────────────────────── */
.stApp {
  background-color: var(--bg) !important;
}

/* ── Hide Streamlit default sidebar navigation & header chrome ───────────── */
section[data-testid="stSidebar"] {
  display: none !important;
}
[data-testid="collapsedControl"] {
  display: none !important;
}
button[kind="header"] {
  display: none !important;
}
header[data-testid="stHeader"] {
  background: transparent !important;
  height: 0px !important;
}
#MainMenu {
  visibility: hidden !important;
}
footer {
  visibility: hidden !important;
}

/* ── Main content area — centered container with clean max-width ─────────── */
.main .block-container {
  padding-top: 0 !important;
  padding-left: 1.5rem !important;
  padding-right: 1.5rem !important;
  padding-bottom: 3rem !important;
  max-width: 1160px !important;
  margin: 0 auto !important;
}

/* ── Top Navigation Bar Container ────────────────────────────────────────── */
[data-testid="stContainer"].st-key-rag_topnav {
  background: var(--nav-bg) !important;
  border-bottom: 1px solid var(--border) !important;
  padding: 8px 20px !important;
  min-height: 64px !important;
  margin-left: -1.5rem !important;
  margin-right: -1.5rem !important;
  margin-top: 0 !important;
  margin-bottom: 24px !important;
  box-sizing: border-box !important;
  position: sticky !important;
  top: 0 !important;
  z-index: 999 !important;
}

.st-key-rag_topnav [data-testid="stHorizontalBlock"] {
  align-items: center !important;
  gap: 8px !important;
}

/* ── Top Nav PageLinks ───────────────────────────────────────────────────── */
.st-key-rag_topnav [data-testid="stPageLink"] {
  display: flex !important;
  align-items: center !important;
  justify-content: center !important;
  margin: 0 !important;
  padding: 0 !important;
}

.st-key-rag_topnav [data-testid="stPageLink"] a {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  padding: 6px 12px !important;
  border-radius: 6px !important;
  font-size: 13.5px !important;
  font-weight: 500 !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  text-decoration: none !important;
  white-space: nowrap !important;
  transition: all 0.15s ease !important;
  width: 100% !important;
  text-align: center !important;
}

.st-key-rag_topnav [data-testid="stPageLink"] a:hover {
  color: var(--text) !important;
  background: rgba(255, 255, 255, 0.05) !important;
  border-color: rgba(255, 255, 255, 0.08) !important;
}

.st-key-rag_topnav [data-testid="stPageLink"] a[aria-current="page"] {
  color: var(--accent) !important;
  background: rgba(139, 124, 255, 0.12) !important;
  border-color: rgba(139, 124, 255, 0.3) !important;
  font-weight: 600 !important;
}

/* Brand Link in Topnav */
.st-key-nav_brand_col [data-testid="stPageLink"] a {
  font-size: 15px !important;
  font-weight: 700 !important;
  color: var(--text) !important;
  letter-spacing: -0.02em !important;
  justify-content: flex-start !important;
  padding: 6px 4px !important;
  border: none !important;
  background: transparent !important;
}
.st-key-nav_brand_col [data-testid="stPageLink"] a:hover {
  color: var(--accent) !important;
  background: transparent !important;
}
.st-key-nav_brand_col [data-testid="stPageLink"] span[data-testid="stIconMaterial"] {
  color: var(--accent) !important;
  font-size: 20px !important;
}

/* Top Nav Popover Button (User profile) */
.st-key-rag_topnav [data-testid="stPopover"] button {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  padding: 4px 12px !important;
  border-radius: 6px !important;
  height: 36px !important;
}
.st-key-rag_topnav [data-testid="stPopover"] button:hover {
  border-color: var(--accent) !important;
  background: var(--card-hi) !important;
}

/* ── Card Components ─────────────────────────────────────────────────────── */
.rag-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
  transition: border-color 0.2s ease;
}

.rag-card:hover {
  border-color: rgba(139, 124, 255, 0.35);
}

.rag-card-elevated {
  background: var(--card-hi);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 24px;
}

/* ── Strategy & Status Badges (Strictly No Emojis) ───────────────────────── */
.badge {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 3px 10px;
  border-radius: 6px;
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
}
.badge-vector { background: rgba(96, 165, 250, 0.12); color: var(--vector); border: 1px solid rgba(96, 165, 250, 0.3); }
.badge-graph  { background: rgba(167, 139, 250, 0.12); color: var(--graph);  border: 1px solid rgba(167, 139, 250, 0.3); }
.badge-hybrid { background: rgba(57, 198, 181, 0.12);  color: var(--hybrid); border: 1px solid rgba(57, 198, 181, 0.3); }
.badge-success { background: rgba(53, 201, 138, 0.12); color: var(--success); border: 1px solid rgba(53, 201, 138, 0.3); }
.badge-warning { background: rgba(245, 185, 66, 0.12); color: var(--warning); border: 1px solid rgba(245, 185, 66, 0.3); }
.badge-error   { background: rgba(255, 107, 120, 0.12); color: var(--error); border: 1px solid rgba(255, 107, 120, 0.3); }

/* ── Section Headers ─────────────────────────────────────────────────────── */
.section-header {
  margin: 32px 0 16px 0;
  padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.section-title {
  font-size: 20px;
  font-weight: 700;
  color: var(--text);
  margin: 0;
  letter-spacing: -0.02em;
}
.section-sub {
  font-size: 13.5px;
  color: var(--text-muted);
  margin-top: 4px;
}

/* ── Form Inputs & Text Areas ────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  font-family: 'Inter', sans-serif !important;
  font-size: 13.5px !important;
}

.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 1px var(--accent) !important;
}

.stSelectbox > div > div {
  background: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
}

/* ── Standardized Buttons ────────────────────────────────────────────────── */
.stButton > button {
  border-radius: var(--radius) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  transition: all 0.15s ease !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  color: var(--text) !important;
  height: 38px !important;
}

.stButton > button:hover {
  border-color: var(--accent) !important;
  color: var(--text) !important;
  background: var(--card-hi) !important;
}

.stButton > button[kind="primary"] {
  background: var(--accent) !important;
  color: #0B0D12 !important;
  border: 1px solid var(--accent) !important;
  font-weight: 700 !important;
}

.stButton > button[kind="primary"]:hover {
  background: #9D90FF !important;
  border-color: #9D90FF !important;
}

/* ── Tables & Dataframes ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  background: var(--card) !important;
}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  padding: 4px !important;
  gap: 2px !important;
  border: 1px solid var(--border);
}
.stTabs [data-baseweb="tab"] {
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
  font-size: 13px !important;
  padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: var(--card-hi) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}
.stTabs [data-baseweb="tab-panel"] {
  padding-top: 16px !important;
}

/* ── Expander ────────────────────────────────────────────────────────────── */
.stExpander {
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--card) !important;
}
.stExpander header {
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
}

/* ── Chat Bubbles ────────────────────────────────────────────────────────── */
.chat-user {
  background: rgba(139, 124, 255, 0.12);
  border: 1px solid rgba(139, 124, 255, 0.25);
  border-radius: 12px 12px 4px 12px;
  padding: 12px 16px;
  margin: 8px 0;
  max-width: 80%;
  margin-left: auto;
  font-size: 14px;
  color: var(--text);
  line-height: 1.6;
}
.chat-assistant {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 12px 12px 12px 4px;
  padding: 14px 18px;
  margin: 8px 0;
  max-width: 85%;
  font-size: 14px;
  color: var(--text);
  line-height: 1.65;
}

/* ── Dev mode notice ─────────────────────────────────────────────────────── */
.dev-banner {
  background: rgba(245, 185, 66, 0.08);
  border: 1px solid rgba(245, 185, 66, 0.3);
  border-left: 3px solid var(--warning);
  padding: 10px 14px;
  border-radius: var(--radius);
  margin-bottom: 20px;
  font-size: 13px;
  color: var(--warning);
}

/* ── Stat Metric Card ────────────────────────────────────────────────────── */
.stat-card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 18px 20px;
  text-align: center;
}
.stat-value {
  font-size: 26px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.02em;
}
.stat-label {
  font-size: 12px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-top: 4px;
}

/* ── Responsive adjustments ──────────────────────────────────────────────── */
@media (max-width: 900px) {
  .st-key-rag_topnav {
    padding: 8px 12px !important;
  }
  .st-key-rag_topnav [data-testid="stPageLink"] a {
    padding: 4px 8px !important;
    font-size: 12px !important;
  }
}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3D4E6A; }
</style>
"""


def inject_custom_css() -> None:
    """Inject the unified dark-theme design system CSS."""
    st.markdown(_DESIGN_SYSTEM_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# NATIVE TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_topnav(active_page: str = "Home") -> None:
    """
    Render a clean, native top navigation bar using Streamlit's official routing components.

    Zero raw HTML strings are injected for navigation to prevent CommonMark escaping bugs.
    Features:
      LEFT: Geometric icon + "RAG Benchmark Lab"
      CENTER: Home, Documents, Chat, Compare, Evaluation
      RIGHT:
        - Authenticated: Active KB pill and User profile popover with sign-out
        - Logged-out: Sign In and Create Account actions
    """
    inject_custom_css()

    from rag_benchmark_lab.utils.auth import get_current_user_name, is_authenticated

    user_name = get_current_user_name()
    is_auth = is_authenticated()
    active_kb_name = st.session_state.get("active_kb_name", "")

    with st.container(key="rag_topnav"):
        c_brand, c_nav, c_right = st.columns([2.5, 4.4, 2.7], vertical_alignment="center")

        with c_brand:
            with st.container(key="nav_brand_col"):
                st.page_link(
                    "pages/1_Home.py",
                    label="RAG Benchmark Lab",
                    icon=":material/hub:",
                )

        with c_nav:
            n1, n2, n3, n4, n5 = st.columns(5, gap="small", vertical_alignment="center")
            with n1:
                st.page_link("pages/1_Home.py", label="Home")
            with n2:
                st.page_link("pages/2_Documents.py", label="Documents")
            with n3:
                st.page_link("pages/3_Chat.py", label="Chat")
            with n4:
                st.page_link("pages/4_Compare.py", label="Compare")
            with n5:
                st.page_link("pages/5_Evaluation.py", label="Evaluation")

        with c_right:
            if is_auth:
                r_kb, r_user = st.columns([1.5, 1.2], gap="small", vertical_alignment="center")
                with r_kb:
                    if active_kb_name:
                        display_kb = active_kb_name if len(active_kb_name) <= 15 else active_kb_name[:13] + "…"
                        st.page_link(
                            "pages/2_Documents.py",
                            label=display_kb,
                            icon=":material/folder_open:",
                            help=f"Active Knowledge Base: {active_kb_name}",
                        )
                    else:
                        st.page_link(
                            "pages/2_Documents.py",
                            label="Create KB",
                            icon=":material/add:",
                            help="Create or select a knowledge base",
                        )
                with r_user:
                    short_name = user_name.split()[0] if user_name else "Account"
                    with st.popover(short_name, icon=":material/account_circle:", use_container_width=True):
                        st.caption("Signed in as")
                        st.markdown(f"**{user_name}**")
                        if active_kb_name:
                            st.caption(f"Knowledge Base: **{active_kb_name}**")
                        st.markdown("<hr style='border-color:#293244;margin:8px 0;'/>", unsafe_allow_html=True)
                        if st.button("Sign Out", use_container_width=True, key="nav_logout_btn"):
                            from rag_benchmark_lab.utils.auth import logout_user
                            logout_user()
                            st.rerun()
            else:
                r_in, r_reg = st.columns([1.0, 1.3], gap="small", vertical_alignment="center")
                with r_in:
                    st.page_link("pages/0_Login.py", label="Sign In", icon=":material/login:")
                with r_reg:
                    st.page_link("pages/0_Login.py", label="Create Account", icon=":material/person_add:")


# ─────────────────────────────────────────────────────────────────────────────
# REUSABLE COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_dev_mode_banner(is_mock: bool = True) -> None:
    """Show a clean notice when Ollama LLM is in mock/offline mode."""
    if is_mock:
        st.markdown(
            """
            <div class="dev-banner">
              <strong>Development Mode:</strong> Ollama LLM is offline — answers are synthesised locally using grounded document context. Vector retrieval (Qdrant) and graph traversal (Neo4j / NetworkX) run normally.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_strategy_badge(strategy_name: str) -> str:
    """Return inline HTML badge for a retrieval strategy without emojis."""
    s = str(strategy_name).upper()
    if "VECTOR" in s:
        return '<span class="badge badge-vector">Vector RAG</span>'
    if "GRAPH" in s:
        return '<span class="badge badge-graph">GraphRAG</span>'
    if "HYBRID" in s:
        return '<span class="badge badge-hybrid">Hybrid RAG</span>'
    return f'<span class="badge badge-vector">{s}</span>'


def render_stat_cards(stats: list[tuple[str, str, str | None]]) -> None:
    """Render a row of stat cards."""
    cols = st.columns(len(stats))
    for col, (label, value, help_text) in zip(cols, stats):
        with col:
            title_attr = f' title="{help_text}"' if help_text else ""
            st.markdown(
                f"""
                <div class="stat-card"{title_attr}>
                  <div class="stat-value">{value}</div>
                  <div class="stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_kb_selector(context: object, user_id: str, key_prefix: str = "kb") -> str | None:
    """
    Inline KB selector used on pages that need the active knowledge base.
    Returns the selected KB id, or None if no KBs exist.
    """
    kbs = context.user_registry.list_knowledge_bases(user_id)
    if not kbs:
        return None

    kb_options = {kb.kb_id: kb.name for kb in kbs}
    active_kb_id = st.session_state.get("active_kb_id")
    if active_kb_id not in kb_options:
        active_kb_id = list(kb_options.keys())[0]
        st.session_state["active_kb_id"] = active_kb_id
        st.session_state["active_kb_name"] = kb_options[active_kb_id]

    selected = st.selectbox(
        "Active Knowledge Base",
        options=list(kb_options.keys()),
        format_func=lambda x: kb_options[x],
        index=list(kb_options.keys()).index(active_kb_id),
        key=f"{key_prefix}_selector",
        label_visibility="collapsed",
    )
    if selected != active_kb_id:
        st.session_state["active_kb_id"] = selected
        st.session_state["active_kb_name"] = kb_options[selected]
        if "active_conv_id" in st.session_state:
            del st.session_state["active_conv_id"]
        st.rerun()

    return selected


def render_lab_header(title: str, subtitle: str, badge_text: str = "") -> None:
    """Compatibility shim for section headers."""
    inject_custom_css()
    badge_html = (
        f'<span style="background:rgba(139,124,255,0.15);color:#A78BFA;'
        f'font-size:11px;padding:3px 10px;border-radius:6px;font-weight:700;'
        f'letter-spacing:0.04em;border:1px solid rgba(167,139,250,0.3);">'
        f'{badge_text}</span>'
        if badge_text else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom:20px;">
          <div style="font-size:22px;font-weight:700;letter-spacing:-0.02em;
                      color:var(--text,#F5F7FA);display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            {title} {badge_html}
          </div>
          <div style="font-size:13.5px;color:var(--text-muted,#A0AABC);margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
