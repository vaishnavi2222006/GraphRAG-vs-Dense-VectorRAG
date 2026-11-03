"""RAG Benchmark Lab — Unified UI design system, theme management, and shared components.

Provides light/dark theme persistence, modern SaaS navigation, responsive layouts,
Plotly chart theming, and reusable UI components inspired by Linear, Vercel, Stripe, and Notion.
"""
from __future__ import annotations

import math
from typing import Any
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
# 1. THEME STATE & PERSISTENCE
# ─────────────────────────────────────────────────────────────────────────────

def get_current_theme() -> str:
    """Retrieve the active theme ('dark' or 'light') with multi-layer persistence.

    Priority:
      1. st.query_params['theme'] (URL parameter - handles browser refresh & link sharing)
      2. st.session_state['theme'] (active session state)
      3. Default: 'dark'

    Synchronizes session state, query params, and client localStorage for seamless cross-refresh persistence.
    """
    theme = "dark"
    # Check query params first
    try:
        qp = st.query_params.get("theme", "")
        if qp in ("dark", "light"):
            theme = qp
    except Exception:
        pass

    # If query param was empty, check session state
    if not st.query_params.get("theme") and "theme" in st.session_state and st.session_state["theme"] in ("dark", "light"):
        theme = st.session_state["theme"]

    st.session_state["theme"] = theme

    # Synchronize query parameter for refresh persistence
    try:
        if st.query_params.get("theme") != theme:
            st.query_params["theme"] = theme
    except Exception:
        pass

    return theme


def toggle_theme() -> str:
    """Toggle current theme between 'light' and 'dark', update persistence, and trigger rerun."""
    current = get_current_theme()
    new_theme = "light" if current == "dark" else "dark"
    st.session_state["theme"] = new_theme
    try:
        st.query_params["theme"] = new_theme
    except Exception:
        pass
    st.rerun()
    return new_theme


def set_theme(theme_name: str) -> None:
    """Explicitly set the theme to 'dark' or 'light'."""
    if theme_name in ("dark", "light"):
        st.session_state["theme"] = theme_name
        try:
            st.query_params["theme"] = theme_name
        except Exception:
            pass
        st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# 2. DESIGN-SYSTEM CSS GENERATOR (LIGHT & DARK MODES)
# ─────────────────────────────────────────────────────────────────────────────

def _generate_theme_css(theme: str) -> str:
    """Generate high-contrast, polished design system CSS tokens and rules."""
    is_dark = (theme == "dark")

    if is_dark:
        # Dark Mode Palette: Deep charcoal obsidian, modern indigo accent, glowing badges
        tokens = """
  --bg:              #08090D;
  --bg-gradient:     linear-gradient(180deg, #0D1017 0%, #08090D 100%);
  --nav-bg:          rgba(13, 16, 23, 0.88);
  --nav-border:      rgba(255, 255, 255, 0.08);
  --card:            #11141D;
  --card-hi:         #171B26;
  --card-inner:      #0B0E14;
  --text:            #F8FAFC;
  --text-muted:      #94A3B8;
  --text-subtle:     #64748B;
  --border:          #1E2433;
  --border-subtle:   #141824;
  --border-focus:    #6366F1;
  --accent:          #6366F1;
  --accent-hover:    #818CF8;
  --accent-light:    rgba(99, 102, 241, 0.14);
  --accent-border:   rgba(99, 102, 241, 0.32);
  --accent2:         #2DD4BF;
  --success:         #10B981;
  --success-bg:      rgba(16, 185, 129, 0.12);
  --success-border:  rgba(16, 185, 129, 0.28);
  --warning:         #F59E0B;
  --warning-bg:      rgba(245, 158, 11, 0.12);
  --warning-border:  rgba(245, 158, 11, 0.28);
  --error:           #EF4444;
  --error-bg:        rgba(239, 68, 68, 0.12);
  --error-border:    rgba(239, 68, 68, 0.28);
  --vector:          #38BDF8;
  --vector-bg:       rgba(56, 189, 248, 0.12);
  --vector-border:   rgba(56, 189, 248, 0.30);
  --graph:           #A855F7;
  --graph-bg:        rgba(168, 85, 247, 0.12);
  --graph-border:    rgba(168, 85, 247, 0.30);
  --hybrid:          #2DD4BF;
  --hybrid-bg:       rgba(45, 212, 191, 0.12);
  --hybrid-border:   rgba(45, 212, 191, 0.30);
  --radius-sm:       6px;
  --radius:          10px;
  --radius-lg:       14px;
  --radius-full:     9999px;
  --shadow-sm:       0 1px 3px rgba(0, 0, 0, 0.35);
  --shadow-md:       0 6px 20px rgba(0, 0, 0, 0.45);
  --shadow-lg:       0 14px 36px rgba(0, 0, 0, 0.60);
  --shadow-glow:     0 0 24px rgba(99, 102, 241, 0.22);
  --input-bg:        #11141D;
  --input-border:    #222838;
  --code-bg:         #0B0E14;
        """
        st_app_overrides = """
  background-color: var(--bg) !important;
  color: var(--text) !important;
        """
    else:
        # Light Mode Palette: Crisp off-white slate, deep modern indigo, clean borders
        tokens = """
  --bg:              #F8FAFC;
  --bg-gradient:     linear-gradient(180deg, #FFFFFF 0%, #F8FAFC 100%);
  --nav-bg:          rgba(255, 255, 255, 0.88);
  --nav-border:      rgba(226, 232, 240, 0.90);
  --card:            #FFFFFF;
  --card-hi:         #F1F5F9;
  --card-inner:      #F8FAFC;
  --text:            #0F172A;
  --text-muted:      #64748B;
  --text-subtle:     #94A3B8;
  --border:          #E2E8F0;
  --border-subtle:   #EDF2F7;
  --border-focus:    #4F46E5;
  --accent:          #4F46E5;
  --accent-hover:    #4338CA;
  --accent-light:    rgba(79, 70, 229, 0.08);
  --accent-border:   rgba(79, 70, 229, 0.25);
  --accent2:         #0D9488;
  --success:         #059669;
  --success-bg:      rgba(5, 150, 105, 0.08);
  --success-border:  rgba(5, 150, 105, 0.25);
  --warning:         #D97706;
  --warning-bg:      rgba(217, 119, 6, 0.08);
  --warning-border:  rgba(217, 119, 6, 0.25);
  --error:           #DC2626;
  --error-bg:        rgba(220, 38, 38, 0.08);
  --error-border:    rgba(220, 38, 38, 0.25);
  --vector:          #0284C7;
  --vector-bg:       rgba(2, 132, 199, 0.08);
  --vector-border:   rgba(2, 132, 199, 0.25);
  --graph:           #7C3AED;
  --graph-bg:        rgba(124, 58, 237, 0.08);
  --graph-border:    rgba(124, 58, 237, 0.25);
  --hybrid:          #0D9488;
  --hybrid-bg:       rgba(13, 148, 136, 0.08);
  --hybrid-border:   rgba(13, 148, 136, 0.25);
  --radius-sm:       6px;
  --radius:          10px;
  --radius-lg:       14px;
  --radius-full:     9999px;
  --shadow-sm:       0 1px 3px rgba(0, 0, 0, 0.05), 0 1px 2px rgba(0, 0, 0, 0.03);
  --shadow-md:       0 4px 14px rgba(0, 0, 0, 0.06), 0 1px 3px rgba(0, 0, 0, 0.03);
  --shadow-lg:       0 12px 28px rgba(0, 0, 0, 0.08), 0 4px 8px rgba(0, 0, 0, 0.03);
  --shadow-glow:     0 0 24px rgba(79, 70, 229, 0.12);
  --input-bg:        #FFFFFF;
  --input-border:    #CBD5E1;
  --code-bg:         #F1F5F9;
        """
        st_app_overrides = """
  background-color: var(--bg) !important;
  color: var(--text) !important;
        """

    return f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Root Design System Tokens ───────────────────────────────────────────── */
:root {{
{tokens}
}}

/* ── Global Typography & Base ────────────────────────────────────────────── */
html, body, [class*="css"] {{
  font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
  color: var(--text) !important;
}}

html, body, .stApp {{
{st_app_overrides}
  overflow-x: hidden !important;
  margin: 0 !important;
  padding: 0 !important;
  transition: background-color 0.25s ease, color 0.25s ease !important;
}}

/* ── Smooth transitions for all theme elements ───────────────────────────── */
.rag-card, .stat-card, .retrieval-card, .arch-node, .stButton button, [data-testid="stContainer"] {{
  transition: background-color 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease, color 0.25s ease !important;
}}

/* ── Scrollbars ──────────────────────────────────────────────────────────── */
::-webkit-scrollbar {{
  width: 7px;
  height: 7px;
}}
::-webkit-scrollbar-track {{
  background: var(--bg);
}}
::-webkit-scrollbar-thumb {{
  background: var(--border);
  border-radius: var(--radius-full);
}}
::-webkit-scrollbar-thumb:hover {{
  background: var(--text-subtle);
}}

/* ── Eliminate Streamlit Default Chrome ───────────────────────────────────── */
header[data-testid="stHeader"],
[data-testid="stHeader"],
.stAppHeader,
[data-testid="stDecoration"],
[data-testid="stToolbar"] {{
  display: none !important;
  height: 0px !important;
  min-height: 0px !important;
  max-height: 0px !important;
  padding: 0 !important;
  margin: 0 !important;
  border: none !important;
  opacity: 0 !important;
  visibility: hidden !important;
  pointer-events: none !important;
}}

section[data-testid="stSidebar"],
[data-testid="collapsedControl"],
button[kind="header"],
#MainMenu,
footer {{
  display: none !important;
  visibility: hidden !important;
}}

/* Clean element containers */
div[data-testid="stElementContainer"]:has(style),
div[data-testid="element-container"]:has(style),
.element-container:has(style),
div[data-testid="stElementContainer"]:empty {{
  display: none !important;
  height: 0px !important;
  min-height: 0px !important;
  margin: 0 !important;
  padding: 0 !important;
  position: absolute !important;
}}

/* ── Streamlit Main Content Block (Centered, 1360px max) ─────────────────── */
.stApp,
[data-testid="stAppViewContainer"],
.main,
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
.main .block-container,
div[data-testid="stAppViewBlockContainer"],
section.main {{
  padding-top: 0 !important;
  margin-top: 0 !important;
  background-color: var(--bg) !important;
}}

.main .block-container,
[data-testid="stMainBlockContainer"] {{
  padding-top: 0 !important;
  padding-left: 24px !important;
  padding-right: 24px !important;
  padding-bottom: 56px !important;
  max-width: 1360px !important;
  margin-left: auto !important;
  margin-right: auto !important;
}}

.main .block-container > div[data-testid="stVerticalBlock"] {{
  gap: 0 !important;
  padding-top: 0 !important;
}}

/* ── Top Navigation Bar Container (Floating Glassmorphism) ───────────────── */
[data-testid="stContainer"].st-key-rag_topnav {{
  background: var(--nav-bg) !important;
  backdrop-filter: blur(20px) !important;
  -webkit-backdrop-filter: blur(20px) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 0 18px !important;
  min-height: 72px !important;
  max-height: 76px !important;
  height: 72px !important;
  width: 100% !important;
  max-width: 1360px !important;
  position: sticky !important;
  top: 14px !important;
  margin-top: 14px !important;
  margin-bottom: 26px !important;
  margin-left: auto !important;
  margin-right: auto !important;
  box-sizing: border-box !important;
  z-index: 9999 !important;
  display: flex !important;
  align-items: center !important;
  box-shadow: var(--shadow-md) !important;
  transition: all 0.25s ease !important;
}}

[data-testid="stContainer"].st-key-rag_topnav > div[data-testid="stVerticalBlockBorderWrapper"],
[data-testid="stContainer"].st-key-rag_topnav > div[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] {{
  width: 100% !important;
  max-width: 100% !important;
  margin: 0 auto !important;
  padding: 0 !important;
}}

.st-key-rag_topnav [data-testid="stHorizontalBlock"] {{
  align-items: center !important;
  gap: 10px !important;
  width: 100% !important;
  margin: 0 auto !important;
}}

/* Topnav Brand Link */
.st-key-nav_brand_col {{
  display: flex !important;
  align-items: center !important;
}}
.st-key-nav_brand_col [data-testid="stPageLink"] a {{
  font-size: 16px !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  letter-spacing: -0.03em !important;
  justify-content: flex-start !important;
  padding: 8px 6px !important;
  border: none !important;
  background: transparent !important;
  white-space: nowrap !important;
  gap: 8px !important;
}}
.st-key-nav_brand_col [data-testid="stPageLink"] a:hover {{
  color: var(--accent) !important;
  background: transparent !important;
}}
.st-key-nav_brand_col [data-testid="stPageLink"] span[data-testid="stIconMaterial"] {{
  color: var(--accent) !important;
  font-size: 22px !important;
}}

/* Topnav Navigation Links */
.st-key-rag_topnav [data-testid="stPageLink"] a {{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  padding: 7px 11px !important;
  border-radius: var(--radius-sm) !important;
  font-size: 13px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  border: 1px solid transparent !important;
  background: transparent !important;
  text-decoration: none !important;
  white-space: nowrap !important;
  transition: all 0.15s ease !important;
}}
.st-key-rag_topnav [data-testid="stPageLink"] a:hover {{
  color: var(--text) !important;
  background: var(--card-hi) !important;
  border-color: var(--border) !important;
}}

/* Topnav Right Side Controls (Theme Toggle & User Profile) */
.st-key-rag_topnav [data-testid="stPopover"] button,
.st-key-rag_topnav [data-testid="stButton"] button {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  padding: 5px 12px !important;
  border-radius: var(--radius-sm) !important;
  height: 36px !important;
  min-height: 36px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.15s ease !important;
}}
.st-key-rag_topnav [data-testid="stPopover"] button:hover,
.st-key-rag_topnav [data-testid="stButton"] button:hover {{
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--card-hi) !important;
}}

/* Theme Toggle Button Specific Styling */
.st-key-nav_theme_toggle button {{
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 6px !important;
  border-radius: var(--radius-full) !important;
  padding: 0 14px !important;
  height: 36px !important;
  min-height: 36px !important;
  border: 1px solid var(--border) !important;
  background: var(--card-hi) !important;
  color: var(--text) !important;
  font-size: 12.5px !important;
  font-weight: 600 !important;
  box-shadow: var(--shadow-sm) !important;
}}
.st-key-nav_theme_toggle button:hover {{
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-glow) !important;
  transform: translateY(-1px) !important;
}}

/* Auth Buttons in Topnav */
.st-key-nav_auth_signin [data-testid="stPageLink"] a {{
  background: var(--accent-light) !important;
  border: 1px solid var(--accent-border) !important;
  color: var(--accent) !important;
  font-weight: 600 !important;
  border-radius: var(--radius-sm) !important;
  height: 36px !important;
}}
.st-key-nav_auth_signin [data-testid="stPageLink"] a:hover {{
  background: var(--accent) !important;
  color: #FFFFFF !important;
}}
.st-key-nav_auth_reg [data-testid="stPageLink"] a {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: var(--radius-sm) !important;
  height: 36px !important;
}}

/* ── BaseWeb Popovers & Selectbox Dropdowns ───────────────────────────────── */
[data-baseweb="popover"],
[data-baseweb="menu"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"] > div,
ul[role="listbox"],
div[role="listbox"] {{
  background-color: var(--card) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-lg) !important;
}}

li[role="option"],
div[role="option"] {{
  background-color: transparent !important;
  color: var(--text) !important;
  font-size: 13.5px !important;
  padding: 8px 14px !important;
  border-radius: var(--radius-sm) !important;
  margin: 2px 4px !important;
  transition: all 0.12s ease !important;
}}

li[role="option"]:hover,
li[role="option"][aria-selected="true"],
div[role="option"]:hover,
div[role="option"][aria-selected="true"] {{
  background-color: var(--card-hi) !important;
  color: var(--accent) !important;
}}

[data-baseweb="tag"] {{
  background: var(--card-hi) !important;
  color: var(--text) !important;
  border: 1px solid var(--border) !important;
}}

/* ── Cards & Surfaces ────────────────────────────────────────────────────── */
.rag-card {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 24px 26px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
}}
.rag-card:hover {{
  border-color: var(--accent-border) !important;
  box-shadow: var(--shadow-md) !important;
}}

.rag-card-elevated {{
  background: var(--card-hi) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 24px 26px !important;
  box-shadow: var(--shadow-md) !important;
}}

/* Streamlit Native Border Containers */
[data-testid="stVerticalBlockBorderWrapper"] > div[data-testid="stVerticalBlock"] > div[data-testid="stContainer"] {{
  background: var(--card) !important;
  border-color: var(--border) !important;
  border-radius: var(--radius-lg) !important;
}}

div[data-testid="stVerticalBlockBorderWrapper"] {{
  border-color: var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-sm) !important;
}}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton > button {{
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 13.5px !important;
  transition: all 0.18s cubic-bezier(0.16, 1, 0.3, 1) !important;
  border: 1px solid var(--border) !important;
  background: var(--card) !important;
  color: var(--text) !important;
  height: 40px !important;
  min-height: 40px !important;
  padding: 0 16px !important;
  box-sizing: border-box !important;
  box-shadow: var(--shadow-sm) !important;
}}
.stButton > button:hover {{
  border-color: var(--accent) !important;
  color: var(--accent) !important;
  background: var(--card-hi) !important;
  transform: translateY(-1px) !important;
  box-shadow: var(--shadow-md) !important;
}}
.stButton > button:active {{
  transform: translateY(0px) !important;
}}

.stButton > button[kind="primary"] {{
  background: var(--accent) !important;
  color: #FFFFFF !important;
  border: 1px solid var(--accent) !important;
  font-weight: 700 !important;
  box-shadow: 0 2px 10px rgba(99, 102, 241, 0.28) !important;
}}
.stButton > button[kind="primary"]:hover {{
  background: var(--accent-hover) !important;
  border-color: var(--accent-hover) !important;
  color: #FFFFFF !important;
  box-shadow: var(--shadow-glow) !important;
  transform: translateY(-1px) !important;
}}

/* ── Form Inputs & Text Areas ────────────────────────────────────────────── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {{
  background: var(--input-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: var(--radius-sm) !important;
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  font-size: 13.5px !important;
  box-shadow: var(--shadow-sm) !important;
  transition: all 0.15s ease !important;
}}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-light) !important;
}}

.stSelectbox > div > div {{
  background: var(--input-bg) !important;
  color: var(--text) !important;
  border: 1px solid var(--input-border) !important;
  border-radius: var(--radius-sm) !important;
}}

/* ── Metrics ─────────────────────────────────────────────────────────────── */
div[data-testid="stMetricValue"] {{
  font-size: 26px !important;
  font-weight: 800 !important;
  color: var(--text) !important;
  letter-spacing: -0.03em !important;
}}
div[data-testid="stMetricLabel"] {{
  font-size: 12px !important;
  font-weight: 600 !important;
  color: var(--text-muted) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.05em !important;
}}

/* ── Tabs ────────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {{
  background: var(--card-hi) !important;
  border-radius: var(--radius) !important;
  padding: 4px !important;
  gap: 4px !important;
  border: 1px solid var(--border) !important;
}}
.stTabs [data-baseweb="tab"] {{
  background: transparent !important;
  color: var(--text-muted) !important;
  border-radius: var(--radius-sm) !important;
  font-weight: 600 !important;
  font-size: 13px !important;
  padding: 7px 16px !important;
  transition: all 0.15s ease !important;
}}
.stTabs [data-baseweb="tab"]:hover {{
  color: var(--text) !important;
}}
.stTabs [aria-selected="true"] {{
  background: var(--card) !important;
  color: var(--accent) !important;
  font-weight: 700 !important;
  box-shadow: var(--shadow-sm) !important;
}}
.stTabs [data-baseweb="tab-panel"] {{
  padding-top: 18px !important;
}}

/* ── Expanders ───────────────────────────────────────────────────────────── */
.stExpander {{
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-sm) !important;
}}
.stExpander header {{
  background: var(--card) !important;
  border-radius: var(--radius) !important;
  color: var(--text) !important;
  font-weight: 600 !important;
}}

/* ── Dataframes & Tables ─────────────────────────────────────────────────── */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div,
.dvn-scroller {{
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  overflow: hidden !important;
  background: var(--card) !important;
  color: var(--text) !important;
  box-shadow: var(--shadow-sm) !important;
}}

/* ── Status & Strategy Badges ────────────────────────────────────────────── */
.badge {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 3px 10px;
  border-radius: var(--radius-sm);
  font-size: 11.5px;
  font-weight: 600;
  letter-spacing: 0.02em;
  line-height: 1.4;
}}
.badge-vector {{
  background: var(--vector-bg);
  color: var(--vector);
  border: 1px solid var(--vector-border);
}}
.badge-graph {{
  background: var(--graph-bg);
  color: var(--graph);
  border: 1px solid var(--graph-border);
}}
.badge-hybrid {{
  background: var(--hybrid-bg);
  color: var(--hybrid);
  border: 1px solid var(--hybrid-border);
}}
.badge-success {{
  background: var(--success-bg);
  color: var(--success);
  border: 1px solid var(--success-border);
}}
.badge-warning {{
  background: var(--warning-bg);
  color: var(--warning);
  border: 1px solid var(--warning-border);
}}
.badge-error {{
  background: var(--error-bg);
  color: var(--error);
  border: 1px solid var(--error-border);
}}

/* ── Chat Messages & Input ───────────────────────────────────────────────── */
.chat-user {{
  background: var(--accent-light);
  border: 1px solid var(--accent-border);
  border-radius: 14px 14px 4px 14px;
  padding: 14px 18px;
  margin: 8px 0;
  max-width: 82%;
  margin-left: auto;
  font-size: 14px;
  color: var(--text);
  line-height: 1.6;
  box-shadow: var(--shadow-sm);
}}
.chat-assistant {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px 14px 14px 4px;
  padding: 16px 20px;
  margin: 8px 0;
  max-width: 88%;
  font-size: 14px;
  color: var(--text);
  line-height: 1.65;
  box-shadow: var(--shadow-sm);
}}

div[data-testid="stChatInput"] {{
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  background: var(--card) !important;
  box-shadow: var(--shadow-md) !important;
  transition: all 0.2s ease !important;
}}
div[data-testid="stChatInput"]:focus-within {{
  border-color: var(--accent) !important;
  box-shadow: 0 0 0 2px var(--accent-light), var(--shadow-md) !important;
}}
div[data-testid="stChatInput"] textarea {{
  color: var(--text) !important;
  background: transparent !important;
  font-family: 'Plus Jakarta Sans', 'Inter', sans-serif !important;
  font-size: 14px !important;
}}
div[data-testid="stChatInput"] button {{
  color: var(--accent) !important;
}}

/* ── Native Code & Pre styling ───────────────────────────────────────────── */
code, pre {{
  font-family: 'JetBrains Mono', monospace !important;
  font-size: 12.5px !important;
  border-radius: var(--radius-sm) !important;
}}
p code, li code, span code {{
  background: var(--card-hi) !important;
  color: var(--accent) !important;
  border: 1px solid var(--border) !important;
  padding: 2px 6px !important;
}}
pre {{
  background: var(--card-inner) !important;
  border: 1px solid var(--border) !important;
  padding: 14px !important;
  color: var(--text) !important;
}}

/* ── Popover & Dialog ────────────────────────────────────────────────────── */
div[data-testid="stPopoverBody"] {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius) !important;
  box-shadow: var(--shadow-lg) !important;
  color: var(--text) !important;
  padding: 16px !important;
}}

div[data-testid="stModal"] > div,
div[data-testid="stDialog"] > div {{
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: var(--radius-lg) !important;
  box-shadow: var(--shadow-lg) !important;
  color: var(--text) !important;
}}

div[data-testid="stAlert"] {{
  border-radius: var(--radius) !important;
  border: 1px solid var(--border) !important;
  background: var(--card-hi) !important;
  color: var(--text) !important;
}}

/* ── Progress Bar ────────────────────────────────────────────────────────── */
div[data-testid="stProgress"] > div {{
  background: var(--card-hi) !important;
  border-radius: var(--radius-full) !important;
}}
div[data-testid="stProgress"] > div > div {{
  background: linear-gradient(90deg, var(--accent) 0%, var(--accent2) 100%) !important;
  border-radius: var(--radius-full) !important;
}}

/* ── Stat Metric Cards ───────────────────────────────────────────────────── */
.stat-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 20px 22px;
  text-align: center;
  box-shadow: var(--shadow-sm);
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}}
.stat-card:hover {{
  border-color: var(--accent-border);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}}
.stat-value {{
  font-size: 28px;
  font-weight: 800;
  color: var(--text);
  letter-spacing: -0.03em;
  line-height: 1.1;
}}
.stat-label {{
  font-size: 11.5px;
  font-weight: 600;
  color: var(--text-muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
  margin-top: 6px;
}}

/* ── Architecture Pipeline Flow ──────────────────────────────────────────── */
.arch-card-wrap {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 28px 30px;
  margin: 0 auto;
  box-shadow: var(--shadow-sm);
}}
.arch-header-wrap {{
  text-align: center;
  margin-bottom: 24px;
}}
.arch-tag {{
  font-size: 11px;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--accent);
}}
.arch-title {{
  font-size: 20px;
  font-weight: 800;
  color: var(--text);
  margin: 4px 0 0 0;
  letter-spacing: -0.025em;
}}
.arch-pipeline {{
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 10px;
  width: 100%;
  overflow-x: auto;
  padding: 8px 4px;
}}
.arch-node {{
  background: var(--card-hi);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 16px 12px;
  text-align: center;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 128px;
  box-sizing: border-box;
  flex: 1;
  min-width: 110px;
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
}}
.arch-node:hover {{
  border-color: var(--accent-border);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}}
.arch-node-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: var(--radius-sm);
  margin-bottom: 8px;
}}
.arch-node-name {{
  font-size: 12px;
  font-weight: 700;
  letter-spacing: 0.02em;
  color: var(--text);
}}
.arch-node-sub {{
  font-size: 10px;
  color: var(--text-muted);
  margin-top: 3px;
}}
.arch-arrow {{
  display: flex;
  align-items: center;
  justify-content: center;
  color: var(--border);
  flex: 0 0 18px;
}}

/* ── Retrieval Architecture Cards ────────────────────────────────────────── */
.retrieval-card {{
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: var(--radius-lg);
  padding: 24px 22px;
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  min-height: 200px;
  box-shadow: var(--shadow-sm);
  transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1);
  box-sizing: border-box;
}}
.retrieval-card:hover {{
  border-color: var(--accent-border);
  transform: translateY(-2px);
  box-shadow: var(--shadow-md);
}}
.retrieval-card-top {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 12px;
}}
.retrieval-card-title-group {{
  display: flex;
  align-items: center;
  gap: 10px;
}}
.retrieval-card-icon {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: var(--radius-sm);
}}
.retrieval-card-title {{
  font-size: 16px;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -0.01em;
}}
.retrieval-card-desc {{
  font-size: 13.5px;
  color: var(--text-muted);
  line-height: 1.55;
  margin-bottom: 0;
}}

/* ── File Uploader Enhancement ───────────────────────────────────────────── */
[data-testid="stFileUploader"] {{
  background: var(--card) !important;
  border: 2px dashed var(--border) !important;
  border-radius: var(--radius-lg) !important;
  padding: 20px !important;
  transition: all 0.2s ease !important;
}}
[data-testid="stFileUploader"]:hover {{
  border-color: var(--accent) !important;
  background: var(--card-hi) !important;
}}
[data-testid="stFileUploader"] section {{
  background: transparent !important;
}}

/* ── Responsive Rules ────────────────────────────────────────────────────── */
@media (max-width: 960px) {{
  .arch-pipeline {{
    flex-wrap: wrap;
    gap: 8px;
  }}
  .arch-arrow {{
    display: none;
  }}
  .arch-node {{
    flex: 0 0 calc(50% - 8px);
  }}
}}
</style>
<script>
// Browser localStorage sync for cross-refresh persistence
(function() {{
  try {{
    var theme = '{theme}';
    localStorage.setItem('rag_benchmark_lab_theme', theme);
    var url = new URL(window.location.href);
    if (url.searchParams.get('theme') !== theme) {{
      url.searchParams.set('theme', theme);
      window.history.replaceState(null, '', url.toString());
    }}
  }} catch (e) {{}}
}})();
</script>
"""


def inject_custom_css() -> None:
    """Inject the unified design system CSS based on the current persistent theme."""
    theme = get_current_theme()
    css_content = _generate_theme_css(theme)
    if hasattr(st, "html"):
        st.html(css_content)
    else:
        st.markdown(css_content, unsafe_allow_html=True)


def render_html(html_str: str) -> None:
    """Render raw HTML safely without markdown escaping."""
    if hasattr(st, "html"):
        st.html(html_str)
    else:
        st.markdown(html_str, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3. NATIVE TOP NAVIGATION BAR
# ─────────────────────────────────────────────────────────────────────────────

def _safe_page_link(page: str, label: str, icon: str | None = None, help: str | None = None) -> None:
    """Render a native page link with resilient fallback."""
    try:
        st.page_link(page, label=label, icon=icon, help=help)
    except Exception:
        alt = page[len("pages/"):] if page.startswith("pages/") else f"pages/{page}"
        try:
            st.page_link(alt, label=label, icon=icon, help=help)
        except Exception:
            if st.button(label, icon=icon, help=help, use_container_width=True):
                try:
                    st.switch_page(page)
                except Exception:
                    st.switch_page(alt)


def render_topnav(active_page: str = "Home") -> None:
    """
    Render a clean, native top navigation bar using Streamlit's official routing components.

    Features:
      LEFT: Brand mark + 'RAG Benchmark Lab' + 'LAB' badge
      CENTER: Home, Documents, Chat, Compare, Evaluation, History
      RIGHT:
        - Light/Dark theme toggle button with sun/moon icon
        - Active KB selector indicator
        - User profile popover with settings and sign-out
    """
    inject_custom_css()

    theme = get_current_theme()
    from rag_benchmark_lab.utils.auth import get_current_user_name, is_authenticated

    user_name = get_current_user_name()
    is_auth = is_authenticated()
    active_kb_name = st.session_state.get("active_kb_name", "")

    # Inject specific active page accent style
    active_slug = active_page.lower().strip()
    accent_color = "#6366F1" if theme == "dark" else "#4F46E5"
    active_bg = "rgba(99, 102, 241, 0.14)" if theme == "dark" else "rgba(79, 70, 229, 0.10)"
    active_border = "rgba(99, 102, 241, 0.35)" if theme == "dark" else "rgba(79, 70, 229, 0.25)"

    active_css = f"""
    <style>
    .st-key-nav_item_{active_slug} [data-testid="stPageLink"] a {{
        color: {accent_color} !important;
        background: {active_bg} !important;
        border: 1px solid {active_border} !important;
        font-weight: 700 !important;
    }}
    .st-key-nav_item_{active_slug} [data-testid="stPageLink"] span[data-testid="stIconMaterial"] {{
        color: {accent_color} !important;
    }}
    </style>
    """
    if hasattr(st, "html"):
        st.html(active_css)
    else:
        st.markdown(active_css, unsafe_allow_html=True)

    with st.container(key="rag_topnav"):
        c_brand, c_nav, c_right = st.columns([2.0, 5.5, 2.5], vertical_alignment="center")

        with c_brand:
            with st.container(key="nav_brand_col"):
                _safe_page_link(
                    "pages/1_Home.py",
                    label="RAG Benchmark Lab",
                    icon=":material/hub:",
                )

        with c_nav:
            n1, n2, n3, n4, n5, n6 = st.columns(6, gap="small", vertical_alignment="center")
            with n1:
                with st.container(key="nav_item_home"):
                    _safe_page_link("pages/1_Home.py", label="Home", icon=":material/home:")
            with n2:
                with st.container(key="nav_item_documents"):
                    _safe_page_link("pages/2_Documents.py", label="Documents", icon=":material/description:")
            with n3:
                with st.container(key="nav_item_chat"):
                    _safe_page_link("pages/3_Chat.py", label="Chat", icon=":material/chat:")
            with n4:
                with st.container(key="nav_item_compare"):
                    _safe_page_link("pages/4_Compare.py", label="Compare", icon=":material/compare_arrows:")
            with n5:
                with st.container(key="nav_item_evaluation"):
                    _safe_page_link("pages/5_Evaluation.py", label="Evaluation", icon=":material/assessment:")
            with n6:
                with st.container(key="nav_item_history"):
                    _safe_page_link("pages/6_History.py", label="History", icon=":material/history:")

        with c_right:
            # Theme toggle on the right
            if is_auth:
                r_theme, r_kb, r_user = st.columns([1.0, 1.2, 1.2], gap="small", vertical_alignment="center")
                with r_theme:
                    with st.container(key="nav_theme_toggle"):
                        if theme == "dark":
                            if st.button("Light", icon=":material/light_mode:", help="Switch to Light Mode", key=f"btn_theme_{active_slug}", use_container_width=True):
                                toggle_theme()
                        else:
                            if st.button("Dark", icon=":material/dark_mode:", help="Switch to Dark Mode", key=f"btn_theme_{active_slug}", use_container_width=True):
                                toggle_theme()
                with r_kb:
                    if active_kb_name:
                        display_kb = active_kb_name if len(active_kb_name) <= 12 else active_kb_name[:10] + "…"
                        _safe_page_link(
                            "pages/2_Documents.py",
                            label=display_kb,
                            icon=":material/folder_open:",
                            help=f"Active Knowledge Base: {active_kb_name}",
                        )
                    else:
                        _safe_page_link(
                            "pages/2_Documents.py",
                            label="Select KB",
                            icon=":material/folder:",
                            help="Select or create a knowledge base",
                        )
                with r_user:
                    short_name = user_name.split()[0] if user_name else "Account"
                    with st.popover(short_name, icon=":material/account_circle:", use_container_width=True):
                        st.caption("Signed in as")
                        st.markdown(f"**{user_name}**")
                        if active_kb_name:
                            st.caption(f"Knowledge Base: **{active_kb_name}**")
                        _safe_page_link("pages/7_Settings.py", label="Settings & Diagnostics", icon=":material/settings:")
                        st.markdown("<hr style='border-color:var(--border);margin:8px 0;'/>", unsafe_allow_html=True)
                        if st.button("Sign Out", use_container_width=True, key=f"nav_logout_btn_{active_slug}"):
                            from rag_benchmark_lab.utils.auth import logout_user
                            logout_user()
                            st.rerun()
            else:
                r_theme, r_in, r_reg = st.columns([0.9, 0.9, 1.2], gap="small", vertical_alignment="center")
                with r_theme:
                    with st.container(key="nav_theme_toggle"):
                        if theme == "dark":
                            if st.button("Light", icon=":material/light_mode:", help="Switch to Light Mode", key=f"btn_theme_{active_slug}", use_container_width=True):
                                toggle_theme()
                        else:
                            if st.button("Dark", icon=":material/dark_mode:", help="Switch to Dark Mode", key=f"btn_theme_{active_slug}", use_container_width=True):
                                toggle_theme()
                with r_in:
                    with st.container(key="nav_auth_signin"):
                        _safe_page_link("pages/0_Login.py", label="Sign In", icon=":material/login:")
                with r_reg:
                    with st.container(key="nav_auth_reg"):
                        _safe_page_link("pages/0_Login.py", label="Create Account", icon=":material/person_add:")


# ─────────────────────────────────────────────────────────────────────────────
# 4. PLOTLY CHART THEMING
# ─────────────────────────────────────────────────────────────────────────────

def get_plotly_layout(theme: str | None = None) -> dict[str, Any]:
    """Return a dictionary of Plotly layout properties matching the active theme."""
    if theme is None:
        theme = get_current_theme()
    is_dark = (theme == "dark")

    bg_color = "#11141D" if is_dark else "#FFFFFF"
    text_color = "#F8FAFC" if is_dark else "#0F172A"
    muted_color = "#94A3B8" if is_dark else "#64748B"
    grid_color = "#1E2433" if is_dark else "#E2E8F0"

    return dict(
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=text_color, family="Plus Jakarta Sans, Inter, sans-serif", size=12),
        xaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(color=muted_color),
            titlefont=dict(color=text_color),
        ),
        yaxis=dict(
            gridcolor=grid_color,
            zerolinecolor=grid_color,
            tickfont=dict(color=muted_color),
            titlefont=dict(color=text_color),
        ),
        legend=dict(
            font=dict(color=text_color),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=24, r=24, t=44, b=24),
    )


def apply_plotly_theme(fig: Any, theme: str | None = None) -> Any:
    """Apply theme-adaptive styling to a Plotly figure."""
    fig.update_layout(**get_plotly_layout(theme))
    return fig


# Compatibility layout objects for legacy imports
DARK_LAYOUT = get_plotly_layout("dark")
LIGHT_LAYOUT = get_plotly_layout("light")


# ─────────────────────────────────────────────────────────────────────────────
# 5. REUSABLE COMPONENT HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_dev_mode_banner(is_mock: bool = True) -> None:
    """Development mode notice (passive)."""
    pass


def render_strategy_badge(strategy_name: str) -> str:
    """Return inline HTML badge for a retrieval strategy with theme tokens."""
    s = str(strategy_name).upper()
    if "VECTOR" in s:
        return '<span class="badge badge-vector">Vector RAG</span>'
    if "GRAPH" in s:
        return '<span class="badge badge-graph">GraphRAG</span>'
    if "HYBRID" in s:
        return '<span class="badge badge-hybrid">Hybrid RAG</span>'
    return f'<span class="badge badge-vector">{s}</span>'


def render_status_badge(status: str) -> str:
    """Return inline HTML badge for an operational status."""
    st_clean = str(status).lower()
    if st_clean in ("ready", "indexed", "success", "completed"):
        return f'<span class="badge badge-success">{status.capitalize()}</span>'
    if st_clean in ("partial", "pending", "indexing"):
        return f'<span class="badge badge-warning">{status.capitalize()}</span>'
    return f'<span class="badge badge-error">{status.capitalize()}</span>'


def render_stat_cards(stats: list[tuple[str, str, str | None]]) -> None:
    """Render a responsive row of elegant SaaS metric cards."""
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


def render_metric_card(label: str, value: Any, subtitle: str | None = None, badge: str | None = None) -> None:
    """Render a single high-impact metric card."""
    badge_html = f'<span class="badge badge-success" style="font-size:10px;">{badge}</span>' if badge else ""
    sub_html = f'<div style="font-size:12px;color:var(--text-muted);margin-top:4px;">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f"""
        <div class="stat-card" style="text-align:left;padding:18px 20px;">
          <div style="display:flex;justify-content:space-between;align-items:center;">
            <span class="stat-label" style="margin-top:0;">{label}</span>
            {badge_html}
          </div>
          <div class="stat-value" style="margin-top:8px;">{value}</div>
          {sub_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state(title: str, description: str, action_label: str | None = None) -> None:
    """Render an elegant empty-state container."""
    st.markdown(
        f"""
        <div class="rag-card" style="text-align:center;padding:48px 24px;margin:16px 0;">
          <div style="display:inline-flex;align-items:center;justify-content:center;width:48px;height:48px;border-radius:12px;background:var(--card-hi);border:1px solid var(--border);margin-bottom:14px;">
            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="var(--accent)" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <circle cx="12" cy="12" r="10"></circle>
              <line x1="12" y1="8" x2="12" y2="12"></line>
              <line x1="12" y1="16" x2="12.01" y2="16"></line>
            </svg>
          </div>
          <h4 style="font-size:17px;font-weight:700;color:var(--text);margin:0 0 8px 0;">{title}</h4>
          <p style="font-size:13.5px;color:var(--text-muted);max-width:460px;margin:0 auto;line-height:1.5;">{description}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_kb_selector(context: object, user_id: str, key_prefix: str = "kb") -> str | None:
    """Inline KB selector used on pages that need the active knowledge base."""
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
    """Section header with title, subtitle, and optional badge."""
    inject_custom_css()
    badge_html = (
        f'<span class="badge" style="background:var(--accent-light);color:var(--accent);'
        f'border:1px solid var(--accent-border);">'
        f'{badge_text}</span>'
        if badge_text else ""
    )
    st.markdown(
        f"""
        <div style="margin-bottom:20px;">
          <div style="font-size:24px;font-weight:800;letter-spacing:-0.03em;
                      color:var(--text);display:flex;align-items:center;gap:10px;flex-wrap:wrap;">
            {title} {badge_html}
          </div>
          <div style="font-size:13.5px;color:var(--text-muted);margin-top:6px;">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_latency(
    val: float | int | None,
    fallback: str = "Unavailable",
    unit: str = "ms",
    precision: int = 0,
    treat_zero_as_unavailable: bool = False,
) -> str:
    """Safely format an optional numeric latency value into a display string."""
    if val is None:
        return fallback
    if not isinstance(val, (int, float)):
        try:
            val = float(val)
        except (ValueError, TypeError):
            return fallback
    if math.isnan(val) or math.isinf(val):
        return fallback
    if treat_zero_as_unavailable and val <= 0:
        return fallback
    if unit:
        return f"{val:.{precision}f} {unit}".strip()
    return f"{val:.{precision}f}"


def format_metric(
    val: float | int | None,
    fallback: str = "N/A",
    precision: int = 3,
) -> str:
    """Safely format an optional numeric metric value into a display string."""
    if val is None:
        return fallback
    if not isinstance(val, (int, float)):
        try:
            val = float(val)
        except (ValueError, TypeError):
            return fallback
    if math.isnan(val) or math.isinf(val):
        return fallback
    return f"{val:.{precision}f}"
