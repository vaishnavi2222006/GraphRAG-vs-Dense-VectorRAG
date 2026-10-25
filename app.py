"""RAG Benchmark Lab — Root Entry Point.

Redirects directly to the new Home workspace (pages/1_Home.py).
"""
import streamlit as st

try:
    st.switch_page("pages/1_Home.py")
except Exception:
    # Fallback if switch_page is called during initial boot
    from rag_benchmark_lab.app import main
    main()
