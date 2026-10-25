from __future__ import annotations

import streamlit as st

from rag_benchmark_lab.app import get_context, render_overview_page

# page config handled in main app
context = get_context()
render_overview_page(context)
