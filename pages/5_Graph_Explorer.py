from __future__ import annotations

import math

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context


st.set_page_config(page_title="Graph Explorer - RAG Benchmark Lab", layout="wide")
context = get_context()

st.title("Knowledge Graph Explorer")
entities = context.registry.list_entities()
relationships = context.registry.list_relationships()

if not entities:
    st.info("No graph data available yet.")
else:
    entity_lookup = {entity["entity_id"]: entity for entity in entities}
    selected_entity_id = st.selectbox("Inspect entity", [entity["entity_id"] for entity in entities], format_func=lambda entity_id: entity_lookup[entity_id]["canonical_name"])
    selected_entity = entity_lookup[selected_entity_id]
    st.write(selected_entity)

    filtered_relationships = [relationship for relationship in relationships if relationship["source_entity_id"] == selected_entity_id or relationship["target_entity_id"] == selected_entity_id]
    node_ids = {selected_entity_id}
    for relationship in filtered_relationships:
        node_ids.add(relationship["source_entity_id"])
        node_ids.add(relationship["target_entity_id"])

    figure = go.Figure()
    node_positions: dict[str, tuple[float, float]] = {}
    radius = 1.0
    for index, entity_id in enumerate(node_ids):
        angle = 2 * math.pi * index / max(1, len(node_ids))
        node_positions[entity_id] = (radius * math.cos(angle), radius * math.sin(angle))
    for relationship in filtered_relationships:
        x0, y0 = node_positions[relationship["source_entity_id"]]
        x1, y1 = node_positions[relationship["target_entity_id"]]
        figure.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(width=1, color="#999"), hoverinfo="none", showlegend=False))
    for entity_id in node_ids:
        x, y = node_positions[entity_id]
        cname = entity_lookup.get(entity_id, {}).get("canonical_name", entity_id)
        etype = entity_lookup.get(entity_id, {}).get("entity_type", "Concept")
        figure.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", text=[cname], textposition="top center", marker=dict(size=20), name=etype))
    figure.update_layout(height=600, margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
    st.plotly_chart(figure, use_container_width=True)
    st.dataframe(pd.DataFrame(filtered_relationships), use_container_width=True)
