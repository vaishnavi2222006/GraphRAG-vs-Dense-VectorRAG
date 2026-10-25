from __future__ import annotations

import math
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from rag_benchmark_lab.app import get_context
from rag_benchmark_lab.utils.auth import get_current_user_name, require_auth
from rag_benchmark_lab.utils.ui import inject_custom_css, render_dev_mode_banner, render_lab_header

st.set_page_config(page_title="Knowledge Graph - RAG Benchmark Lab", page_icon="🕸️", layout="wide")
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
            key="graph_kb_selector",
        )
        if selected_kb_id != active_kb_id:
            st.session_state["active_kb_id"] = selected_kb_id
            st.rerun()
        active_kb_id = selected_kb_id
    else:
        active_kb_id = None
        st.info("No knowledge bases yet.")
    st.markdown("---")
    if st.button("🚪 Logout", use_container_width=True, key="graph_logout"):
        from rag_benchmark_lab.utils.auth import logout_user
        logout_user()
        st.rerun()

render_lab_header(
    title="Neo4j Knowledge Graph Explorer",
    subtitle="Inspect real entity nodes, relational edges, multi-hop neighborhoods, and ground-truth text provenance.",
    badge_text="GRAPH EXPLORATION",
)

render_dev_mode_banner(context.settings.llm_mock_mode)

# Check Neo4j connectivity and fetch real data
neo4j_available = False
neo4j_err = None
try:
    context.neo4j_store.health()
    neo4j_available = True
    total_nodes = context.neo4j_store.count_entities()
    total_edges = context.neo4j_store.count_relationships()
    total_docs = context.neo4j_store.count_documents()
except Exception as exc:
    neo4j_available = False
    neo4j_err = str(exc)
    total_nodes = 0
    total_edges = 0
    total_docs = 0

# If Neo4j is offline, display explicit required status
if not neo4j_available:
    st.error(f"❌ **Neo4j: Unavailable** — Could not establish Bolt connection to `bolt://localhost:7687` ({neo4j_err})")
    st.markdown(
        """
        <div style="background: #fff1f2; border: 1px solid #fecdd3; border-left: 4px solid #e11d48; padding: 12px 16px; border-radius: 6px; font-size: 13px; color: #9f1239; margin-bottom: 16px;">
            <strong>Database Offline:</strong> The production Neo4j container is not running or unreachable.
            To start Neo4j, execute: <code>docker compose up -d neo4j</code> in your terminal.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Check if local SQLite registry has extracted graph provenance to inspect
    reg_counts = context.registry.counts()
    if reg_counts.get("entities", 0) > 0:
        st.info(f"ℹ️ Displaying extracted knowledge graph provenance from durable SQLite document registry ({reg_counts['entities']} entities, {reg_counts['relationships']} relationships).")
        source_mode = "registry"
        total_nodes = reg_counts["entities"]
        total_edges = reg_counts["relationships"]
        total_docs = reg_counts["documents"]
    else:
        st.info("🕸️ The Knowledge Graph is currently empty. Upload documents in **Corpus Management** to extract entities and build relationships.")
        st.stop()
else:
    source_mode = "neo4j"
    if total_nodes == 0:
        st.info("🕸️ The Neo4j Knowledge Graph is currently empty. Upload documents in **Corpus Management** to extract entities and build relationships.")
        st.stop()

# Graph Telemetry Bar
mcol1, mcol2, mcol3, mcol4 = st.columns(4)
with mcol1:
    st.metric("Total Graph Entities", total_nodes)
with mcol2:
    st.metric("Total Relationships", total_edges)
with mcol3:
    st.metric("Indexed Documents", total_docs)
with mcol4:
    st.metric("Graph Store Backend", "Neo4j 5.23 (Active)" if neo4j_available else "Local Registry (Neo4j Offline)")

st.markdown("---")

# Filters & Controls
st.markdown("### 🔍 Search & Filter Graph")
filter_col1, filter_col2, filter_col3, filter_col4 = st.columns([1, 1, 1, 1.5])

# Fetch all entities to populate filters
if source_mode == "neo4j":
    all_entities = context.neo4j_store.list_entities(limit=300)
    all_rels = context.neo4j_store.list_relationships(limit=500)
else:
    all_entities = context.registry.list_entities()
    all_rels = context.registry.list_relationships()

docs = context.registry.list_documents()
doc_options = ["All Documents"] + [d["document_name"] for d in docs]

entity_types = ["All"] + sorted(list({e.get("entity_type", "Concept") for e in all_entities if e.get("entity_type")}))
rel_types = ["All"] + sorted(list({r.get("relationship_type", "RELATED_TO") for r in all_rels if r.get("relationship_type")}))

with filter_col1:
    selected_type = st.selectbox("Filter by Entity Type", options=entity_types, index=0)
with filter_col2:
    selected_rel = st.selectbox("Filter by Relationship Type", options=rel_types, index=0)
with filter_col3:
    selected_doc = st.selectbox("Filter by Source Document", options=doc_options, index=0)
with filter_col4:
    search_query = st.text_input("Search Entity Name", placeholder="e.g. Qdrant, Neo4j, Hybrid RAG...")

# Filter entities according to selection
if source_mode == "neo4j":
    filtered_entities = context.neo4j_store.list_entities(
        entity_type=selected_type if selected_type != "All" else None,
        search=search_query if search_query else None,
        limit=200,
    )
else:
    filtered_entities = all_entities
    if selected_type != "All":
        filtered_entities = [e for e in filtered_entities if e.get("entity_type") == selected_type]
    if search_query:
        filtered_entities = [e for e in filtered_entities if search_query.lower() in e.get("canonical_name", "").lower()]
    if selected_doc != "All Documents":
        doc_match = next((d["document_id"] for d in docs if d["document_name"] == selected_doc), None)
        if doc_match:
            filtered_entities = [e for e in filtered_entities if e.get("document_id") == doc_match]

if not filtered_entities:
    st.warning("No entities match the current filters.")
    st.stop()

entity_map = {e["entity_id"]: e for e in filtered_entities}
entity_ids = list(entity_map.keys())

# Select primary entity to inspect neighborhood
st.markdown("### 🎯 Inspect Entity Neighborhood")
sel_col1, sel_col2 = st.columns([2, 1])
with sel_col1:
    selected_entity_id = st.selectbox(
        "Focus on Entity (Center Node)",
        options=entity_ids,
        format_func=lambda eid: f"{entity_map[eid]['canonical_name']} ({entity_map[eid].get('entity_type', 'Concept')})",
    )
with sel_col2:
    max_hops = st.slider("Neighborhood Hops", min_value=1, max_value=3, value=1)

selected_entity = entity_map[selected_entity_id]

# Fetch neighborhood relationships
rel_filter = selected_rel if selected_rel != "All" else None
if source_mode == "neo4j":
    neighborhood_rels = context.neo4j_store.list_relationships(relationship_type=rel_filter, entity_id=selected_entity_id, limit=100)
else:
    neighborhood_rels = [
        r for r in all_rels
        if (r.get("source_entity_id") == selected_entity_id or r.get("target_entity_id") == selected_entity_id)
        and (rel_filter is None or r.get("relationship_type") == rel_filter)
    ]

# Build visualization nodes and edges
node_ids = {selected_entity_id}
for r in neighborhood_rels:
    node_ids.add(r["source_entity_id"])
    node_ids.add(r["target_entity_id"])

# Fetch missing nodes in neighborhood if not already in entity_map
missing_nodes = node_ids - set(entity_map.keys())
if missing_nodes:
    if source_mode == "neo4j":
        extra_entities = context.neo4j_store.search_entities(list(missing_nodes), limit=50)
        for ee in extra_entities:
            entity_map[ee.entity_id] = asdict(ee)
    else:
        for ee in all_entities:
            if ee.get("entity_id") in missing_nodes:
                entity_map[ee["entity_id"]] = ee

# Color palette for entity types
TYPE_COLORS = {
    "Technology": "#0284c7",    # blue
    "Architecture": "#7c3aed",  # violet
    "Component": "#0d9488",     # teal
    "Algorithm": "#d97706",     # amber
    "DataStructure": "#4f46e5", # indigo
    "Metric": "#dc2626",        # red
    "System": "#16a34a",        # green
    "Concept": "#64748b",       # slate
}

# Layout nodes in circular/radial pattern with focal center
figure = go.Figure()
node_positions: dict[str, tuple[float, float]] = {}

# Center focal entity at (0, 0)
node_positions[selected_entity_id] = (0.0, 0.0)
peripheral_nodes = [nid for nid in node_ids if nid != selected_entity_id]

radius = 2.0
for index, entity_id in enumerate(peripheral_nodes):
    angle = 2 * math.pi * index / max(1, len(peripheral_nodes))
    node_positions[entity_id] = (radius * math.cos(angle), radius * math.sin(angle))

# Draw relationship edges
edge_x = []
edge_y = []
edge_labels = []

for r in neighborhood_rels:
    src_id = r["source_entity_id"]
    tgt_id = r["target_entity_id"]
    if src_id in node_positions and tgt_id in node_positions:
        x0, y0 = node_positions[src_id]
        x1, y1 = node_positions[tgt_id]
        figure.add_trace(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(width=1.8, color="#94a3b8"),
                hoverinfo="text",
                text=f"{r.get('source_entity_name')} &rarr; {r.get('relationship_type')} &rarr; {r.get('target_entity_name')}",
                showlegend=False,
            )
        )
        # Add midpoint annotation for relationship label
        mid_x = (x0 + x1) / 2
        mid_y = (y0 + y1) / 2
        figure.add_trace(
            go.Scatter(
                x=[mid_x],
                y=[mid_y],
                mode="text",
                text=[r.get("relationship_type", "")],
                textposition="middle center",
                textfont=dict(size=9, color="#64748b"),
                hoverinfo="none",
                showlegend=False,
            )
        )

# Draw nodes grouped by type for legend
rendered_types = set()
for entity_id in node_ids:
    if entity_id not in node_positions:
        continue
    x, y = node_positions[entity_id]
    ent = entity_map.get(entity_id, {})
    cname = ent.get("canonical_name", entity_id[:8])
    etype = ent.get("entity_type", "Concept")
    color = TYPE_COLORS.get(etype, "#64748b")
    size = 28 if entity_id == selected_entity_id else 18
    show_leg = etype not in rendered_types
    rendered_types.add(etype)

    figure.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            text=[cname],
            textposition="top center",
            textfont=dict(size=11, color="#0f172a"),
            marker=dict(size=size, color=color, line=dict(width=2, color="#ffffff")),
            name=etype,
            hoverinfo="text",
            hovertext=f"<b>{cname}</b><br/>Type: {etype}<br/>ID: {entity_id[:12]}...",
            showlegend=show_leg,
        )
    )

figure.update_layout(
    height=550,
    margin=dict(l=20, r=20, t=30, b=20),
    plot_bgcolor="#f8fafc",
    paper_bgcolor="#f8fafc",
    xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

st.plotly_chart(figure, use_container_width=True)

# Details & Ground-Truth Provenance
st.markdown("### 📋 Entity Details & Ground-Truth Text Provenance")
tab_entity_info, tab_relations = st.tabs(["Selected Entity Profile", f"Incident Relationships ({len(neighborhood_rels)})"])

with tab_entity_info:
    det_cols = st.columns(3)
    with det_cols[0]:
        st.markdown(f"**Canonical Name:** `{selected_entity['canonical_name']}`")
        st.markdown(f"**Entity Type:** `{selected_entity.get('entity_type', 'Concept')}`")
    with det_cols[1]:
        st.markdown(f"**Entity ID:** `{selected_entity['entity_id']}`")
        st.markdown(f"**Confidence:** `{selected_entity.get('confidence', 0.85)}`")
    with det_cols[2]:
        st.markdown(f"**Extraction Method:** `{selected_entity.get('extraction_method', 'heuristic')}`")
        st.markdown(f"**Source Document:** `{selected_entity.get('document_id') or 'General'}`")

with tab_relations:
    if neighborhood_rels:
        df_display_rels = []
        for r in neighborhood_rels:
            df_display_rels.append(
                {
                    "Source Entity": r["source_entity_name"],
                    "Relationship": r["relationship_type"],
                    "Target Entity": r["target_entity_name"],
                    "Confidence": r.get("confidence", 0.8),
                    "Extraction Method": r.get("extraction_method", "heuristic"),
                    "Source Text Evidence": r.get("source_text") or "N/A",
                }
            )
        st.dataframe(pd.DataFrame(df_display_rels), use_container_width=True)
    else:
        st.info("No incident relationships for this entity under current filters.")
