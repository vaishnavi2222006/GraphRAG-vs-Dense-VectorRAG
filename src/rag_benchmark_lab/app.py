from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_benchmark_lab.config.settings import AppSettings, get_settings
from rag_benchmark_lab.embeddings.embedding_service import EmbeddingService
from rag_benchmark_lab.evaluation.result_store import ResultStore
from rag_benchmark_lab.generation.llm_service import OllamaLLMService
from rag_benchmark_lab.graph_store.entity_extractor import EntityRelationshipExtractor
from rag_benchmark_lab.graph_store.graph_retriever import GraphRetriever
from rag_benchmark_lab.graph_store.neo4j_store import Neo4jStore
from rag_benchmark_lab.retrieval.hybrid_retriever import HybridRetriever
from rag_benchmark_lab.retrieval.vector_retriever import VectorRetriever
from rag_benchmark_lab.services.document_registry import DocumentRegistry
from rag_benchmark_lab.services.document_service import DocumentService
from rag_benchmark_lab.services.health_service import HealthService
from rag_benchmark_lab.services.query_service import QueryService
from rag_benchmark_lab.services.user_registry import UserRegistry
from rag_benchmark_lab.utils.ui import inject_custom_css, render_dev_mode_banner, render_lab_header
from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore


@dataclass(slots=True)
class AppContext:
    settings: AppSettings
    registry: DocumentRegistry
    user_registry: UserRegistry
    embedding_service: EmbeddingService
    qdrant_store: QdrantStore
    neo4j_store: Neo4jStore
    llm_service: OllamaLLMService
    extractor: EntityRelationshipExtractor
    document_service: DocumentService
    vector_retriever: VectorRetriever
    graph_retriever: GraphRetriever
    hybrid_retriever: HybridRetriever
    query_service: QueryService
    result_store: ResultStore
    health_service: HealthService


def build_context() -> AppContext:
    settings = get_settings()
    registry = DocumentRegistry(settings.sqlite_path)
    # User registry uses the same SQLite DB path but a separate file to avoid schema conflicts
    user_db_path = settings.sqlite_path.parent / "users.sqlite3"
    user_registry = UserRegistry(user_db_path)
    embedding_service = EmbeddingService(settings.embedding_model, mock_mode=settings.llm_mock_mode)

    # 1. Qdrant client adapter: try configured URL, fallback to local in-memory
    qdrant_store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)
    try:
        qdrant_store.health()
        qdrant_store.ensure_collection(embedding_service.dimension)
    except Exception:
        # Docker Qdrant is unavailable; use in-memory Qdrant instance so Vector RAG works smoothly
        from rag_benchmark_lab.vector_store.qdrant_store import QdrantStore as MemQdrant
        qdrant_store = MemQdrant(":memory:", None, settings.qdrant_collection)
        qdrant_store.ensure_collection(embedding_service.dimension)

    # 2. Neo4j graph store adapter: try Bolt connection, fallback to NetworkX graph store
    neo4j_store = Neo4jStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
    try:
        neo4j_store.health()
        neo4j_store.ensure_schema()
    except Exception:
        # Docker Neo4j is unavailable; use NetworkX in-memory graph store synced with SQLite registry
        from rag_benchmark_lab.graph_store.networkx_store import NetworkXStore
        neo4j_store = NetworkXStore()
        neo4j_store.restore_from_registry(registry)

    # 3. LLM Service
    llm_service = OllamaLLMService(
        settings.ollama_base_url,
        settings.ollama_model,
        settings.ollama_timeout_seconds,
        settings.ollama_retries,
        mock_mode=settings.llm_mock_mode,
    )

    extractor = EntityRelationshipExtractor(llm_service)
    document_service = DocumentService(
        registry,
        embedding_service,
        qdrant_store,
        neo4j_store,
        extractor,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        upload_size_limit_mb=settings.upload_size_limit_mb,
    )
    vector_retriever = VectorRetriever(embedding_service, qdrant_store, settings.default_top_k)
    graph_retriever = GraphRetriever(neo4j_store, max_hops=settings.graph_max_hops, max_results=settings.graph_max_results)
    hybrid_retriever = HybridRetriever(vector_retriever, graph_retriever, fusion_k=settings.rrf_k)
    query_service = QueryService(
        vector_retriever, graph_retriever, hybrid_retriever, llm_service, settings.default_top_k, settings.answer_temperature
    )
    result_store = ResultStore(registry)
    health_service = HealthService(qdrant_store, neo4j_store, llm_service, registry)

    return AppContext(
        settings=settings,
        registry=registry,
        user_registry=user_registry,
        embedding_service=embedding_service,
        qdrant_store=qdrant_store,
        neo4j_store=neo4j_store,
        llm_service=llm_service,
        extractor=extractor,
        document_service=document_service,
        vector_retriever=vector_retriever,
        graph_retriever=graph_retriever,
        hybrid_retriever=hybrid_retriever,
        query_service=query_service,
        result_store=result_store,
        health_service=health_service,
    )


def get_context() -> AppContext:
    import streamlit as st

    return st.cache_resource(build_context)()


def render_overview_page(context: AppContext) -> None:
    import pandas as pd
    import streamlit as st

    render_lab_header(
        title="NEXUS AI RESEARCH STUDIO",
        subtitle="Intelligent Retrieval. Connected Knowledge. Measurable Results.",
        badge_text="RESEARCH PLATFORM",
    )

    render_dev_mode_banner(context.settings.llm_mock_mode)

    # Core Strategy Pillars
    st.markdown(
        """
        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-bottom: 28px;">
            <div class="pillar-card" style="border-top: 4px solid #0284c7;">
                <h4 style="color: #0369a1;">⚡ Dense Vector RAG</h4>
                <p>
                    Retrieves passages using <strong>dense vector semantic similarity</strong> in high-dimensional embedding space. Ideal for direct factual queries and unstructured domain documents.
                </p>
            </div>
            <div class="pillar-card" style="border-top: 4px solid #059669;">
                <h4 style="color: #047857;">🕸️ GraphRAG</h4>
                <p>
                    Executes <strong>multi-hop entity and relationship traversal</strong> over structured Neo4j graph schemas. Solves complex relational queries and cross-document reasoning.
                </p>
            </div>
            <div class="pillar-card" style="border-top: 4px solid #7c3aed;">
                <h4 style="color: #6d28d9;">🔀 Hybrid RAG (RRF)</h4>
                <p>
                    Fuses dense vector semantic candidates with connected knowledge graph paths using <strong>Reciprocal Rank Fusion (RRF)</strong> to optimize overall recall and precision.
                </p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Real Live Database Counts
    st.markdown("### 📊 Live System Metrics")
    real_counts = context.health_service.get_real_counts()
    docs = context.registry.list_documents()
    failed_ingestions = len([d for d in docs if d.get("status") in ("failed", "partial")])

    m_cols = st.columns(6)
    with m_cols[0]:
        st.metric("Documents", real_counts["documents"])
    with m_cols[1]:
        st.metric("Corpus Chunks", real_counts["chunks"])
    with m_cols[2]:
        st.metric("Qdrant Points", real_counts["vector_points"], help="Live vector points in Qdrant")
    with m_cols[3]:
        st.metric("Graph Entities", real_counts["entities"], help="Live nodes in Neo4j")
    with m_cols[4]:
        st.metric("Relationships", real_counts["relationships"], help="Live edges in Neo4j")
    with m_cols[5]:
        st.metric("Evaluation Runs", real_counts["evaluation_runs"])

    st.markdown("---")

    # Research Workflow Actions & Service Health
    col_actions, col_health = st.columns([3, 2])

    with col_actions:
        st.markdown("### 🚀 Guided Research Actions")
        st.markdown("Select an investigation phase to proceed with your empirical evaluation:")

        import os
        from pathlib import Path
        prefix = "pages/" if Path("pages").exists() else "src/rag_benchmark_lab/pages/"

        btn_cols = st.columns(2)
        with btn_cols[0]:
            try:
                st.page_link(f"{prefix}2_Corpus_Management.py", label="📁 Manage Document Corpus", icon="📄")
            except Exception:
                st.markdown("[📁 Manage Document Corpus](2_Corpus_Management)")
            try:
                st.page_link(f"{prefix}3_Chat.py", label="💬 Interactive Chatbot", icon="🤖")
            except Exception:
                st.markdown("[💬 Interactive Chatbot](3_Chat)")
            try:
                st.page_link(f"{prefix}4_Comparison.py", label="⚖️ Side-by-Side Comparison", icon="📊")
            except Exception:
                st.markdown("[⚖️ Side-by-Side Comparison](4_Comparison)")
        with btn_cols[1]:
            try:
                st.page_link(f"{prefix}5_Graph_Explorer.py", label="🕸️ Explore Knowledge Graph", icon="🌐")
            except Exception:
                st.markdown("[🕸️ Explore Knowledge Graph](5_Graph_Explorer)")
            try:
                st.page_link(f"{prefix}6_Evaluation_Lab.py", label="🧪 Run Benchmark Experiments", icon="🔬")
            except Exception:
                st.markdown("[🧪 Run Benchmark Experiments](6_Evaluation_Lab)")
            try:
                st.page_link(f"{prefix}7_Analytics.py", label="📈 View Results & Analytics", icon="📉")
            except Exception:
                st.markdown("[📈 View Results & Analytics](7_Analytics)")

    with col_health:
        st.markdown("### 🔌 Service Health & Diagnostics")
        health_items = context.health_service.check()
        for item in health_items:
            if item.healthy:
                st.markdown(
                    f"""
                    <div class="status-pill-healthy">
                        <span>✅</span>
                        <div>
                            <strong>{item.name}</strong><br/>
                            <span style="font-size: 11.5px; opacity: 0.85;">{item.details or 'Healthy'}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif "mock" in item.name.lower() or "dev" in item.name.lower():
                st.markdown(
                    f"""
                    <div class="status-pill-dev">
                        <span>⚡</span>
                        <div>
                            <strong>{item.name}</strong><br/>
                            <span style="font-size: 11.5px; opacity: 0.85;">{item.details or 'Offline dev mode'}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="status-pill-offline">
                        <span>❌</span>
                        <div>
                            <strong>{item.name}</strong><br/>
                            <span style="font-size: 11.5px; opacity: 0.85;">{item.details or 'Unavailable'}</span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Research Architecture Flowchart Section
    st.markdown("### 🏛️ System Research Architecture")
    st.markdown(
        """
        The architecture enforces parallel dual-indexing and 3-way comparative retrieval:
        """
    )
    st.markdown(
        """
```mermaid
flowchart TD
    subgraph Ingestion ["1. DUAL-PIPELINE INGESTION"]
        DOCS[Document Upload PDF / TXT / DOCX] --> CLEAN[Text Extraction & Cleaning]
        CLEAN --> CHUNK[Sliding Window Chunking]
        CHUNK -->|Dense Embeddings| EMB[Sentence Transformers]
        EMB -->|Vector Upsert| QD[(Qdrant Vector Database)]
        CHUNK -->|Structured Extraction| EXT[Entity & Relationship Extractor]
        EXT -->|Graph Upsert| NEO[(Neo4j Knowledge Graph)]
    end

    subgraph QueryExecution ["2. QUERY-TIME RETRIEVAL & FUSION"]
        Q[User Query] --> ROUTE{Query Router}
        ROUTE -->|Direct Similarity| VR[Vector RAG]
        ROUTE -->|Multi-Hop Traversal| GR[GraphRAG]
        ROUTE -->|Comprehensive| HR[Hybrid RAG]
        VR --> QD
        GR --> NEO
        HR --> QD & NEO
        HR --> RRF[Reciprocal Rank Fusion RRF]
    end

    subgraph Generation ["3. GROUNDED ANSWER & EVALUATION"]
        VR & GR & RRF --> CTX[Grounded Context Construction]
        CTX --> LLM[Ollama Local LLM]
        LLM --> ANS[Answer with Passages & Citations]
        ANS --> EVAL[Evaluation Engine: Faithfulness, Recall@K, MRR, Latency]
    end
```
        """
    )

    # Current Ingested Corpus Section
    st.markdown("### 📚 Ingested Corpus Documents")
    if docs:
        df_docs = pd.DataFrame(
            [
                {
                    "Document Name": d["document_name"],
                    "Status": d["status"].upper(),
                    "Chunks": d["chunk_count"],
                    "Entities": d["entity_count"],
                    "Relationships": d["relationship_count"],
                    "Ingested At": d["created_at"][:19].replace("T", " ") if d.get("created_at") else "N/A",
                }
                for d in docs
            ]
        )
        st.dataframe(df_docs, use_container_width=True)
    else:
        st.info("No documents have been ingested yet. Navigate to Corpus Management to upload your research files.")


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="Research Overview - RAG Benchmark Lab", page_icon="🧪", layout="wide")
    context = get_context()
    render_overview_page(context)


if __name__ == "__main__":
    main()
