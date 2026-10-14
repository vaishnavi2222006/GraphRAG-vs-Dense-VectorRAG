from __future__ import annotations

from dataclasses import dataclass

from .config.settings import AppSettings, get_settings
from .embeddings.embedding_service import EmbeddingService
from .evaluation.result_store import ResultStore
from .generation.llm_service import OllamaLLMService
from .graph_store.entity_extractor import EntityRelationshipExtractor
from .graph_store.graph_retriever import GraphRetriever
from .graph_store.neo4j_store import Neo4jStore
from .retrieval.hybrid_retriever import HybridRetriever
from .retrieval.vector_retriever import VectorRetriever
from .services.document_registry import DocumentRegistry
from .services.document_service import DocumentService
from .services.health_service import HealthService
from .services.query_service import QueryService
from .vector_store.qdrant_store import QdrantStore


@dataclass(slots=True)
class AppContext:
    settings: AppSettings
    registry: DocumentRegistry
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
    embedding_service = EmbeddingService(settings.embedding_model, mock_mode=settings.llm_mock_mode)
    qdrant_store = QdrantStore(settings.qdrant_url, settings.qdrant_api_key, settings.qdrant_collection)
    neo4j_store = Neo4jStore(settings.neo4j_uri, settings.neo4j_username, settings.neo4j_password)
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
    query_service = QueryService(vector_retriever, graph_retriever, hybrid_retriever, llm_service, settings.default_top_k, settings.answer_temperature)
    result_store = ResultStore(registry)
    health_service = HealthService(qdrant_store, neo4j_store, llm_service, registry)
    return AppContext(
        settings=settings,
        registry=registry,
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


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="RAG Benchmark Lab", page_icon="🧪", layout="wide")
    context = get_context()
    st.title("RAG Benchmark Lab")
    st.caption("An experimental platform for Vector RAG, GraphRAG, and Hybrid retrieval")

    counts = context.registry.counts()
    health = context.health_service.check()
    columns = st.columns(5)
    with columns[0]:
        st.metric("Documents", counts["documents"])
    with columns[1]:
        st.metric("Chunks", counts["chunks"])
    with columns[2]:
        st.metric("Entities", counts["entities"])
    with columns[3]:
        st.metric("Relationships", counts["relationships"])
    with columns[4]:
        st.metric("Evaluation Runs", counts["evaluation_runs"])

    st.subheader("Service Health")
    for item in health:
        st.write(f"{item.name}: {'healthy' if item.healthy else 'unavailable'}{f' - {item.details}' if item.details else ''}")

    st.subheader("Recent Documents")
    documents = context.registry.list_documents()
    if documents:
        st.dataframe(documents, use_container_width=True)
    else:
        st.info("No documents have been ingested yet.")


if __name__ == "__main__":
    main()
