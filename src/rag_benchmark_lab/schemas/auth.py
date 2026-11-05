from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(slots=True)
class UserRecord:
    user_id: str
    full_name: str
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = True


@dataclass(slots=True)
class KnowledgeBaseRecord:
    kb_id: str
    user_id: str
    name: str
    description: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    document_count: int = 0
    chunk_count: int = 0
    entity_count: int = 0


@dataclass(slots=True)
class ConversationRecord:
    conversation_id: str
    user_id: str
    kb_id: str
    title: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    message_count: int = 0


@dataclass(slots=True)
class MessageRecord:
    message_id: str
    conversation_id: str
    user_id: str
    role: str  # "user" or "assistant"
    content: str
    retrieval_strategy: str | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    citations: list[dict[str, Any]] = field(default_factory=list)
    graph_evidence: dict[str, Any] | None = None
    model_name: str | None = None
    llm_provider: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    cost_usd: float | None = None
    cost_inr: float | None = None
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    mock_mode: bool = False
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
