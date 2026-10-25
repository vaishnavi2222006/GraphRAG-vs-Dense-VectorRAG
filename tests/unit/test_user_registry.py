from __future__ import annotations

import pytest
import sqlite3
import tempfile
from pathlib import Path

from rag_benchmark_lab.schemas.auth import UserRecord, KnowledgeBaseRecord, ConversationRecord, MessageRecord
from rag_benchmark_lab.services.user_registry import UserRegistry


@pytest.fixture
def user_registry(tmp_path: Path) -> UserRegistry:
    db_file = tmp_path / "test_auth.sqlite3"
    return UserRegistry(db_file)


def test_user_registration_and_validation(user_registry: UserRegistry):
    # Valid registration
    user, errors = user_registry.register_user("Alice Researcher", "alice@example.com", "SecurePass123!")
    assert errors == []
    assert user is not None
    assert user.full_name == "Alice Researcher"
    assert user.email == "alice@example.com"
    assert user.password_hash != "SecurePass123!"

    # Duplicate email rejection
    dup_user, dup_errors = user_registry.register_user("Alice Clone", "alice@example.com", "SecurePass123!")
    assert dup_user is None
    assert any("already exists" in e.lower() for e in dup_errors)

    # Password validation (short password)
    weak_user, weak_errors = user_registry.register_user("Bob", "bob@example.com", "short")
    assert weak_user is None
    assert len(weak_errors) > 0


def test_user_authentication(user_registry: UserRegistry):
    user, _ = user_registry.register_user("Bob Smith", "bob@example.com", "SuperSecret99#")
    assert user is not None

    # Correct login
    authed_user, err = user_registry.authenticate_user("bob@example.com", "SuperSecret99#")
    assert err == ""
    assert authed_user is not None
    assert authed_user.user_id == user.user_id

    # Wrong password
    fail_user, err_msg = user_registry.authenticate_user("bob@example.com", "WrongPassword")
    assert fail_user is None
    assert "incorrect" in err_msg.lower() or "invalid" in err_msg.lower()

    # Nonexistent user
    unknown_user, err_msg = user_registry.authenticate_user("unknown@example.com", "Pass12345!")
    assert unknown_user is None
    assert "no account" in err_msg.lower()


def test_knowledge_base_isolation(user_registry: UserRegistry):
    user_a, _ = user_registry.register_user("User A", "usera@example.com", "PasswordA1!")
    user_b, _ = user_registry.register_user("User B", "userb@example.com", "PasswordB1!")

    # User A creates KB
    kb_a = user_registry.create_knowledge_base(user_a.user_id, "Biology KB", "Biology research")
    assert kb_a.name == "Biology KB"

    # User B creates KB
    kb_b = user_registry.create_knowledge_base(user_b.user_id, "Physics KB", "Physics research")
    assert kb_b.name == "Physics KB"

    # User A lists KBs - should only see their own
    kbs_a = user_registry.list_knowledge_bases(user_a.user_id)
    assert len(kbs_a) == 1
    assert kbs_a[0].name == "Biology KB"

    # User B cannot access User A's KB
    assert user_registry.get_knowledge_base(kb_a.kb_id, user_b.user_id) is None

    # User B cannot delete User A's KB
    del_docs = user_registry.delete_knowledge_base(kb_a.kb_id, user_b.user_id)
    assert del_docs == []
    # Verify User A's KB still exists
    assert user_registry.get_knowledge_base(kb_a.kb_id, user_a.user_id) is not None


def test_conversation_persistence(user_registry: UserRegistry):
    user, _ = user_registry.register_user("Charlie", "charlie@example.com", "PassWordCharlie1!")
    kb = user_registry.create_knowledge_base(user.user_id, "AI KB")

    # Create conversation
    conv = user_registry.create_conversation(user.user_id, kb.kb_id, "Initial Chat")
    assert conv.title == "Initial Chat"

    # Save user message
    user_msg = MessageRecord(
        message_id="msg-1",
        conversation_id=conv.conversation_id,
        user_id=user.user_id,
        role="user",
        content="What is GraphRAG?",
    )
    user_registry.save_message(user_msg)

    # Save assistant message with metadata
    assistant_msg = MessageRecord(
        message_id="msg-2",
        conversation_id=conv.conversation_id,
        user_id=user.user_id,
        role="assistant",
        content="GraphRAG combines knowledge graphs with dense vector retrieval.",
        retrieval_strategy="hybrid",
        citations=[{"doc_id": "doc-1", "page": 1}],
        retrieval_latency_ms=12.4,
        generation_latency_ms=45.2,
        total_latency_ms=57.6,
        model_name="llama3.1:8b",
    )
    user_registry.save_message(assistant_msg)

    # List messages
    msgs = user_registry.list_messages(conv.conversation_id, user.user_id)
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "What is GraphRAG?"
    assert msgs[1]["role"] == "assistant"
    assert msgs[1]["retrieval_strategy"] == "hybrid"
    assert msgs[1]["total_latency_ms"] == 57.6
