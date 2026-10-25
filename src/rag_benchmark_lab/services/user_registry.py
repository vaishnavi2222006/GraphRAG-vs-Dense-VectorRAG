from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas.auth import (
    ConversationRecord,
    KnowledgeBaseRecord,
    MessageRecord,
    UserRecord,
)

# Try argon2 first, fallback to bcrypt, finally use PBKDF2 (stdlib)
try:
    from argon2 import PasswordHasher as Argon2Hasher  # type: ignore
    from argon2.exceptions import VerifyMismatchError  # type: ignore
    _HASHER = "argon2"
    _argon2 = Argon2Hasher()
except ImportError:
    try:
        import bcrypt  # type: ignore
        _HASHER = "bcrypt"
    except ImportError:
        _HASHER = "pbkdf2"


def _hash_password(plaintext: str) -> str:
    if _HASHER == "argon2":
        return _argon2.hash(plaintext)
    if _HASHER == "bcrypt":
        import bcrypt
        return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt(rounds=12)).decode()
    # PBKDF2 fallback
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", plaintext.encode(), salt.encode(), 260_000)
    return f"pbkdf2:sha256:260000:{salt}:{dk.hex()}"


def _verify_password(plaintext: str, stored_hash: str) -> bool:
    if stored_hash.startswith("$argon2") or _HASHER == "argon2":
        try:
            from argon2 import PasswordHasher as Argon2Hasher
            from argon2.exceptions import VerifyMismatchError
            ph = Argon2Hasher()
            return ph.verify(stored_hash, plaintext)
        except Exception:
            return False
    if stored_hash.startswith("$2") or _HASHER == "bcrypt":
        try:
            import bcrypt
            return bcrypt.checkpw(plaintext.encode(), stored_hash.encode())
        except Exception:
            return False
    # PBKDF2
    if stored_hash.startswith("pbkdf2:"):
        try:
            _, alg, iterations, salt, dk_hex = stored_hash.split(":")
            dk = hashlib.pbkdf2_hmac(alg.split(":")[0] if ":" in alg else alg, plaintext.encode(), salt.encode(), int(iterations))
            return secrets.compare_digest(dk.hex(), dk_hex)
        except Exception:
            return False
    return False


def _validate_email(email: str) -> bool:
    pattern = r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$"
    return bool(re.match(pattern, email.strip()))


def _validate_password(password: str) -> list[str]:
    errors = []
    if len(password) < 8:
        errors.append("Password must be at least 8 characters long.")
    if not any(c.isupper() for c in password):
        errors.append("Password must contain at least one uppercase letter.")
    if not any(c.islower() for c in password):
        errors.append("Password must contain at least one lowercase letter.")
    if not any(c.isdigit() for c in password):
        errors.append("Password must contain at least one digit.")
    return errors


class UserRegistry:
    """Manages users, knowledge bases, conversations, and messages in SQLite."""

    def __init__(self, sqlite_path: Path | str) -> None:
        self.sqlite_path = Path(sqlite_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.sqlite_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialize(self) -> None:
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    full_name TEXT NOT NULL,
                    email TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    is_active INTEGER NOT NULL DEFAULT 1,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS knowledge_bases (
                    kb_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    name TEXT NOT NULL,
                    description TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    document_count INTEGER NOT NULL DEFAULT 0,
                    chunk_count INTEGER NOT NULL DEFAULT 0,
                    entity_count INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
                    title TEXT NOT NULL,
                    message_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL REFERENCES conversations(conversation_id),
                    user_id TEXT NOT NULL REFERENCES users(user_id),
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    retrieval_strategy TEXT,
                    retrieval_latency_ms REAL,
                    generation_latency_ms REAL,
                    total_latency_ms REAL,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    graph_evidence_json TEXT,
                    model_name TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    mock_mode INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS kb_documents (
                    kb_id TEXT NOT NULL REFERENCES knowledge_bases(kb_id),
                    document_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (kb_id, document_id)
                );

                CREATE INDEX IF NOT EXISTS idx_kb_user ON knowledge_bases(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_kb ON conversations(kb_id);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id);
                CREATE INDEX IF NOT EXISTS idx_kb_docs ON kb_documents(kb_id);
            """)

    # ─── User Management ───────────────────────────────────────────────────────

    def register_user(self, full_name: str, email: str, password: str) -> tuple[UserRecord | None, list[str]]:
        """Register a new user. Returns (user, errors)."""
        errors: list[str] = []
        full_name = full_name.strip()
        email = email.strip().lower()

        if not full_name:
            errors.append("Full name is required.")
        if not _validate_email(email):
            errors.append("Invalid email address format.")
        pwd_errors = _validate_password(password)
        errors.extend(pwd_errors)

        if errors:
            return None, errors

        with self._connect() as conn:
            existing = conn.execute("SELECT user_id FROM users WHERE email = ?", (email,)).fetchone()
            if existing:
                return None, ["An account with this email address already exists."]

            user_id = hashlib.sha256(f"{email}:{secrets.token_hex(16)}".encode()).hexdigest()[:32]
            now = datetime.now(timezone.utc).isoformat()
            password_hash = _hash_password(password)
            conn.execute(
                "INSERT INTO users (user_id, full_name, email, password_hash, is_active, created_at, updated_at) VALUES (?, ?, ?, ?, 1, ?, ?)",
                (user_id, full_name, email, password_hash, now, now),
            )

        return UserRecord(
            user_id=user_id,
            full_name=full_name,
            email=email,
            password_hash=password_hash,
            created_at=now,
            updated_at=now,
        ), []

    def authenticate_user(self, email: str, password: str) -> tuple[UserRecord | None, str]:
        """Authenticate a user. Returns (user, error_message)."""
        email = email.strip().lower()
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE email = ? AND is_active = 1", (email,)).fetchone()
        if row is None:
            return None, "No account found with that email address."
        stored_hash = row["password_hash"]
        if not _verify_password(password, stored_hash):
            return None, "Incorrect password."
        return UserRecord(
            user_id=row["user_id"],
            full_name=row["full_name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
        ), ""

    def get_user(self, user_id: str) -> UserRecord | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM users WHERE user_id = ?", (user_id,)).fetchone()
        if not row:
            return None
        return UserRecord(
            user_id=row["user_id"],
            full_name=row["full_name"],
            email=row["email"],
            password_hash=row["password_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            is_active=bool(row["is_active"]),
        )

    # ─── Knowledge Base Management ─────────────────────────────────────────────

    def create_knowledge_base(self, user_id: str, name: str, description: str = "") -> KnowledgeBaseRecord:
        import uuid
        kb_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO knowledge_bases (kb_id, user_id, name, description, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (kb_id, user_id, name.strip(), description.strip(), now, now),
            )
        return KnowledgeBaseRecord(kb_id=kb_id, user_id=user_id, name=name.strip(), description=description.strip(), created_at=now, updated_at=now)

    def list_knowledge_bases(self, user_id: str) -> list[KnowledgeBaseRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM knowledge_bases WHERE user_id = ? ORDER BY updated_at DESC",
                (user_id,),
            ).fetchall()
        return [
            KnowledgeBaseRecord(
                kb_id=row["kb_id"],
                user_id=row["user_id"],
                name=row["name"],
                description=row["description"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                document_count=row["document_count"],
                chunk_count=row["chunk_count"],
                entity_count=row["entity_count"],
            )
            for row in rows
        ]

    def get_knowledge_base(self, kb_id: str, user_id: str) -> KnowledgeBaseRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM knowledge_bases WHERE kb_id = ? AND user_id = ?",
                (kb_id, user_id),
            ).fetchone()
        if not row:
            return None
        return KnowledgeBaseRecord(
            kb_id=row["kb_id"],
            user_id=row["user_id"],
            name=row["name"],
            description=row["description"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            document_count=row["document_count"],
            chunk_count=row["chunk_count"],
            entity_count=row["entity_count"],
        )

    def rename_knowledge_base(self, kb_id: str, user_id: str, new_name: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE knowledge_bases SET name = ?, updated_at = ? WHERE kb_id = ? AND user_id = ?",
                (new_name.strip(), now, kb_id, user_id),
            )
        return result.rowcount > 0

    def delete_knowledge_base(self, kb_id: str, user_id: str) -> list[str]:
        """Delete a knowledge base and return list of document_ids in it."""
        with self._connect() as conn:
            doc_rows = conn.execute("SELECT document_id FROM kb_documents WHERE kb_id = ?", (kb_id,)).fetchall()
            doc_ids = [row["document_id"] for row in doc_rows]
            # Delete conversations and messages
            conv_rows = conn.execute("SELECT conversation_id FROM conversations WHERE kb_id = ?", (kb_id,)).fetchall()
            for conv_row in conv_rows:
                conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv_row["conversation_id"],))
            conn.execute("DELETE FROM conversations WHERE kb_id = ?", (kb_id,))
            conn.execute("DELETE FROM kb_documents WHERE kb_id = ?", (kb_id,))
            conn.execute("DELETE FROM knowledge_bases WHERE kb_id = ? AND user_id = ?", (kb_id, user_id))
        return doc_ids

    def update_kb_counts(self, kb_id: str) -> None:
        with self._connect() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM kb_documents WHERE kb_id = ?", (kb_id,)).fetchone()[0]
            conn.execute(
                "UPDATE knowledge_bases SET document_count = ?, updated_at = ? WHERE kb_id = ?",
                (doc_count, datetime.now(timezone.utc).isoformat(), kb_id),
            )

    # ─── Document ↔ KB Association ─────────────────────────────────────────────

    def add_document_to_kb(self, kb_id: str, document_id: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO kb_documents (kb_id, document_id, created_at) VALUES (?, ?, ?)",
                (kb_id, document_id, now),
            )
        self.update_kb_counts(kb_id)

    def get_kb_document_ids(self, kb_id: str, user_id: str) -> list[str]:
        """Return document IDs for a KB, after verifying ownership."""
        with self._connect() as conn:
            # Verify ownership
            kb_row = conn.execute("SELECT kb_id FROM knowledge_bases WHERE kb_id = ? AND user_id = ?", (kb_id, user_id)).fetchone()
            if not kb_row:
                return []
            rows = conn.execute("SELECT document_id FROM kb_documents WHERE kb_id = ?", (kb_id,)).fetchall()
        return [row["document_id"] for row in rows]

    def remove_document_from_kb(self, kb_id: str, document_id: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM kb_documents WHERE kb_id = ? AND document_id = ?", (kb_id, document_id))
        self.update_kb_counts(kb_id)

    # ─── Conversation Management ────────────────────────────────────────────────

    def create_conversation(self, user_id: str, kb_id: str, title: str) -> ConversationRecord:
        import uuid
        conv_id = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (conversation_id, user_id, kb_id, title, message_count, created_at, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
                (conv_id, user_id, kb_id, title.strip(), now, now),
            )
        return ConversationRecord(conversation_id=conv_id, user_id=user_id, kb_id=kb_id, title=title.strip(), created_at=now, updated_at=now)

    def list_conversations(self, user_id: str, kb_id: str | None = None) -> list[ConversationRecord]:
        with self._connect() as conn:
            if kb_id:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE user_id = ? AND kb_id = ? ORDER BY updated_at DESC",
                    (user_id, kb_id),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                    (user_id,),
                ).fetchall()
        return [
            ConversationRecord(
                conversation_id=row["conversation_id"],
                user_id=row["user_id"],
                kb_id=row["kb_id"],
                title=row["title"],
                message_count=row["message_count"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
            )
            for row in rows
        ]

    def get_conversation(self, conversation_id: str, user_id: str) -> ConversationRecord | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE conversation_id = ? AND user_id = ?",
                (conversation_id, user_id),
            ).fetchone()
        if not row:
            return None
        return ConversationRecord(
            conversation_id=row["conversation_id"],
            user_id=row["user_id"],
            kb_id=row["kb_id"],
            title=row["title"],
            message_count=row["message_count"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def rename_conversation(self, conversation_id: str, user_id: str, new_title: str) -> bool:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            result = conn.execute(
                "UPDATE conversations SET title = ?, updated_at = ? WHERE conversation_id = ? AND user_id = ?",
                (new_title.strip(), now, conversation_id, user_id),
            )
        return result.rowcount > 0

    def delete_conversation(self, conversation_id: str, user_id: str) -> bool:
        with self._connect() as conn:
            row = conn.execute("SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
            conn.execute("DELETE FROM conversations WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id))
        return True

    # ─── Message Management ─────────────────────────────────────────────────────

    def save_message(self, message: MessageRecord) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO messages (
                    message_id, conversation_id, user_id, role, content,
                    retrieval_strategy, retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                    citations_json, graph_evidence_json, model_name,
                    warnings_json, errors_json, mock_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(message_id) DO NOTHING""",
                (
                    message.message_id,
                    message.conversation_id,
                    message.user_id,
                    message.role,
                    message.content,
                    message.retrieval_strategy,
                    message.retrieval_latency_ms,
                    message.generation_latency_ms,
                    message.total_latency_ms,
                    json.dumps(message.citations, ensure_ascii=False),
                    json.dumps(message.graph_evidence, ensure_ascii=False) if message.graph_evidence else None,
                    message.model_name,
                    json.dumps(message.warnings, ensure_ascii=False),
                    json.dumps(message.errors, ensure_ascii=False),
                    1 if message.mock_mode else 0,
                    message.created_at,
                ),
            )
            conn.execute(
                "UPDATE conversations SET message_count = message_count + 1, updated_at = ? WHERE conversation_id = ?",
                (now, message.conversation_id),
            )

    def list_messages(self, conversation_id: str, user_id: str) -> list[dict]:
        """Return messages only if conversation belongs to user."""
        with self._connect() as conn:
            # Verify ownership
            conv = conn.execute("SELECT conversation_id FROM conversations WHERE conversation_id = ? AND user_id = ?", (conversation_id, user_id)).fetchone()
            if not conv:
                return []
            rows = conn.execute(
                "SELECT * FROM messages WHERE conversation_id = ? ORDER BY created_at ASC",
                (conversation_id,),
            ).fetchall()
        result = []
        for row in rows:
            result.append({
                "message_id": row["message_id"],
                "conversation_id": row["conversation_id"],
                "user_id": row["user_id"],
                "role": row["role"],
                "content": row["content"],
                "retrieval_strategy": row["retrieval_strategy"],
                "retrieval_latency_ms": row["retrieval_latency_ms"],
                "generation_latency_ms": row["generation_latency_ms"],
                "total_latency_ms": row["total_latency_ms"],
                "citations": json.loads(row["citations_json"] or "[]"),
                "graph_evidence": json.loads(row["graph_evidence_json"]) if row["graph_evidence_json"] else None,
                "model_name": row["model_name"],
                "warnings": json.loads(row["warnings_json"] or "[]"),
                "errors": json.loads(row["errors_json"] or "[]"),
                "mock_mode": bool(row["mock_mode"]),
                "created_at": row["created_at"],
            })
        return result

    def validate_email(self, email: str) -> bool:
        return _validate_email(email)

    def validate_password(self, password: str) -> list[str]:
        return _validate_password(password)
