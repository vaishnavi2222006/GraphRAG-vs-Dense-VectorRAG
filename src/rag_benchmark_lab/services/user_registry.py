from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import sqlite3
from pymongo import MongoClient
from pymongo.errors import DuplicateKeyError
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..schemas.auth import (
    ConversationRecord,
    KnowledgeBaseRecord,
    MessageRecord,
    UserRecord,
)
from ..schemas.comparison import (
    ComparisonRunRecord,
    ComparisonStrategyResultRecord,
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
        # pyrefly: ignore[missing-import]
        import bcrypt
        return bcrypt.hashpw(plaintext.encode(), bcrypt.gensalt(rounds=12)).decode()
    # PBKDF2 fallback
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", plaintext.encode(), salt.encode(), 260_000)
    return f"pbkdf2:sha256:260000:{salt}:{dk.hex()}"


def _verify_password(plaintext: str, stored_hash: str) -> bool:
    """Verify Argon2, bcrypt, or PBKDF2 password hashes."""

    # Argon2 verification
    if stored_hash.startswith("$argon2"):
        try:
            from argon2 import PasswordHasher
            ph = PasswordHasher()
            return ph.verify(stored_hash, plaintext)
        except Exception:
            return False

    # bcrypt verification
    if stored_hash.startswith("$2"):
        try:
            import bcrypt
            return bcrypt.checkpw(
                plaintext.encode(),
                stored_hash.encode(),
            )
        except Exception:
            return False

    # PBKDF2 verification
    if stored_hash.startswith("pbkdf2:"):
        try:
            _, algorithm, iterations, salt, expected_hash = stored_hash.split(":")
            derived_key = hashlib.pbkdf2_hmac(
                algorithm,
                plaintext.encode(),
                salt.encode(),
                int(iterations),
            )
            return secrets.compare_digest(
                derived_key.hex(),
                expected_hash,
            )
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
    """Manages authentication in MongoDB and application records in SQLite."""

    def __init__(
        self,
        sqlite_path: Path | str,
        mongodb_uri: str = "",
        mongodb_database: str = "rag_benchmark_lab",
        mongodb_users_collection: str = "users",
    ) -> None:
        self.sqlite_path = Path(sqlite_path)

        self.mongo_client = None
        self.mongo_users = None

        if mongodb_uri:
            try:
                self.mongo_client = MongoClient(
                    mongodb_uri,
                    serverSelectionTimeoutMS=2000,
                )
                self.mongo_client.admin.command("ping")
                self.mongo_users = self.mongo_client[
                    mongodb_database
                ][mongodb_users_collection]
            except Exception:
                self.mongo_client = None
                self.mongo_users = None

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
                    llm_provider TEXT,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    total_tokens INTEGER,
                    cost_usd REAL,
                    cost_inr REAL,
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

                CREATE TABLE IF NOT EXISTS comparison_runs (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT REFERENCES users(user_id),
                    kb_id TEXT REFERENCES knowledge_bases(kb_id),
                    query TEXT NOT NULL,
                    status TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    strategies_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS comparison_strategy_results (
                    result_id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES comparison_runs(run_id) ON DELETE CASCADE,
                    strategy TEXT NOT NULL,
                    strategy_label TEXT NOT NULL,
                    status TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    error_message TEXT,
                    retrieval_latency_ms REAL,
                    generation_latency_ms REAL,
                    total_latency_ms REAL,
                    prompt_tokens INTEGER,
                    completion_tokens INTEGER,
                    model_name TEXT,
                    citations_json TEXT NOT NULL DEFAULT '[]',
                    evidence_sources_json TEXT NOT NULL DEFAULT '[]',
                    graph_evidence_json TEXT,
                    warnings_json TEXT NOT NULL DEFAULT '[]',
                    errors_json TEXT NOT NULL DEFAULT '[]',
                    metadata_json TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_kb_user ON knowledge_bases(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_user ON conversations(user_id);
                CREATE INDEX IF NOT EXISTS idx_conv_kb ON conversations(kb_id);
                CREATE INDEX IF NOT EXISTS idx_msg_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_msg_user ON messages(user_id);
                CREATE INDEX IF NOT EXISTS idx_kb_docs ON kb_documents(kb_id);
                CREATE INDEX IF NOT EXISTS idx_comp_user ON comparison_runs(user_id);
                CREATE INDEX IF NOT EXISTS idx_comp_kb ON comparison_runs(kb_id);
                CREATE INDEX IF NOT EXISTS idx_comp_created ON comparison_runs(created_at);
                CREATE INDEX IF NOT EXISTS idx_comp_strat_run ON comparison_strategy_results(run_id);
                CREATE INDEX IF NOT EXISTS idx_comp_strat_strat ON comparison_strategy_results(strategy);
            """)

            # Ensure schema compatibility for messages table in existing databases
            existing_msg_cols = {row["name"] for row in conn.execute("PRAGMA table_info(messages)").fetchall()}
            for col_name, col_type in [
                ("llm_provider", "TEXT"),
                ("prompt_tokens", "INTEGER"),
                ("completion_tokens", "INTEGER"),
                ("total_tokens", "INTEGER"),
                ("cost_usd", "REAL"),
                ("cost_inr", "REAL"),
            ]:
                if col_name not in existing_msg_cols:
                    conn.execute(f"ALTER TABLE messages ADD COLUMN {col_name} {col_type}")

    # ─── User Management ───────────────────────────────────────────────────────

    def register_user(
        self,
        full_name: str,
        email: str,
        password: str,
    ) -> tuple[UserRecord | None, list[str]]:
        """Register a user in MongoDB and maintain a SQLite compatibility record."""

        errors: list[str] = []
        full_name = full_name.strip()
        email = email.strip().lower()

        if not full_name:
            errors.append("Full name is required.")

        if not _validate_email(email):
            errors.append("Invalid email address format.")

        errors.extend(_validate_password(password))

        if errors:
            return None, errors

        existing_sqlite = None
        with self._connect() as conn:
            existing_sqlite = conn.execute(
                "SELECT user_id FROM users WHERE email = ?",
                (email,),
            ).fetchone()

        if existing_sqlite or (self.mongo_users is not None and self.mongo_users.find_one({"email": email})):
            return None, [
                "An account with this email address already exists."
            ]

        user_id = hashlib.sha256(
            f"{email}:{secrets.token_hex(16)}".encode()
        ).hexdigest()[:32]

        now = datetime.now(timezone.utc).isoformat()
        password_hash = _hash_password(password)

        if self.mongo_users is not None:
            mongo_user = {
                "user_id": user_id,
                "full_name": full_name,
                "email": email,
                "password_hash": password_hash,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
            try:
                self.mongo_users.insert_one(mongo_user)
            except DuplicateKeyError:
                return None, [
                    "An account with this email address already exists."
                ]
            except Exception as exc:
                return None, [f"Registration failed: {exc}"]

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO users (
                    user_id, full_name, email, password_hash,
                    is_active, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    user_id,
                    full_name,
                    email,
                    password_hash,
                    now,
                    now,
                ),
            )

        return UserRecord(
            user_id=user_id,
            full_name=full_name,
            email=email,
            password_hash=password_hash,
            created_at=now,
            updated_at=now,
        ), []

    def authenticate_user(
        self,
        email: str,
        password: str,
    ) -> tuple[UserRecord | None, str]:
        """Authenticate through MongoDB and migrate legacy SQLite users."""

        email = email.strip().lower()

        if self.mongo_users is None:
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
            if not row:
                return None, "No account found with that email address."
            if not bool(row["is_active"]):
                return None, "This account is inactive."
            if not _verify_password(password, row["password_hash"]):
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

        try:
            # 1. Check MongoDB first.
            mongo_user = self.mongo_users.find_one({"email": email})

            if mongo_user is not None:
                if not mongo_user.get("is_active", False):
                    return None, "This account is inactive."

                stored_hash = mongo_user.get("password_hash", "")

                if not _verify_password(password, stored_hash):
                    return None, "Incorrect password."

                user_id = mongo_user["user_id"]

                # 2. Ensure a SQLite compatibility record exists.
                with self._connect() as conn:
                    existing = conn.execute(
                        "SELECT * FROM users WHERE user_id = ? OR email = ?",
                        (user_id, email),
                    ).fetchone()

                    if existing is None:
                        conn.execute(
                            """
                            INSERT INTO users (
                                user_id, full_name, email, password_hash,
                                is_active, created_at, updated_at
                            )
                            VALUES (?, ?, ?, ?, 1, ?, ?)
                            """,
                            (
                                user_id,
                                mongo_user["full_name"],
                                email,
                                stored_hash,
                                mongo_user["created_at"],
                                mongo_user["updated_at"],
                            ),
                        )
                    elif (
                        existing["user_id"] != user_id
                        or existing["email"] != email
                    ):
                        return None, (
                            "Account synchronization conflict. "
                            "Please contact support."
                        )

                return UserRecord(
                    user_id=user_id,
                    full_name=mongo_user["full_name"],
                    email=email,
                    password_hash=stored_hash,
                    created_at=mongo_user["created_at"],
                    updated_at=mongo_user["updated_at"],
                    is_active=True,
                ), ""

            # 3. MongoDB account not found: check legacy SQLite.
            with self._connect() as conn:
                row = conn.execute(
                    """
                    SELECT * FROM users
                    WHERE email = ? AND is_active = 1
                    """,
                    (email,),
                ).fetchone()

            if row is None:
                return None, "No account found with that email address."

            stored_hash = row["password_hash"]

            if not _verify_password(password, stored_hash):
                return None, "Incorrect password."

            # 4. Migrate the existing account without changing its ID.
            legacy_user = {
                "user_id": row["user_id"],
                "full_name": row["full_name"],
                "email": row["email"],
                "password_hash": stored_hash,
                "is_active": True,
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }

            try:
                self.mongo_users.insert_one(legacy_user)
                mongo_user = legacy_user

            except DuplicateKeyError:
                # Another login may have migrated this account concurrently.
                mongo_user = self.mongo_users.find_one({"email": email})

                if (
                    mongo_user is None
                    or mongo_user.get("user_id") != row["user_id"]
                    or not _verify_password(
                        password,
                        mongo_user.get("password_hash", ""),
                    )
                ):
                    return None, "Account migration conflict."

            return UserRecord(
                user_id=row["user_id"],
                full_name=row["full_name"],
                email=row["email"],
                password_hash=row["password_hash"],
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                is_active=True,
            ), ""

        except Exception:
            return None, (
                "Authentication service is temporarily unavailable. "
                "Please try again."
            )
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
                    llm_provider, prompt_tokens, completion_tokens, total_tokens, cost_usd, cost_inr,
                    warnings_json, errors_json, mock_mode, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                    getattr(message, "llm_provider", None),
                    getattr(message, "prompt_tokens", None),
                    getattr(message, "completion_tokens", None),
                    getattr(message, "total_tokens", None),
                    getattr(message, "cost_usd", None),
                    getattr(message, "cost_inr", None),
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
            keys = row.keys()
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
                "llm_provider": row["llm_provider"] if "llm_provider" in keys else None,
                "prompt_tokens": row["prompt_tokens"] if "prompt_tokens" in keys else None,
                "completion_tokens": row["completion_tokens"] if "completion_tokens" in keys else None,
                "total_tokens": row["total_tokens"] if "total_tokens" in keys else None,
                "cost_usd": row["cost_usd"] if "cost_usd" in keys else None,
                "cost_inr": row["cost_inr"] if "cost_inr" in keys else None,
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

    # ─── Comparison History Management ──────────────────────────────────────────

    def save_comparison_run(
        self,
        run: ComparisonRunRecord,
        results: list[ComparisonStrategyResultRecord] | None = None,
    ) -> str:
        """Persist a full comparison run and its strategy results atomically."""
        strat_results = results if results is not None else run.results
        with self._connect() as conn:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """INSERT INTO comparison_runs (
                    run_id, user_id, kb_id, query, status,
                    config_json, strategies_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    query=excluded.query,
                    status=excluded.status,
                    config_json=excluded.config_json,
                    strategies_json=excluded.strategies_json,
                    updated_at=excluded.updated_at
                """,
                (
                    run.run_id,
                    run.user_id,
                    run.kb_id,
                    run.query,
                    run.status,
                    json.dumps(run.config, ensure_ascii=False),
                    json.dumps(run.strategies, ensure_ascii=False),
                    run.created_at or now,
                    run.updated_at or now,
                ),
            )
            conn.execute("DELETE FROM comparison_strategy_results WHERE run_id = ?", (run.run_id,))
            for sr in strat_results:
                conn.execute(
                    """INSERT INTO comparison_strategy_results (
                        result_id, run_id, strategy, strategy_label, status,
                        answer, error_message, retrieval_latency_ms, generation_latency_ms, total_latency_ms,
                        prompt_tokens, completion_tokens, model_name,
                        citations_json, evidence_sources_json, graph_evidence_json,
                        warnings_json, errors_json, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        sr.result_id,
                        run.run_id,
                        sr.strategy,
                        sr.strategy_label,
                        sr.status,
                        sr.answer,
                        sr.error_message,
                        sr.retrieval_latency_ms,
                        sr.generation_latency_ms,
                        sr.total_latency_ms,
                        sr.prompt_tokens,
                        sr.completion_tokens,
                        sr.model_name,
                        json.dumps(sr.citations, ensure_ascii=False),
                        json.dumps(sr.evidence_sources, ensure_ascii=False),
                        json.dumps(sr.graph_evidence, ensure_ascii=False) if sr.graph_evidence else None,
                        json.dumps(sr.warnings, ensure_ascii=False),
                        json.dumps(sr.errors, ensure_ascii=False),
                        json.dumps(sr.metadata, ensure_ascii=False),
                        sr.created_at or now,
                    ),
                )
        return run.run_id

    def get_comparison_run(self, run_id: str, user_id: str | None = None) -> ComparisonRunRecord | None:
        """Retrieve a specific comparison run and its strategy results."""
        with self._connect() as conn:
            query = """
                SELECT cr.*, kb.name as kb_name
                FROM comparison_runs cr
                LEFT JOIN knowledge_bases kb ON cr.kb_id = kb.kb_id
                WHERE cr.run_id = ?
            """
            params: list[Any] = [run_id]
            if user_id:
                query += " AND cr.user_id = ?"
                params.append(user_id)
            row = conn.execute(query, params).fetchone()
            if not row:
                return None

            results_rows = conn.execute(
                "SELECT * FROM comparison_strategy_results WHERE run_id = ? ORDER BY rowid ASC",
                (run_id,),
            ).fetchall()

        strat_results = [
            ComparisonStrategyResultRecord(
                result_id=sr["result_id"],
                run_id=sr["run_id"],
                strategy=sr["strategy"],
                strategy_label=sr["strategy_label"],
                status=sr["status"],
                answer=sr["answer"],
                error_message=sr["error_message"],
                retrieval_latency_ms=sr["retrieval_latency_ms"],
                generation_latency_ms=sr["generation_latency_ms"],
                total_latency_ms=sr["total_latency_ms"],
                prompt_tokens=sr["prompt_tokens"],
                completion_tokens=sr["completion_tokens"],
                model_name=sr["model_name"],
                citations=json.loads(sr["citations_json"] or "[]"),
                evidence_sources=json.loads(sr["evidence_sources_json"] or "[]"),
                graph_evidence=json.loads(sr["graph_evidence_json"]) if sr["graph_evidence_json"] else None,
                warnings=json.loads(sr["warnings_json"] or "[]"),
                errors=json.loads(sr["errors_json"] or "[]"),
                metadata=json.loads(sr["metadata_json"] or "{}"),
                created_at=sr["created_at"],
            )
            for sr in results_rows
        ]

        return ComparisonRunRecord(
            run_id=row["run_id"],
            user_id=row["user_id"],
            kb_id=row["kb_id"],
            query=row["query"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            config=json.loads(row["config_json"] or "{}"),
            strategies=json.loads(row["strategies_json"] or "[]"),
            results=strat_results,
            kb_name=row["kb_name"] if "kb_name" in row.keys() else None,
        )

    def list_comparison_runs(
        self,
        user_id: str | None = None,
        kb_id: str | None = None,
        status: str | None = None,
        strategy: str | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        limit: int = 100,
    ) -> list[ComparisonRunRecord]:
        """List comparison runs matching optional filters."""
        with self._connect() as conn:
            query = """
                SELECT DISTINCT cr.*, kb.name as kb_name
                FROM comparison_runs cr
                LEFT JOIN knowledge_bases kb ON cr.kb_id = kb.kb_id
            """
            conditions: list[str] = []
            params: list[Any] = []

            if strategy:
                query += " JOIN comparison_strategy_results csr ON cr.run_id = csr.run_id"
                conditions.append("csr.strategy = ?")
                params.append(strategy)

            if user_id:
                conditions.append("cr.user_id = ?")
                params.append(user_id)

            if kb_id:
                conditions.append("cr.kb_id = ?")
                params.append(kb_id)

            if status:
                conditions.append("cr.status = ?")
                params.append(status)

            if start_date:
                conditions.append("cr.created_at >= ?")
                params.append(start_date)

            if end_date:
                conditions.append("cr.created_at <= ?")
                params.append(end_date)

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " ORDER BY cr.created_at DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

            runs = []
            for row in rows:
                run_id = row["run_id"]
                res_rows = conn.execute(
                    "SELECT * FROM comparison_strategy_results WHERE run_id = ? ORDER BY rowid ASC",
                    (run_id,),
                ).fetchall()
                strat_results = [
                    ComparisonStrategyResultRecord(
                        result_id=sr["result_id"],
                        run_id=sr["run_id"],
                        strategy=sr["strategy"],
                        strategy_label=sr["strategy_label"],
                        status=sr["status"],
                        answer=sr["answer"],
                        error_message=sr["error_message"],
                        retrieval_latency_ms=sr["retrieval_latency_ms"],
                        generation_latency_ms=sr["generation_latency_ms"],
                        total_latency_ms=sr["total_latency_ms"],
                        prompt_tokens=sr["prompt_tokens"],
                        completion_tokens=sr["completion_tokens"],
                        model_name=sr["model_name"],
                        citations=json.loads(sr["citations_json"] or "[]"),
                        evidence_sources=json.loads(sr["evidence_sources_json"] or "[]"),
                        graph_evidence=json.loads(sr["graph_evidence_json"]) if sr["graph_evidence_json"] else None,
                        warnings=json.loads(sr["warnings_json"] or "[]"),
                        errors=json.loads(sr["errors_json"] or "[]"),
                        metadata=json.loads(sr["metadata_json"] or "{}"),
                        created_at=sr["created_at"],
                    )
                    for sr in res_rows
                ]
                runs.append(
                    ComparisonRunRecord(
                        run_id=row["run_id"],
                        user_id=row["user_id"],
                        kb_id=row["kb_id"],
                        query=row["query"],
                        status=row["status"],
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        config=json.loads(row["config_json"] or "{}"),
                        strategies=json.loads(row["strategies_json"] or "[]"),
                        results=strat_results,
                        kb_name=row["kb_name"] if "kb_name" in row.keys() else None,
                    )
                )
            return runs

    def delete_comparison_run(self, run_id: str, user_id: str | None = None) -> bool:
        """Delete a comparison run and its cascade strategy results."""
        with self._connect() as conn:
            if user_id:
                row = conn.execute("SELECT run_id FROM comparison_runs WHERE run_id = ? AND user_id = ?", (run_id, user_id)).fetchone()
            else:
                row = conn.execute("SELECT run_id FROM comparison_runs WHERE run_id = ?", (run_id,)).fetchone()
            if not row:
                return False
            conn.execute("DELETE FROM comparison_strategy_results WHERE run_id = ?", (run_id,))
            conn.execute("DELETE FROM comparison_runs WHERE run_id = ?", (run_id,))
        return True
