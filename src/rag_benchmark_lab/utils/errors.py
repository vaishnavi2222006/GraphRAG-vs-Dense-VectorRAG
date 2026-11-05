from __future__ import annotations

import os
import re
from typing import Any


def sanitize_secrets(text: Any, extra_secrets: list[str] | None = None) -> str:
    """Scrub secret API keys, tokens, and credentials from strings, logs, and error traces."""
    if text is None:
        return ""
    sanitized = str(text)

    # 1. Redact known environment variables
    env_keys: list[str] = [
        os.getenv("GROQ_API_KEY", ""),
        os.getenv("GEMINI_API_KEY", ""),
        os.getenv("GOOGLE_API_KEY", ""),
        os.getenv("QDRANT_API_KEY", ""),
        os.getenv("NEO4J_PASSWORD", ""),
    ]
    if extra_secrets:
        env_keys.extend(extra_secrets)

    for secret in env_keys:
        secret_clean = (secret or "").strip()
        if len(secret_clean) >= 6:
            sanitized = sanitized.replace(secret_clean, "[REDACTED_SECRET]")

    # 2. Redact API key patterns (Google AIzaSy..., Groq gsk_...)
    sanitized = re.sub(r"AIza[0-9A-Za-z-_]{35}", "[REDACTED_API_KEY]", sanitized)
    sanitized = re.sub(r"gsk_[0-9A-Za-z-_]{20,}", "[REDACTED_API_KEY]", sanitized)

    # 3. Redact key=... / api_key=... / bearer ... query strings and headers
    sanitized = re.sub(r"(?i)(key|api_key|token|bearer|secret|password)=([^\s&]+)", r"\1=[REDACTED]", sanitized)
    sanitized = re.sub(r"(?i)bearer\s+[a-zA-Z0-9_\-\.]{15,}", "Bearer [REDACTED]", sanitized)

    return sanitized


class ConfigurationError(ValueError):
    """Raised when user configuration is invalid."""

    def __init__(self, message: str, technical_details: str | None = None) -> None:
        super().__init__(message)
        self.clean_message: str = message
        self.technical_details: str = sanitize_secrets(technical_details or message)


class AuthenticationError(ConfigurationError):
    """Raised when API key is missing or invalid."""

    def __init__(self, message: str, technical_details: str | None = None) -> None:
        super().__init__(message, technical_details=technical_details)


class ServiceUnavailableError(RuntimeError):
    """Raised when a backend service cannot be reached."""

    def __init__(self, message: str, technical_details: str | None = None) -> None:
        super().__init__(message)
        self.clean_message: str = message
        self.technical_details: str = sanitize_secrets(technical_details or message)


class QuotaExceededError(ServiceUnavailableError):
    """Raised when API rate limit or quota is exhausted."""

    def __init__(
        self,
        message: str,
        technical_details: str | None = None,
        is_daily_limit: bool = True,
        explanation: str | None = None,
    ) -> None:
        super().__init__(message, technical_details=technical_details)
        self.is_daily_limit: bool = is_daily_limit
        self.explanation: str = explanation or (
            "Keys belonging to the same project generally share the project's quota. "
            "Creating a new key within the same project will not reset the quota."
        )


class RateLimitError(QuotaExceededError):
    """Raised when per-minute request rate limit is reached."""

    def __init__(self, message: str, technical_details: str | None = None) -> None:
        super().__init__(
            message=message,
            technical_details=technical_details,
            is_daily_limit=False,
            explanation="Per-minute rate limit reached. Please wait a moment before trying again.",
        )


class ModelUnavailableError(ServiceUnavailableError):
    """Raised when the specified model is not found or unsupported."""


class NetworkError(ServiceUnavailableError):
    """Raised when a network failure or connection error occurs."""


class StructuredOutputError(ValueError):
    """Raised when model output cannot be parsed safely."""


class IngestionError(RuntimeError):
    """Raised when a document cannot be ingested."""

