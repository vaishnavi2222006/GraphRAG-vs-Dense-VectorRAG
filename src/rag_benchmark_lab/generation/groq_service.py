from __future__ import annotations

import json
import logging
import os
import re
from time import perf_counter
from typing import Any

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from .llm_service import LLMResponse
from ..utils.errors import (
    AuthenticationError,
    ConfigurationError,
    ModelUnavailableError,
    NetworkError,
    QuotaExceededError,
    RateLimitError,
    ServiceUnavailableError,
    StructuredOutputError,
    sanitize_secrets,
)

logger = logging.getLogger(__name__)


class _TransientGroqError(Exception):
    """Internal exception for transient Groq errors eligible for retry."""


class GroqLLMService:
    """LLM generation service powered by Groq API."""

    provider_name: str = "Groq"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        timeout_seconds: int = 60,
        retries: int = 2,
    ) -> None:
        if api_key is not None:
            self.api_key = api_key.strip()
        else:
            self.api_key = (os.getenv("GROQ_API_KEY") or "").strip()
        self.model_name = (model_name or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")).strip()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._client: Any = None

    def __repr__(self) -> str:
        key_status = "configured" if self.is_configured() else "missing"
        return f"GroqLLMService(model={self.model_name!r}, api_key={key_status})"

    def is_configured(self) -> bool:
        """Validate API key is present and non-empty without making network calls."""
        key = (self.api_key or "").strip()
        placeholders = {
            "change-me",
            "your-api-key-here",
            "your_groq_api_key",
            "gsk_your_api_key",
            "none",
            "null",
        }
        return bool(key and key.lower() not in placeholders)

    def _get_client(self) -> Any:
        if not self.is_configured():
            raise AuthenticationError(
                "Groq API key is not configured. Please update GROQ_API_KEY in your .env file and restart the application."
            )
        if self._client is None:
            try:
                from groq import Groq
                self._client = Groq(
                    api_key=self.api_key,
                    timeout=float(self.timeout_seconds),
                )
            except Exception as exc:
                sanitized_exc = sanitize_secrets(str(exc))
                raise ConfigurationError(
                    f"Failed to initialize Groq client: {sanitized_exc}",
                    technical_details=sanitized_exc,
                ) from exc
        return self._client

    def health(self, check_remote: bool = False) -> bool:
        """Safe health check.

        By default (check_remote=False), only validates local configuration and client
        initialization without making remote network requests to avoid consuming rate limits
        during page rendering.
        """
        if not self.is_configured():
            return False
        if not check_remote:
            try:
                self._get_client()
                return True
            except Exception:
                return False
        try:
            client = self._get_client()
            client.models.list()
            return True
        except Exception:
            return False

    def complete(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """Generate text completion from Groq with retry and exception normalization."""
        client = self._get_client()

        messages: list[dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        # Retry transient errors up to self.retries times
        attempts = max(1, self.retries + 1)

        last_transient_exc: Exception | None = None
        for attempt in range(attempts):
            start = perf_counter()
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model_name,
                    "messages": messages,
                    "temperature": max(0.0, min(2.0, float(temperature))),
                }
                if max_tokens is not None and max_tokens > 0:
                    kwargs["max_tokens"] = max_tokens

                response = client.chat.completions.create(**kwargs)
                latency_ms = (perf_counter() - start) * 1000.0

                text = ""
                if response.choices and len(response.choices) > 0:
                    text = response.choices[0].message.content or ""

                prompt_tokens = None
                completion_tokens = None
                total_tokens = None
                if hasattr(response, "usage") and response.usage:
                    prompt_tokens = getattr(response.usage, "prompt_tokens", None)
                    completion_tokens = getattr(response.usage, "completion_tokens", None)
                    total_tokens = getattr(response.usage, "total_tokens", None)

                return LLMResponse(
                    text=text,
                    model_name=self.model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    latency_ms=latency_ms,
                    mock_mode=False,
                    warnings=[],
                )
            except Exception as exc:
                self._handle_groq_exception(exc, attempt, attempts)

        raise ServiceUnavailableError("Groq call failed after retry attempts.")

    def _handle_groq_exception(self, exc: Exception, attempt: int, max_attempts: int) -> None:
        """Map Groq SDK exceptions to application error hierarchy with sanitization."""
        from groq import (
            APIConnectionError,
            APIError,
            APITimeoutError,
            AuthenticationError as GroqAuthError,
            InternalServerError,
            NotFoundError,
            RateLimitError as GroqRateLimitError,
        )

        raw_msg = str(exc)
        sanitized_tech = sanitize_secrets(raw_msg)
        msg_lower = raw_msg.lower()
        status_code = getattr(exc, "status_code", None)

        # 1. Invalid or Unauthorized API keys
        if isinstance(exc, GroqAuthError) or status_code in (401, 403) or "invalid api key" in msg_lower:
            raise AuthenticationError(
                "Groq API key is invalid or unauthorized. Please verify GROQ_API_KEY in your .env file.",
                technical_details=sanitized_tech,
            ) from exc

        # 2. Model not found / unavailable (404)
        if isinstance(exc, NotFoundError) or status_code == 404 or "model_not_found" in msg_lower or "does not exist" in msg_lower:
            raise ModelUnavailableError(
                f"Groq model '{self.model_name}' not found: {raw_msg}",
                technical_details=sanitized_tech,
            ) from exc

        # 3. Rate Limit / Quota Exceeded (429)
        if isinstance(exc, GroqRateLimitError) or status_code == 429 or "rate_limit_exceeded" in msg_lower:
            is_daily = any(k in msg_lower for k in ("perday", "daily", "per day", "requests per day"))
            raise RateLimitError(
                "Groq request rate limit reached. Please wait a moment before trying again.",
                technical_details=sanitized_tech,
            ) from exc

        # 4. Transient network / timeout errors - retry if attempts remain
        if isinstance(exc, (APITimeoutError, APIConnectionError, TimeoutError, ConnectionError)):
            if attempt < max_attempts - 1:
                logger.warning("Groq connection timeout/failure on attempt %d/%d. Retrying...", attempt + 1, max_attempts)
                return
            raise NetworkError(
                f"Groq network connection failure or timeout: {raw_msg}",
                technical_details=sanitized_tech,
            ) from exc

        # 5. Internal Server Error (500, 502, 503, 504)
        if isinstance(exc, InternalServerError) or (status_code and status_code >= 500):
            if attempt < max_attempts - 1:
                logger.warning("Groq server error %s on attempt %d/%d. Retrying...", status_code, attempt + 1, max_attempts)
                return
            raise ServiceUnavailableError(
                f"Groq service is temporarily unavailable (HTTP {status_code}). Please retry in a few moments.",
                technical_details=sanitized_tech,
            ) from exc

        # 6. General API error
        if isinstance(exc, APIError):
            raise ServiceUnavailableError(
                f"Groq API error (HTTP {status_code}): {raw_msg}",
                technical_details=sanitized_tech,
            ) from exc

        exc_name = exc.__class__.__name__
        if "Timeout" in exc_name or "Connect" in exc_name or "Network" in exc_name:
            if attempt < max_attempts - 1:
                return
            raise NetworkError(
                f"Groq network error ({exc_name}): {raw_msg}",
                technical_details=sanitized_tech,
            ) from exc

        raise ServiceUnavailableError(
            f"Groq call failed ({exc_name}): {raw_msg}",
            technical_details=sanitized_tech,
        ) from exc

    def complete_json(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> dict:
        """Generate structured JSON response with fallback parsing."""
        client = self._get_client()

        # Augment system prompt if needed to guide JSON generation
        json_sys_prompt = system_prompt or "You are a helpful assistant that always outputs valid JSON."
        if "json" not in json_sys_prompt.lower():
            json_sys_prompt = f"{json_sys_prompt}\nYou must respond in valid JSON format."

        messages: list[dict[str, str]] = [
            {"role": "system", "content": json_sys_prompt},
            {"role": "user", "content": prompt},
        ]

        # First attempt: json_object response format
        try:
            kwargs: dict[str, Any] = {
                "model": self.model_name,
                "messages": messages,
                "temperature": max(0.0, min(2.0, float(temperature))),
                "response_format": {"type": "json_object"},
            }
            if max_tokens is not None and max_tokens > 0:
                kwargs["max_tokens"] = max_tokens

            response = client.chat.completions.create(**kwargs)
            text = (response.choices[0].message.content or "").strip() if response.choices else ""
            return json.loads(text)
        except Exception:
            # Fallback to standard completion and manual markdown fence extraction
            pass

        # Second attempt: standard completion + robust JSON extraction
        response = self.complete(prompt, system_prompt=json_sys_prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            text = response.text.strip()
            # Clean markdown code blocks
            if "```json" in text:
                text = text.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in text:
                text = text.split("```", 1)[1].split("```", 1)[0].strip()

            start_idx = text.find("{")
            end_idx = text.rfind("}")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return json.loads(text[start_idx : end_idx + 1])
            return json.loads(text)
        except Exception as exc:
            raise StructuredOutputError(
                f"Could not parse JSON from Groq output: {response.text[:200]}"
            ) from exc

