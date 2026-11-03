from __future__ import annotations

import json
import os
from time import perf_counter
from typing import Any

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


class GeminiLLMService:
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
            self.api_key = (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY", "")).strip()
        self.model_name = (model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")).strip()
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self._client: Any = None

    def __repr__(self) -> str:
        key_status = "configured" if self.is_configured() else "missing"
        return f"GeminiLLMService(model={self.model_name!r}, api_key={key_status})"

    def is_configured(self) -> bool:
        """Validate API key is present and non-empty without making network calls."""
        key = (self.api_key or "").strip()
        return bool(key and key != "change-me" and key != "your-api-key-here")

    def _get_client(self) -> Any:
        if not self.is_configured():
            raise AuthenticationError(
                "Gemini API key is not configured. Please update GEMINI_API_KEY in your .env file and restart the application."
            )
        if self._client is None:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except Exception as exc:
                sanitized_exc = sanitize_secrets(str(exc))
                raise ConfigurationError(
                    f"Failed to initialize Gemini client: {sanitized_exc}",
                    technical_details=sanitized_exc,
                ) from exc
        return self._client

    def health(self, check_remote: bool = False) -> bool:
        """Safe health check.

        By default (check_remote=False), only validates local configuration and client
        initialization without making remote network requests to avoid consuming quota
        or triggering 429 rate limit errors during page rendering.
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
            client.models.get(model=self.model_name)
            return True
        except Exception:
            return False

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        client = self._get_client()
        start = perf_counter()
        from google.genai import types, errors

        config = types.GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_prompt if system_prompt else None,
        )

        try:
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=config,
            )
            latency_ms = (perf_counter() - start) * 1000.0

            text = response.text or ""
            prompt_tokens = None
            completion_tokens = None
            total_tokens = None
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
                completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)
                total_tokens = getattr(response.usage_metadata, "total_token_count", None)

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
        except errors.APIError as exc:
            code = getattr(exc, "code", None)
            raw_msg = str(exc)
            sanitized_tech = sanitize_secrets(f"HTTP {code}: {raw_msg}")
            msg_lower = raw_msg.lower()

            # 1. Invalid or Unauthorized API keys
            if (
                code in (401, 403)
                or "api_key_invalid" in msg_lower
                or "permission_denied" in msg_lower
                or "unauthenticated" in msg_lower
                or "api key not valid" in msg_lower
            ):
                raise AuthenticationError(
                    "Gemini API key is invalid or unauthorized. Please verify GEMINI_API_KEY in your .env file.",
                    technical_details=sanitized_tech,
                ) from exc

            # 2. Model not found (404)
            if code == 404 or "not_found" in msg_lower or "model unavailable" in msg_lower:
                raise ModelUnavailableError(
                    f"Gemini model '{self.model_name}' not found: {raw_msg}",
                    technical_details=sanitized_tech,
                ) from exc

            # 3. Quota & Rate Limit (HTTP 429 RESOURCE_EXHAUSTED)
            if code == 429 or "resource_exhausted" in msg_lower or "quota" in msg_lower:
                is_daily = any(k in msg_lower for k in ("perday", "daily", "per day", "requests per day")) or (
                    "quota exceeded" in msg_lower and "minute" not in msg_lower
                )
                if is_daily:
                    clean_msg = (
                        "Gemini's available quota has been exhausted for the configured model. "
                        "Check the project's quota or wait until it resets."
                    )
                    explanation = (
                        "Keys belonging to the same project generally share the project's quota. "
                        "Creating a new key within the same project will not reset the quota."
                    )
                else:
                    clean_msg = (
                        "Gemini per-minute request rate limit reached. "
                        "Please wait a moment before trying again."
                    )
                    explanation = (
                        "Requests per minute (RPM) limit reached for the configured model. "
                        "Wait 30-60 seconds before retrying."
                    )
                raise QuotaExceededError(
                    clean_msg,
                    technical_details=sanitized_tech,
                    is_daily_limit=is_daily,
                    explanation=explanation,
                ) from exc

            # 4. HTTP 503 Temporary Service Failure
            if (
                code == 503
                or "temporarily unavailable" in msg_lower
                or "service unavailable" in msg_lower
                or (code is None and "unavailable" in msg_lower)
            ):
                raise ServiceUnavailableError(
                    "Gemini service is temporarily unavailable (HTTP 503). Please retry in a few moments.",
                    technical_details=sanitized_tech,
                ) from exc

            # General API error
            raise ServiceUnavailableError(
                f"Gemini API error (HTTP {code}): {raw_msg}",
                technical_details=sanitized_tech,
            ) from exc
        except (TimeoutError, ConnectionError) as exc:
            sanitized_tech = sanitize_secrets(str(exc))
            raise NetworkError(
                f"Gemini network connection failure or timeout: {exc}",
                technical_details=sanitized_tech,
            ) from exc
        except Exception as exc:
            exc_name = exc.__class__.__name__
            sanitized_tech = sanitize_secrets(str(exc))
            if "Timeout" in exc_name or "Connect" in exc_name or "Network" in exc_name:
                raise NetworkError(
                    f"Gemini network error ({exc_name}): {exc}",
                    technical_details=sanitized_tech,
                ) from exc
            raise ServiceUnavailableError(
                f"Gemini call failed ({exc_name}): {exc}",
                technical_details=sanitized_tech,
            ) from exc

    def complete_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> dict:
        response = self.complete(prompt, system_prompt=system_prompt, temperature=temperature)
        try:
            text = response.text.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip().startswith("```"):
                    lines = lines[:-1]
                text = "\n".join(lines).strip()
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception as exc:
            raise StructuredOutputError(f"Could not parse JSON from Gemini output: {response.text[:200]}") from exc
