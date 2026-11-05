from __future__ import annotations

import os
from unittest.mock import MagicMock, patch
import pytest

from rag_benchmark_lab.generation.groq_service import GroqLLMService
from rag_benchmark_lab.generation.llm_service import create_llm_service
from rag_benchmark_lab.utils.errors import (
    AuthenticationError,
    ModelUnavailableError,
    NetworkError,
    RateLimitError,
    ServiceUnavailableError,
    StructuredOutputError,
    sanitize_secrets,
)


def _make_mock_groq_client(content: str = "Test response", model: str = "llama-3.3-70b-versatile"):
    client = MagicMock()
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    response.model = model
    response.usage.prompt_tokens = 15
    response.usage.completion_tokens = 25
    response.usage.total_tokens = 40
    client.chat.completions.create.return_value = response
    return client


def test_groq_initialization_and_configured():
    # Valid key
    svc = GroqLLMService(api_key="gsk_test1234567890abcdef123456", model_name="llama-3.3-70b-versatile")
    assert svc.is_configured() is True
    assert svc.model_name == "llama-3.3-70b-versatile"
    assert svc.provider_name == "Groq"

    # Placeholder / missing key
    svc_empty = GroqLLMService(api_key="", model_name="llama-3.3-70b-versatile")
    assert svc_empty.is_configured() is False

    svc_placeholder = GroqLLMService(api_key="your_groq_api_key", model_name="llama-3.3-70b-versatile")
    assert svc_placeholder.is_configured() is False


def test_groq_safe_health_check_no_network():
    svc = GroqLLMService(api_key="gsk_validkey1234567890abcdef", model_name="llama-3.3-70b-versatile")
    with patch("groq.Groq"):
        # Local validation does not call models.list()
        assert svc.health(check_remote=False) is True

    svc_unconfigured = GroqLLMService(api_key="", model_name="llama-3.3-70b-versatile")
    assert svc_unconfigured.health(check_remote=False) is False


def test_groq_text_completion_with_system_and_user_prompts():
    svc = GroqLLMService(api_key="gsk_test1234567890abcdef123456", model_name="llama-3.3-70b-versatile")
    mock_client = _make_mock_groq_client(content="Paris is the capital of France.")
    svc._client = mock_client

    resp = svc.complete(
        prompt="What is the capital of France?",
        system_prompt="You are a helpful geography tutor.",
        temperature=0.2,
        max_tokens=150,
    )

    assert resp.text == "Paris is the capital of France."
    assert resp.model_name == "llama-3.3-70b-versatile"
    assert resp.prompt_tokens == 15
    assert resp.completion_tokens == 25
    assert resp.total_tokens == 40
    assert resp.latency_ms is not None and resp.latency_ms >= 0

    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "llama-3.3-70b-versatile"
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["max_tokens"] == 150
    assert call_kwargs["messages"] == [
        {"role": "system", "content": "You are a helpful geography tutor."},
        {"role": "user", "content": "What is the capital of France?"},
    ]


def test_groq_complete_json_success():
    svc = GroqLLMService(api_key="gsk_test1234567890abcdef123456", model_name="llama-3.3-70b-versatile")
    mock_client = _make_mock_groq_client(content='{"entities": ["Qdrant", "Neo4j"], "status": "active"}')
    svc._client = mock_client

    result = svc.complete_json("Extract entities in JSON.")
    assert isinstance(result, dict)
    assert result["entities"] == ["Qdrant", "Neo4j"]
    assert result["status"] == "active"


def test_groq_complete_json_fallback_markdown():
    svc = GroqLLMService(api_key="gsk_test1234567890abcdef123456", model_name="llama-3.3-70b-versatile")
    mock_client = MagicMock()
    # First attempt (response_format=json_object) throws an error
    # Second attempt succeeds with markdown fences
    fallback_choice = MagicMock()
    fallback_choice.message.content = '```json\n{"result": "extracted_info"}\n```'
    fallback_resp = MagicMock()
    fallback_resp.choices = [fallback_choice]
    fallback_resp.model = "llama-3.3-70b-versatile"
    fallback_resp.usage = None

    mock_client.chat.completions.create.side_effect = [
        Exception("json_mode not supported"),
        fallback_resp,
    ]
    svc._client = mock_client

    result = svc.complete_json("Extract JSON.")
    assert result == {"result": "extracted_info"}


def test_groq_missing_api_key_raises_auth_error():
    svc = GroqLLMService(api_key="", model_name="llama-3.3-70b-versatile")
    with pytest.raises(AuthenticationError) as exc_info:
        svc.complete("Hello")
    assert "Groq API key is not configured" in str(exc_info.value)


def test_groq_invalid_api_key_401():
    from groq import AuthenticationError as GroqAuthError

    svc = GroqLLMService(api_key="gsk_invalid_test_key", model_name="llama-3.3-70b-versatile")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = GroqAuthError(
        message="Invalid API Key",
        response=MagicMock(status_code=401),
        body={"error": {"message": "Invalid API Key"}},
    )
    svc._client = mock_client

    with pytest.raises(AuthenticationError) as exc_info:
        svc.complete("Test query")
    assert "invalid or unauthorized" in str(exc_info.value)


def test_groq_model_not_found_404():
    from groq import NotFoundError

    svc = GroqLLMService(api_key="gsk_valid_looking_key", model_name="nonexistent-model")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = NotFoundError(
        message="The model `nonexistent-model` does not exist",
        response=MagicMock(status_code=404),
        body={"error": {"message": "The model `nonexistent-model` does not exist"}},
    )
    svc._client = mock_client

    with pytest.raises(ModelUnavailableError) as exc_info:
        svc.complete("Test query")
    assert "nonexistent-model" in str(exc_info.value)


def test_groq_rate_limit_error_429():
    from groq import RateLimitError as GroqRateLimitError

    svc = GroqLLMService(api_key="gsk_valid_looking_key", model_name="llama-3.3-70b-versatile")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = GroqRateLimitError(
        message="Rate limit reached",
        response=MagicMock(status_code=429),
        body={"error": {"message": "Rate limit reached"}},
    )
    svc._client = mock_client

    with pytest.raises(RateLimitError) as exc_info:
        svc.complete("Test query")
    assert "rate limit reached" in str(exc_info.value).lower()


def test_groq_network_timeout_error():
    from groq import APITimeoutError

    svc = GroqLLMService(api_key="gsk_valid_looking_key", model_name="llama-3.3-70b-versatile", retries=0)
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = APITimeoutError(
        request=MagicMock()
    )
    svc._client = mock_client

    with pytest.raises(NetworkError) as exc_info:
        svc.complete("Test query")
    assert "timeout" in str(exc_info.value).lower()


def test_secret_sanitization_masks_groq_keys():
    raw = "Failed with key gsk_abcdef1234567890abcdef123456 on model llama"
    sanitized = sanitize_secrets(raw)
    assert "gsk_abcdef" not in sanitized
    assert "[REDACTED_API_KEY]" in sanitized


def test_create_llm_service_factory_selection():
    # 1. Default is Groq
    svc_default = create_llm_service(
        provider="groq",
        groq_api_key="gsk_test1234567890abcdef123456",
        groq_model="llama-3.3-70b-versatile",
    )
    assert svc_default.__class__.__name__ == "GroqLLMService"
    assert svc_default.model_name == "llama-3.3-70b-versatile"

    # 2. Gemini provider
    svc_gemini = create_llm_service(
        provider="gemini",
        gemini_api_key="test_gemini_key",
        gemini_model="gemini-2.5-flash",
    )
    assert svc_gemini.__class__.__name__ == "GeminiLLMService"
    assert svc_gemini.model_name == "gemini-2.5-flash"

    # 3. Ollama provider
    svc_ollama = create_llm_service(
        provider="ollama",
        ollama_base_url="http://localhost:11434",
        ollama_model="llama3.1:8b",
        mock_mode=True,
    )
    assert svc_ollama.__class__.__name__ == "OllamaLLMService"
    assert svc_ollama.model_name == "llama3.1:8b"

