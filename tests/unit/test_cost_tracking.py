"""Unit tests for Per-Query Token & Cost Tracking Service and Chat persistence."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import pytest

from rag_benchmark_lab.schemas.auth import MessageRecord
from rag_benchmark_lab.schemas.documents import AnswerResult, RetrievalStrategy
from rag_benchmark_lab.services.cost_service import CostEstimate, CostService, VERIFIED_MODEL_PRICING
from rag_benchmark_lab.services.user_registry import UserRegistry


def test_cost_calculation_verified_groq_model():
    """Verify that verified Groq models compute exact USD and INR costs accurately."""
    service = CostService(default_usd_to_inr_rate=86.5)

    # llama-3.3-70b-versatile: $0.59 / 1M input, $0.79 / 1M output
    # For 1,000 prompt tokens and 500 completion tokens:
    # in_cost = 1000 * 0.59 / 1_000_000 = 0.000590
    # out_cost = 500 * 0.79 / 1_000_000 = 0.000395
    # total = 0.000985 USD
    # inr = 0.000985 * 86.5 = 0.0852025 INR
    estimate = service.calculate_cost(
        prompt_tokens=1000,
        completion_tokens=500,
        model_name="llama-3.3-70b-versatile",
        provider="Groq",
    )

    assert estimate.is_available is True
    assert estimate.cost_usd is not None
    assert abs(estimate.cost_usd - 0.000985) < 1e-8
    assert estimate.cost_inr is not None
    assert abs(estimate.cost_inr - (0.000985 * 86.5)) < 1e-6
    assert estimate.total_tokens == 1500
    assert "$0.000985 USD" in estimate.formatted_usd
    assert "₹0.0852 INR" in estimate.formatted_inr
    assert estimate.pricing_source == "verified_catalog"


def test_cost_calculation_verified_gemini_model():
    """Verify Gemini pricing computation from catalog."""
    service = CostService(default_usd_to_inr_rate=85.0)

    # gemini-2.5-flash: $0.075 / 1M input, $0.30 / 1M output
    estimate = service.calculate_cost(
        prompt_tokens=2000,
        completion_tokens=400,
        model_name="gemini-2.5-flash",
        provider="Google Gemini",
    )

    assert estimate.is_available is True
    expected_usd = (2000 * 0.075 + 400 * 0.30) / 1_000_000.0
    assert abs(estimate.cost_usd - expected_usd) < 1e-8
    assert abs(estimate.cost_inr - (expected_usd * 85.0)) < 1e-6


def test_cost_calculation_local_ollama_zero():
    """Verify that local Ollama models report zero API cost."""
    service = CostService()
    estimate = service.calculate_cost(
        prompt_tokens=500,
        completion_tokens=250,
        model_name="llama3.1:8b-instruct-q4_K_M",
        provider="Ollama",
    )

    assert estimate.is_available is True
    assert estimate.cost_usd == 0.0
    assert estimate.cost_inr == 0.0
    assert "$0.000000 USD" in estimate.formatted_usd
    assert "₹0.0000 INR" in estimate.formatted_inr


def test_cost_unavailable_for_unknown_model_without_env():
    """Verify that models without published pricing report 'Cost unavailable' instead of assumed amounts."""
    # Ensure no environment overrides are set
    for env_k in ("LLM_INPUT_COST_PER_MTOK", "LLM_OUTPUT_COST_PER_MTOK"):
        if env_k in os.environ:
            del os.environ[env_k]

    service = CostService()
    estimate = service.calculate_cost(
        prompt_tokens=800,
        completion_tokens=300,
        model_name="custom-unknown-model-v1",
        provider="Groq",
    )

    assert estimate.is_available is False
    assert estimate.cost_usd is None
    assert estimate.cost_inr is None
    assert estimate.formatted_usd == "Cost unavailable"
    assert estimate.formatted_inr == "Cost unavailable"
    assert estimate.formatted_combined == "Cost unavailable"


def test_cost_unavailable_when_tokens_missing():
    """Verify that cost is unavailable when token usage is missing."""
    service = CostService()
    estimate = service.calculate_cost(
        prompt_tokens=None,
        completion_tokens=None,
        model_name="llama-3.3-70b-versatile",
        provider="Groq",
    )

    assert estimate.is_available is False
    assert estimate.cost_usd is None
    assert estimate.formatted_usd == "Cost unavailable"


def test_cost_env_override():
    """Verify that explicit environment pricing overrides take precedence."""
    service = CostService(default_usd_to_inr_rate=80.0)

    try:
        os.environ["LLM_INPUT_COST_PER_MTOK"] = "0.20"
        os.environ["LLM_OUTPUT_COST_PER_MTOK"] = "0.40"
        os.environ["USD_TO_INR_RATE"] = "80.0"

        estimate = service.calculate_cost(
            prompt_tokens=1000,
            completion_tokens=1000,
            model_name="qwen/qwen3.8-27b",
            provider="Groq",
        )

        assert estimate.is_available is True
        # (1000 * 0.20 + 1000 * 0.40) / 1M = 0.000600 USD
        assert abs(estimate.cost_usd - 0.000600) < 1e-8
        assert abs(estimate.cost_inr - 0.048000) < 1e-8
        assert estimate.pricing_source == "configured_env"
    finally:
        os.environ.pop("LLM_INPUT_COST_PER_MTOK", None)
        os.environ.pop("LLM_OUTPUT_COST_PER_MTOK", None)
        os.environ.pop("USD_TO_INR_RATE", None)


def test_user_registry_message_cost_persistence():
    """Verify that token consumption, provider, and costs are persisted to and retrieved from SQLite."""
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_messages.sqlite3"
        registry = UserRegistry(sqlite_path=db_path)

        # Register user, knowledge base, and conversation
        user, _ = registry.register_user("Cost Tester", "tester@med.org", "Password123!")
        assert user is not None
        kb = registry.create_knowledge_base(user.user_id, "Test KB", "KB Description")
        conv = registry.create_conversation(user.user_id, kb.kb_id, "Test Chat")

        # Save assistant message with full cost and token telemetry
        msg = MessageRecord(
            message_id="msg-cost-1",
            conversation_id=conv.conversation_id,
            user_id=user.user_id,
            role="assistant",
            content="MediCore has 120 inpatient beds.",
            retrieval_strategy="vector",
            retrieval_latency_ms=15.4,
            generation_latency_ms=820.0,
            total_latency_ms=835.4,
            model_name="llama-3.3-70b-versatile",
            llm_provider="Groq",
            prompt_tokens=650,
            completion_tokens=120,
            total_tokens=770,
            cost_usd=0.000478,
            cost_inr=0.041347,
        )

        registry.save_message(msg)

        # Retrieve messages
        retrieved = registry.list_messages(conv.conversation_id, user.user_id)
        assert len(retrieved) == 1
        r_msg = retrieved[0]

        assert r_msg["message_id"] == "msg-cost-1"
        assert r_msg["model_name"] == "llama-3.3-70b-versatile"
        assert r_msg["llm_provider"] == "Groq"
        assert r_msg["prompt_tokens"] == 650
        assert r_msg["completion_tokens"] == 120
        assert r_msg["total_tokens"] == 770
        assert abs(r_msg["cost_usd"] - 0.000478) < 1e-8
        assert abs(r_msg["cost_inr"] - 0.041347) < 1e-8
        assert r_msg["retrieval_latency_ms"] == 15.4
        assert r_msg["generation_latency_ms"] == 820.0
        assert r_msg["total_latency_ms"] == 835.4
