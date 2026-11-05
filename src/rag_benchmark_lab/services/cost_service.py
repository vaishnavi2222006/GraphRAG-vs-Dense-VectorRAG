"""LLM API Cost and Token Tracking Service.

Calculates estimated per-query LLM costs in USD and INR based on actual token
consumption and configurable pricing models.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Any


# Verified officially published pricing (USD per 1,000,000 tokens)
# Format: model_key -> (input_cost_per_mtok, output_cost_per_mtok)
VERIFIED_MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Groq Official Pricing (per 1M tokens)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.3-70b-specdec": (0.59, 0.99),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "llama-guard-3-8b": (0.20, 0.20),
    "llama3-70b-8192": (0.59, 0.79),
    "llama3-8b-8192": (0.05, 0.08),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
    "gemma-7b-it": (0.07, 0.07),

    # Google Gemini Official Pricing (per 1M tokens <= 128k prompt length)
    "gemini-2.5-flash": (0.075, 0.30),
    "gemini-1.5-flash": (0.075, 0.30),
    "gemini-1.5-flash-latest": (0.075, 0.30),
    "gemini-1.5-pro": (1.25, 5.00),
    "gemini-1.5-pro-latest": (1.25, 5.00),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.0-flash-exp": (0.0, 0.0),

    # Local Ollama / Offline models (Zero API cost)
    "ollama": (0.0, 0.0),
}


@dataclass(slots=True)
class CostEstimate:
    """Detailed cost and token estimation for a single query."""
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    cost_usd: float | None
    cost_inr: float | None
    input_price_per_mtok: float | None
    output_price_per_mtok: float | None
    currency_rate: float
    is_available: bool
    model_name: str
    provider: str
    reason: str | None = None
    pricing_source: str = "none"  # "configured_env", "verified_catalog", "free_tier", "none"

    @property
    def formatted_usd(self) -> str:
        if not self.is_available or self.cost_usd is None:
            return "Cost unavailable"
        if self.cost_usd == 0.0:
            return "$0.000000 USD"
        if self.cost_usd < 0.01:
            return f"${self.cost_usd:.6f} USD"
        return f"${self.cost_usd:.4f} USD"

    @property
    def formatted_inr(self) -> str:
        if not self.is_available or self.cost_inr is None:
            return "Cost unavailable"
        if self.cost_inr == 0.0:
            return "₹0.0000 INR"
        if self.cost_inr < 0.01:
            return f"₹{self.cost_inr:.4f} INR"
        return f"₹{self.cost_inr:.4f} INR"

    @property
    def formatted_combined(self) -> str:
        if not self.is_available or self.cost_usd is None or self.cost_inr is None:
            return "Cost unavailable"
        return f"{self.formatted_usd} ({self.formatted_inr})"


class CostService:
    """Manages token pricing lookup and per-query cost calculation."""

    def __init__(
        self,
        default_usd_to_inr_rate: float = 86.5,
        custom_pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.default_usd_to_inr_rate = default_usd_to_inr_rate
        self.custom_pricing = custom_pricing or {}

    def get_usd_to_inr_rate(self) -> float:
        """Fetch current USD to INR conversion rate from environment or default."""
        env_rate = os.getenv("USD_TO_INR_RATE")
        if env_rate:
            try:
                rate = float(env_rate.strip())
                if rate > 0:
                    return rate
            except ValueError:
                pass
        return self.default_usd_to_inr_rate

    def get_model_pricing(
        self,
        model_name: str | None,
        provider: str | None = None,
    ) -> tuple[float | None, float | None, str]:
        """Resolve input and output pricing (USD per 1M tokens) and its source.

        Priority order:
        1. Explicit environment variables (LLM_INPUT_COST_PER_MTOK, LLM_OUTPUT_COST_PER_MTOK).
        2. Provider-specific env variables (e.g. GROQ_INPUT_COST_PER_MTOK).
        3. Custom pricing dictionary passed at initialization.
        4. Verified official provider catalog (VERIFIED_MODEL_PRICING).
        5. If provider is Ollama -> (0.0, 0.0, 'free_tier').
        6. Otherwise -> (None, None, 'none') -> Cost unavailable.
        """
        # 1. Global env override
        global_in = os.getenv("LLM_INPUT_COST_PER_MTOK")
        global_out = os.getenv("LLM_OUTPUT_COST_PER_MTOK")
        if global_in is not None and global_out is not None:
            try:
                return float(global_in.strip()), float(global_out.strip()), "configured_env"
            except ValueError:
                pass

        # 2. Provider-specific env override
        prov_upper = (provider or "").upper().strip()
        if prov_upper:
            p_in = os.getenv(f"{prov_upper}_INPUT_COST_PER_MTOK")
            p_out = os.getenv(f"{prov_upper}_OUTPUT_COST_PER_MTOK")
            if p_in is not None and p_out is not None:
                try:
                    return float(p_in.strip()), float(p_out.strip()), "configured_env"
                except ValueError:
                    pass

        # 3. Provider is Ollama / local -> free
        if prov_upper in ("OLLAMA", "LOCAL"):
            return 0.0, 0.0, "free_tier"

        if not model_name:
            return None, None, "none"

        norm_model = model_name.lower().strip()
        base_name = norm_model.split("/")[-1]

        # 4. Custom pricing map
        if norm_model in self.custom_pricing:
            return self.custom_pricing[norm_model][0], self.custom_pricing[norm_model][1], "custom"
        if base_name in self.custom_pricing:
            return self.custom_pricing[base_name][0], self.custom_pricing[base_name][1], "custom"

        # 5. Verified catalog lookup
        if norm_model in VERIFIED_MODEL_PRICING:
            rates = VERIFIED_MODEL_PRICING[norm_model]
            return rates[0], rates[1], "verified_catalog"
        if base_name in VERIFIED_MODEL_PRICING:
            rates = VERIFIED_MODEL_PRICING[base_name]
            return rates[0], rates[1], "verified_catalog"

        # Prefix matching for versioned names like "gemini-1.5-flash-001" or "llama-3.3-70b-versatile-preview"
        for key, rates in VERIFIED_MODEL_PRICING.items():
            if norm_model.startswith(key) or base_name.startswith(key):
                return rates[0], rates[1], "verified_catalog"

        return None, None, "none"

    def calculate_cost(
        self,
        prompt_tokens: int | None,
        completion_tokens: int | None,
        model_name: str | None,
        provider: str | None = None,
        total_tokens: int | None = None,
    ) -> CostEstimate:
        """Calculate per-query cost in USD and INR with strict availability validation.

        If tokens or pricing are unavailable, returns is_available=False with formatted
        strings displaying 'Cost unavailable' to prevent showing assumed or incorrect amounts.
        """
        model_str = model_name or "Unknown model"
        prov_str = provider or "LLM"
        usd_to_inr = self.get_usd_to_inr_rate()

        if total_tokens is None:
            if prompt_tokens is not None and completion_tokens is not None:
                total_tokens = prompt_tokens + completion_tokens
            elif prompt_tokens is not None:
                total_tokens = prompt_tokens
            elif completion_tokens is not None:
                total_tokens = completion_tokens

        # Check if tokens are available
        if prompt_tokens is None or completion_tokens is None:
            return CostEstimate(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=None,
                cost_inr=None,
                input_price_per_mtok=None,
                output_price_per_mtok=None,
                currency_rate=usd_to_inr,
                is_available=False,
                model_name=model_str,
                provider=prov_str,
                reason="Token usage not reported by LLM provider response",
                pricing_source="none",
            )

        in_price, out_price, source = self.get_model_pricing(model_name, provider)

        if in_price is None or out_price is None:
            return CostEstimate(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=None,
                cost_inr=None,
                input_price_per_mtok=None,
                output_price_per_mtok=None,
                currency_rate=usd_to_inr,
                is_available=False,
                model_name=model_str,
                provider=prov_str,
                reason=f"Pricing not configured for model '{model_str}'",
                pricing_source=source,
            )

        # Cost calculation: (prompt_tokens * in_rate + completion_tokens * out_rate) / 1,000,000
        cost_usd = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0
        cost_inr = cost_usd * usd_to_inr

        return CostEstimate(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost_usd=cost_usd,
            cost_inr=cost_inr,
            input_price_per_mtok=in_price,
            output_price_per_mtok=out_price,
            currency_rate=usd_to_inr,
            is_available=True,
            model_name=model_str,
            provider=prov_str,
            reason=None,
            pricing_source=source,
        )
