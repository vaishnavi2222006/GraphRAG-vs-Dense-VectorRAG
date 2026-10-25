from __future__ import annotations

from dataclasses import dataclass, field
import json
from time import perf_counter

import requests
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.errors import ServiceUnavailableError, StructuredOutputError


@dataclass(slots=True)
class LLMResponse:
    text: str
    model_name: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: float | None = None
    mock_mode: bool = False
    warnings: list[str] = field(default_factory=list)


class OllamaLLMService:
    def __init__(self, base_url: str, model_name: str, timeout_seconds: int, retries: int, mock_mode: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.mock_mode = mock_mode

    def health(self) -> bool:
        if self.mock_mode:
            return False
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=min(self.timeout_seconds, 3))
            response.raise_for_status()
            return True
        except Exception:
            return False

    def _mock_completion(self, prompt: str) -> LLMResponse:
        # Extract question and context from prompt if structured
        context_block = ""
        if "Context:" in prompt:
            parts = prompt.split("Context:", 1)
            after_ctx = parts[1]
            if "\nQuestion:" in after_ctx:
                context_block = after_ctx.split("\nQuestion:", 1)[0].strip()
            elif "\n\n" in after_ctx:
                context_block = after_ctx.split("\n\nReturn")[0].strip()
            else:
                context_block = after_ctx.strip()

        if not context_block or context_block.lower() == "none":
            answer_body = "Insufficient evidence was found in the retrieved corpus to answer this question."
        else:
            # Extract meaningful factual sentences from retrieved context passages
            passages = [p.strip() for p in context_block.split("\n\n") if p.strip()]
            clean_lines = []
            for p in passages:
                # Strip out chunk prefix if present e.g. [doc:chunk]
                cleaned = p
                if cleaned.startswith("[") and "]" in cleaned:
                    cleaned = cleaned.split("]", 1)[1].strip()
                if len(cleaned) > 15:
                    clean_lines.append(cleaned)
            answer_body = "\n\n".join(clean_lines[:3]) if clean_lines else context_block[:400]

        text = (
            f"[DEVELOPMENT / MOCK MODE: Ollama Service Offline]\n\n"
            f"{answer_body}\n\n"
            f"*(Notice: This answer is synthesized directly from retrieved corpus passages in offline development mode because the local Ollama LLM is currently unreachable.)*"
        )
        return LLMResponse(
            text=text,
            model_name=f"{self.model_name} (Mock/Dev)",
            prompt_tokens=len(prompt.split()),
            completion_tokens=len(text.split()),
            total_tokens=len(prompt.split()) + len(text.split()),
            latency_ms=1.5,
            mock_mode=True,
            warnings=["mock_mode_active"],
        )

    def complete(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> LLMResponse:
        if self.mock_mode:
            return self._mock_completion(prompt)

        start = perf_counter()
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
            if response.status_code >= 400:
                raise ServiceUnavailableError(f"Ollama error {response.status_code}: {response.text[:200]}")
            data = response.json()
            latency_ms = (perf_counter() - start) * 1000.0
            message = data.get("message", {})
            text = message.get("content", "")
            usage = data.get("usage", {}) or {}
            return LLMResponse(
                text=text,
                model_name=data.get("model", self.model_name),
                prompt_tokens=usage.get("prompt_eval_count"),
                completion_tokens=usage.get("eval_count"),
                total_tokens=(usage.get("prompt_eval_count") or 0) + (usage.get("eval_count") or 0) or None,
                latency_ms=latency_ms,
                mock_mode=False,
                warnings=[],
            )
        except requests.exceptions.RequestException as exc:
            raise ServiceUnavailableError(
                f"LLM unavailable: check the configured Ollama service and model ({exc.__class__.__name__})"
            ) from exc

    def complete_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> dict:
        response = self.complete(prompt, system_prompt=system_prompt, temperature=temperature)
        try:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception as exc:
            raise StructuredOutputError(f"Could not parse JSON from LLM output: {response.text[:200]}") from exc
