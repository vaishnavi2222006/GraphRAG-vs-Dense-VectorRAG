from __future__ import annotations

from dataclasses import dataclass
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
    warnings: list[str] = None


class OllamaLLMService:
    def __init__(self, base_url: str, model_name: str, timeout_seconds: int, retries: int, mock_mode: bool = False) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.retries = retries
        self.mock_mode = mock_mode

    def health(self) -> bool:
        if self.mock_mode:
            return True
        response = requests.get(f"{self.base_url}/api/tags", timeout=self.timeout_seconds)
        response.raise_for_status()
        return True

    def _mock_completion(self, prompt: str) -> LLMResponse:
        text = (
            "Mock response. The system is running in mock mode because Ollama is unavailable or disabled. "
            "This response should not be used as a real experiment result.\n\n"
            f"Prompt preview: {prompt[:180]}"
        )
        return LLMResponse(text=text, model_name=self.model_name, mock_mode=True, warnings=["mock_mode"])

    @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
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
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout_seconds)
        if response.status_code >= 400:
            raise ServiceUnavailableError(f"Ollama error: {response.status_code} {response.text[:200]}")
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

    def complete_json(self, prompt: str, system_prompt: str | None = None, temperature: float = 0.0) -> dict:
        response = self.complete(prompt, system_prompt=system_prompt, temperature=temperature)
        try:
            text = response.text.strip()
            if text.startswith("```"):
                text = text.strip("`")
            return json.loads(text[text.find("{") : text.rfind("}") + 1])
        except Exception as exc:
            raise StructuredOutputError(f"Could not parse JSON from LLM output: {response.text[:200]}") from exc

