"""Narrow OpenRouter API client with explicit JSON output handling."""

import json
from abc import ABC, abstractmethod
from typing import Any

import httpx

from backend.app.core.exceptions import ConfigurationError, ExternalServiceError


class OpenRouterClient(ABC):
    @abstractmethod
    async def complete_json(self, *, system_prompt: str, payload: dict[str, Any], model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        """Return a decoded JSON object from a selected OpenRouter model."""


class HttpOpenRouterClient(OpenRouterClient):
    def __init__(self, api_key: str | None, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def complete_json(self, *, system_prompt: str, payload: dict[str, Any], model: str, temperature: float, max_tokens: int) -> dict[str, Any]:
        if not self.api_key:
            raise ConfigurationError("TEN_OPENROUTER_API_KEY is required for AI scoring")
        request = {
            "model": model,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": json.dumps(payload)}],
        }
        try:
            async with httpx.AsyncClient(timeout=self.timeout_seconds) as client:
                response = await client.post(f"{self.base_url}/chat/completions", headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}, json=request)
                response.raise_for_status()
                content = response.json()["choices"][0]["message"]["content"]
                parsed = json.loads(content)
        except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError) as exc:
            raise ExternalServiceError("OpenRouter returned an invalid scoring response") from exc
        if not isinstance(parsed, dict):
            raise ExternalServiceError("OpenRouter scoring response must be a JSON object")
        return parsed

