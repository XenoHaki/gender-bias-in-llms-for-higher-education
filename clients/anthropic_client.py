from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class AnthropicClient(ChatClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = get_env_or_raise(config.env)
        self.url = "https://api.anthropic.com/v1/messages"
        self._headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        payload = {
            "model": self.config.model_id,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        response = httpx.post(self.url, headers=self._headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            content = data["content"][0]["text"]
            return content.strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Anthropic response parsing failed: {data}") from exc

