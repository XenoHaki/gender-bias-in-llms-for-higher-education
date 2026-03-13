from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class OpenAIStyleClient(ChatClient):
    """OpenAI compatible chat completions client."""

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = get_env_or_raise(config.env)
        self.base_url = config.base_url or "https://api.openai.com/v1/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        payload = {
            "model": self.config.model_id,
            "messages": [
                {"role": "system", "content": "You are a precise assistant for bias diagnostics."},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = httpx.post(self.base_url, headers=self._headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"].strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Invalid response from {self.config.name}: {data}") from exc

