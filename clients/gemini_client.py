from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class GeminiClient(ChatClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = get_env_or_raise(config.env)
        base = config.base_url.rstrip("/") if config.base_url else "https://generativelanguage.googleapis.com/v1beta"
        self.url = f"{base}/models/{config.model_id}:generateContent?key={self.api_key}"

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        payload = {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens},
        }
        response = httpx.post(self.url, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            texts = [part.get("text", "") for part in parts]
            return "".join(texts).strip()
        except (KeyError, IndexError) as exc:
            raise RuntimeError(f"Gemini response parsing failed: {data}") from exc

