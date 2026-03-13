from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class DoubaoClient(ChatClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = get_env_or_raise(config.env)
        self.url = config.base_url or "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        payload = {
            "model": self.config.model_id,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": temperature, "max_new_tokens": max_tokens},
        }
        response = httpx.post(self.url, headers=self._headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        output = data.get("output") or {}
        if "choices" in output:
            try:
                segments = output["choices"][0]["message"]["content"]
                text = "".join(part.get("text", "") for part in segments)
                return text.strip()
            except (KeyError, IndexError) as exc:
                raise RuntimeError(f"Doubao response parsing failed: {data}") from exc
        if "text" in output:
            return str(output["text"]).strip()
        raise RuntimeError(f"Unexpected Doubao payload: {data}")

