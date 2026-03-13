from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class QwenClient(ChatClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.api_key = get_env_or_raise(config.env)
        self.url = (
            config.base_url
            or "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
        )
        self._headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        payload = {
            "model": self.config.model_id,
            "input": {"messages": [{"role": "user", "content": prompt}]},
            "parameters": {"temperature": temperature, "max_tokens": max_tokens},
        }
        response = httpx.post(self.url, headers=self._headers, json=payload, timeout=60)
        response.raise_for_status()
        data = response.json()
        output = data.get("output", {})
        if "choices" in output:
            choice = output["choices"][0]
            message = choice.get("message", {})
            content = message.get("content")
            if isinstance(content, list):
                text = "".join(part.get("text", "") for part in content)
            else:
                text = content or ""
            return text.strip()
        if "text" in output:
            return str(output["text"]).strip()
        raise RuntimeError(f"Unexpected Qwen payload: {data}")

