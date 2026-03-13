from __future__ import annotations

from typing import Dict, Type

from .anthropic_client import AnthropicClient
from .baidu_client import BaiduClient
from .base import ChatClient, ModelConfig
from .doubao_client import DoubaoClient
from .gemini_client import GeminiClient
from .openai_style import OpenAIStyleClient
from .qwen_client import QwenClient


PROVIDER_MAP: Dict[str, Type[ChatClient]] = {
    "openai": OpenAIStyleClient,
    "gemini": GeminiClient,
    "anthropic": AnthropicClient,
    "groq": OpenAIStyleClient,
    "deepseek": OpenAIStyleClient,
    "moonshot": OpenAIStyleClient,
    "doubao": DoubaoClient,
    "baidu": BaiduClient,
    "qwen": QwenClient,
    "siliconflow": OpenAIStyleClient,
}


def build_client(config: ModelConfig) -> ChatClient:
    provider = config.provider.lower()
    client_cls = PROVIDER_MAP.get(provider)
    if not client_cls:
        raise ValueError(f"Unsupported provider: {config.provider}")
    return client_cls(config)
