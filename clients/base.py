from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelConfig:
    name: str  # unique key including version
    display_name: str
    provider: str
    model_id: str
    env: str
    base_url: Optional[str] = None
    env_secret: Optional[str] = None
    token_url: Optional[str] = None
    family: str | None = None
    version: str | None = None
    version_label: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


class ChatClient:
    """Base synchronous chat completion client."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        raise NotImplementedError
