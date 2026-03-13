from __future__ import annotations

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import ChatClient, ModelConfig
from .helpers import get_env_or_raise


class BaiduClient(ChatClient):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.chat_url = config.base_url or "https://qianfan.baidubce.com/v2/chat/completions"
        self.auth_mode = (config.extra.get("auth_mode") or "token").lower()
        self._headers = {"Content-Type": "application/json"}

        if self.auth_mode == "bearer":
            token = get_env_or_raise(config.env)
            if not token.lower().startswith("bearer "):
                token = f"Bearer {token}"
            self._headers["Authorization"] = token
            self._token_refresh_needed = False
        else:
            if not config.env_secret:
                raise RuntimeError("BAIDU_SECRET_KEY environment variable is required for Turbo token mode.")
            self.api_key = get_env_or_raise(config.env)
            self.secret_key = get_env_or_raise(config.env_secret)
            self.token_url = (
                config.token_url or "https://aip.baidubce.com/oauth/2.0/token?grant_type=client_credentials"
            )
            self._access_token: str | None = None
            self._token_refresh_needed = True

    def _refresh_token(self) -> str:
        url = (
            f"{self.token_url}"
            f"&client_id={self.api_key}"
            f"&client_secret={self.secret_key}"
        )
        response = httpx.post(url, timeout=30)
        response.raise_for_status()
        data = response.json()
        token = data.get("access_token")
        if not token:
            raise RuntimeError(f"Unable to fetch Baidu access token: {data}")
        self._access_token = token
        return token

    def _ensure_token(self) -> str:
        return self._access_token or self._refresh_token()

    def _request(self, prompt: str, temperature: float, max_tokens: int) -> httpx.Response:
        headers = dict(self._headers)
        url = self.chat_url
        if self._token_refresh_needed:
            token = self._ensure_token()
            connector = "&" if "?" in url else "?"
            url = f"{url}{connector}access_token={token}"
        payload = {
            "model": self.config.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
        }
        return httpx.post(url, headers=headers, json=payload, timeout=60)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=6))
    def send(self, prompt: str, temperature: float = 0.8, max_tokens: int = 800) -> str:
        response = self._request(prompt, temperature, max_tokens)
        if response.status_code == 401 and self._token_refresh_needed:
            self._refresh_token()
            response = self._request(prompt, temperature, max_tokens)
        response.raise_for_status()
        data = response.json()
        if "error_code" in data and data["error_code"] != 0:
            raise RuntimeError(f"Baidu Turbo error: {data}")
        if "result" in data:
            return str(data["result"]).strip()
        if "output" in data and "text" in data["output"]:
            return str(data["output"]["text"]).strip()
        if isinstance(data.get("choices"), list):
            try:
                return data["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                pass
        raise RuntimeError(f"Unexpected Baidu payload: {data}")
