"""OpenAI-compatible chat client for OpenRouter, with 429 backoff.

Free-tier models sit behind a shared rate-limited pool, so every call
retries with exponential backoff and honors Retry-After when present.
Configuration comes from env (see config.py); no secrets in code.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass

import requests

log = logging.getLogger("llm")

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass
class ChatResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str


class LLMClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        max_retries: int = 5,
        timeout_s: int = 120,
    ):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set - copy .env.example to .env")
        self.base_url = (base_url or os.environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.max_retries = max_retries
        self.timeout_s = timeout_s

    def chat(
        self,
        messages: list[dict],
        model: str,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        response_format: dict | None = None,
    ) -> ChatResult:
        payload: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if response_format:
            payload["response_format"] = response_format

        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                r = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self.api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                    timeout=self.timeout_s,
                )
                if r.status_code == 429:
                    retry_after = r.headers.get("Retry-After")
                    wait = float(retry_after) if retry_after else min(2 ** attempt * 5, 60)
                    log.warning("rate limited (429), retrying in %.0fs", wait)
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                data = r.json()
                choice = data["choices"][0]
                usage = data.get("usage") or {}
                return ChatResult(
                    text=(choice["message"].get("content") or "").strip(),
                    prompt_tokens=int(usage.get("prompt_tokens", 0)),
                    completion_tokens=int(usage.get("completion_tokens", 0)),
                    model=data.get("model", model),
                )
            except requests.RequestException as e:
                last_err = e
                wait = min(2 ** attempt * 5, 60)
                log.warning("LLM request failed (%s), retrying in %.0fs", e, wait)
                time.sleep(wait)
        raise RuntimeError(f"LLM call failed after {self.max_retries} attempts: {last_err}")
