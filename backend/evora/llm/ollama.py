"""Thin Ollama client used for the on-prem path and as the fallback for Groq."""
from __future__ import annotations

import base64
from typing import Any

import httpx

from evora.llm.schemas import LLMError

MAX_NEW_TOKENS = 700  # plans and yes/no lists are short; a hard cap keeps a runaway model from stalling a query


class OllamaClient:
    def __init__(self, host: str, client: httpx.AsyncClient, timeout_s: float = 120.0) -> None:
        self._host = host.rstrip("/")
        self._client = client
        self._timeout = timeout_s

    async def _chat(self, payload: dict[str, Any]) -> str:
        try:
            resp = await self._client.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        try:
            return str(resp.json()["message"]["content"])
        except (KeyError, ValueError) as exc:
            raise LLMError("ollama reply had no message content") from exc

    async def chat_json(self, model: str, messages: list[dict], json_schema: dict | None) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "format": json_schema if json_schema is not None else "json",
            "think": False,  # reasoning models otherwise spend the whole budget thinking and cut the JSON off
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS},
        }
        return await self._chat(payload)

    async def vision_json(self, model: str, image_jpeg: bytes, prompt: str) -> str:
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [base64.b64encode(image_jpeg).decode("ascii")]}
            ],
            "stream": False,
            "format": "json",
            "think": False,
            "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS},
        }
        return await self._chat(payload)
