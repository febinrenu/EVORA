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

    async def installed_models(self) -> list[str]:
        """Names of the models installed in Ollama; an empty list when it cannot be asked."""
        try:
            resp = await self._client.get(f"{self._host}/api/tags", timeout=10.0)
            return [m["name"] for m in resp.json().get("models", []) if isinstance(m, dict) and "name" in m]
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return []

    async def can_see(self, model: str) -> bool:
        """Does this model accept images? (Ollama reports a 'vision' capability.)"""
        try:
            resp = await self._client.post(f"{self._host}/api/show", json={"model": model}, timeout=10.0)
            return "vision" in (resp.json().get("capabilities") or [])
        except (httpx.HTTPError, ValueError, AttributeError, TypeError):
            return False

    async def _chat(self, payload: dict[str, Any]) -> str:
        try:
            resp = await self._client.post(f"{self._host}/api/chat", json=payload, timeout=self._timeout)
        except httpx.HTTPError as exc:
            raise LLMError(f"ollama unreachable: {exc}") from exc
        if resp.status_code != 200:
            raise LLMError(f"ollama returned {resp.status_code}: {resp.text[:200]}")
        try:
            message = resp.json()["message"]
            content = str(message["content"])
        except (KeyError, ValueError) as exc:
            raise LLMError("ollama reply had no message content") from exc
        if not content.strip() and message.get("thinking"):
            # reasoning models can put the whole reply in `thinking` when the answer format is constrained
            return str(message["thinking"])
        return content

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

    async def vision_text(self, model: str, image_jpeg: bytes, prompt: str, num_predict: int) -> str:
        """One free-text answer about an image. `num_predict` must leave room for the model's own reasoning."""
        payload = {
            "model": model,
            "messages": [
                {"role": "user", "content": prompt, "images": [base64.b64encode(image_jpeg).decode("ascii")]}
            ],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_predict": num_predict},
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
