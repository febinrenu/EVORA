"""The single place where product code talks to language models.

Every outbound connection of the backend goes through this module so the privacy
guard can reason about egress. Routing: Groq (key pool) first, the local Ollama
model when Groq is unavailable, rate limited or returns something unusable, and
Ollama only when the on-prem switch is on.
"""
from __future__ import annotations

import base64
import json
import logging
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from evora.llm.keypool import KeyPool
from evora.llm.ollama import OllamaClient
from evora.llm.schemas import GROQ_BASE_URL, GatewayConfig, LLMError

log = logging.getLogger("evora.llm")

M = TypeVar("M", bound=BaseModel)


class _Unavailable(Exception):
    """Groq could not produce a usable result; the caller should fall back."""


def _approx_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 1


def _validated(schema: type[M], text: str) -> M:
    return schema.model_validate_json(_strip_fences(text))


def _retry_after(resp: httpx.Response) -> float | None:
    try:
        return float(resp.headers["retry-after"])
    except (KeyError, ValueError):
        return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    return text.strip()


def _repair_messages(messages: list[dict], bad_reply: str, error: str) -> list[dict]:
    return [
        *messages,
        {"role": "assistant", "content": bad_reply},
        {
            "role": "user",
            "content": f"That reply was not valid for the schema ({error[:300]}). "
            "Reply again with only the corrected JSON.",
        },
    ]


class Gateway:
    def __init__(
        self,
        config: GatewayConfig,
        pool: KeyPool,
        client: httpx.AsyncClient,
        onprem: Callable[[], bool] = lambda: False,
    ) -> None:
        self._cfg = config
        self._pool = pool
        self._client = client
        self._onprem = onprem
        self._ollama = OllamaClient(config.ollama_host, client)

    # ----------------------------------------------------------------- logging
    def _log(self, **record: Any) -> None:
        record["ts"] = time.time()
        log.info("llm %s", record)
        path = self._cfg.log_path
        if path is None:
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
        except OSError as exc:
            log.warning("could not write %s: %s", path, exc)

    # -------------------------------------------------------------- groq plumbing
    async def _groq_post(
        self, path: str, model: str, need_tokens: int, task: str, **request: Any
    ) -> tuple[httpx.Response, int] | None:
        tried: set[int] = set()
        while True:
            idx = self._pool.pick(model, need_tokens)
            if idx is None or idx in tried:
                return None
            tried.add(idx)
            headers = {"Authorization": f"Bearer {self._pool.key(idx)}"}
            started = time.monotonic()
            try:
                resp = await self._client.post(
                    f"{GROQ_BASE_URL}{path}", headers=headers, timeout=self._cfg.timeout_s, **request
                )
            except httpx.HTTPError as exc:
                self._pool.record_failure(idx, model)
                self._log(task=task, backend="groq", model=model, key=self._pool.label(idx), ok=False,
                          error=type(exc).__name__)
                continue
            self._pool.update_from_headers(idx, model, resp.headers)
            latency_ms = round((time.monotonic() - started) * 1000)
            if resp.status_code == 429:
                self._pool.record_rate_limited(idx, model, _retry_after(resp))
                self._log(task=task, backend="groq", model=model, key=self._pool.label(idx), ok=False,
                          status=429, latency_ms=latency_ms)
                continue
            if resp.status_code in (401, 403):
                self._pool.record_rate_limited(idx, model, 300.0)  # bad key: sideline it
                self._log(task=task, backend="groq", model=model, key=self._pool.label(idx), ok=False,
                          status=resp.status_code)
                continue
            if resp.status_code >= 500:
                self._pool.record_failure(idx, model)
                self._log(task=task, backend="groq", model=model, key=self._pool.label(idx), ok=False,
                          status=resp.status_code, latency_ms=latency_ms)
                continue
            self._pool.record_success(idx, model)
            return resp, idx

    def _log_usage(self, task: str, model: str, idx: int, resp: httpx.Response, ok: bool) -> None:
        usage: dict = {}
        try:
            usage = resp.json().get("usage", {}) or {}
        except ValueError:
            pass
        details = usage.get("prompt_tokens_details") or {}
        self._log(
            task=task, backend="groq", model=model, key=self._pool.label(idx), ok=ok,
            prompt_tokens=usage.get("prompt_tokens"), completion_tokens=usage.get("completion_tokens"),
            cached_tokens=details.get("cached_tokens"),
        )

    async def _groq_json(self, task: str, messages: list[dict], schema: type[M]) -> M:
        model = self._cfg.groq_model_for(task)
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "response_format": (
                {"type": "json_object"}
                if task in self._cfg.json_object_tasks
                else {
                    "type": "json_schema",
                    "json_schema": {"name": schema.__name__, "schema": schema.model_json_schema()},
                }
            ),
        }
        if "gpt-oss" in model:
            body["reasoning_effort"] = self._cfg.reasoning_effort

        for attempt in range(2):  # second attempt is the single repair retry
            got = await self._groq_post("/chat/completions", model, _approx_tokens(messages), task, json=body)
            if got is None:
                raise _Unavailable("no key available")
            resp, idx = got
            schema_mode = body["response_format"]["type"] == "json_schema"
            if resp.status_code == 400 and "response_format" in resp.text and schema_mode:
                body["response_format"] = {"type": "json_object"}  # model lacks schema mode
                got = await self._groq_post("/chat/completions", model, _approx_tokens(messages), task, json=body)
                if got is None:
                    raise _Unavailable("no key available")
                resp, idx = got
            if resp.status_code != 200:
                self._log_usage(task, model, idx, resp, ok=False)
                raise _Unavailable(f"groq status {resp.status_code}")
            try:
                text = resp.json()["choices"][0]["message"]["content"] or ""
            except (KeyError, IndexError, ValueError) as exc:
                raise _Unavailable("malformed groq reply") from exc
            try:
                result = _validated(schema, text)
            except (ValidationError, ValueError) as exc:
                self._log_usage(task, model, idx, resp, ok=False)
                if attempt == 1:
                    raise _Unavailable("invalid after repair") from exc
                messages = _repair_messages(messages, text, str(exc))
                body["messages"] = messages
                continue
            self._log_usage(task, model, idx, resp, ok=True)
            return result
        raise _Unavailable("unreachable")

    # ------------------------------------------------------------------ local
    async def _local_json(self, task: str, messages: list[dict], schema: type[M]) -> M:
        model = self._cfg.local_text_model
        json_schema = schema.model_json_schema()
        last_error = ""
        for _ in range(2):
            started = time.monotonic()
            text = await self._ollama.chat_json(model, messages, json_schema)
            try:
                result = _validated(schema, text)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                self._log(task=task, backend="ollama", model=model, ok=False,
                          latency_ms=round((time.monotonic() - started) * 1000))
                messages = _repair_messages(messages, text, last_error)
                continue
            self._log(task=task, backend="ollama", model=model, ok=True,
                      latency_ms=round((time.monotonic() - started) * 1000))
            return result
        raise LLMError(f"local model gave no valid {schema.__name__}: {last_error[:200]}")

    # ----------------------------------------------------------------- public
    async def chat_json_ex(self, task: str, messages: list[dict], schema: type[M]) -> tuple[M, str]:
        """Like `chat_json` but also reports which backend answered: 'groq' or 'local'."""
        if not self._onprem():
            try:
                return await self._groq_json(task, messages, schema), "groq"
            except _Unavailable as exc:
                log.info("groq unavailable for %s (%s); using local model", task, exc)
        return await self._local_json(task, messages, schema), "local"

    async def chat_json(self, task: str, messages: list[dict], schema: type[M]) -> M:
        return (await self.chat_json_ex(task, messages, schema))[0]

    async def vision_yesno(self, image_jpeg: bytes, questions: list[str]) -> list[bool | None]:
        n = len(questions)
        if n == 0:
            return []
        numbered = "\n".join(f"{i + 1}. {q}" for i, q in enumerate(questions))
        prompt = (
            "Look at the image and answer each numbered question with true or false. "
            "Use null when you cannot tell.\n"
            f"{numbered}\n"
            f'Reply only with JSON: {{"answers": [...]}} containing exactly {n} items.'
        )
        try:
            text = await self._ollama.vision_json(self._cfg.local_vision_model, image_jpeg, prompt)
            return _parse_answers(text, n)
        except LLMError as exc:
            log.info("local vision failed (%s)", exc)
        if self._onprem():
            return [None] * n
        model = self._cfg.groq_vision_model
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        body = {
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}},
            ]}],
        }
        got = await self._groq_post("/chat/completions", model, 2048 + 300, "vision", json=body)
        if got is None or got[0].status_code != 200:
            return [None] * n
        resp, idx = got
        self._log_usage("vision", model, idx, resp, ok=True)
        try:
            return _parse_answers(resp.json()["choices"][0]["message"]["content"], n)
        except (KeyError, IndexError, ValueError):
            return [None] * n

    async def transcribe(self, audio: bytes) -> str:
        if self._onprem():
            raise LLMError("transcription needs a cloud model; unavailable in on-prem mode")
        model = self._cfg.groq_whisper_model
        got = await self._groq_post(
            "/audio/transcriptions", model, 0, "transcribe",
            files={"file": ("audio.webm", audio)}, data={"model": model, "response_format": "json"},
        )
        if got is None or got[0].status_code != 200:
            raise LLMError("transcription unavailable")
        try:
            return str(got[0].json()["text"]).strip()
        except (KeyError, ValueError) as exc:
            raise LLMError("transcription reply had no text") from exc


def _coerce(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes"):
            return True
        if v in ("false", "no"):
            return False
    return None


def _parse_answers(text: str, n: int) -> list[bool | None]:
    try:
        data = json.loads(_strip_fences(text))
        raw = data["answers"] if isinstance(data, dict) else data
        answers = [_coerce(v) for v in raw]
    except (ValueError, KeyError, TypeError):
        return [None] * n
    return (answers + [None] * n)[:n]
