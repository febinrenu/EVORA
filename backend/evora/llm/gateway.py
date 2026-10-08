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
import re
import time
from collections.abc import Callable
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from evora.llm.keypool import KeyPool
from evora.llm.ollama import OllamaClient
from evora.llm.replay import ReplayStore
from evora.llm.schemas import GROQ_BASE_URL, GatewayConfig, LLMError

log = logging.getLogger("evora.llm")

NTFY_TIMEOUT_S = 5.0
# when the configured local vision model is not installed, installed models are tried in this order of name
VISION_MODEL_PREFERENCE = ("qwen3-vl", "qwen3.5", "qwen2.5vl", "gemma3", "llama3.2-vision", "llava", "minicpm-v")
VISION_REASONING_HEADROOM = 512  # tokens the local vision model spends reasoning before it answers
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_NTFY_TOPIC = re.compile(r"[A-Za-z0-9_-]{1,64}")

M = TypeVar("M", bound=BaseModel)


class _Unavailable(Exception):
    """Groq could not produce a usable result; the caller should fall back."""


def _approx_tokens(messages: list[dict]) -> int:
    return sum(len(str(m.get("content", ""))) for m in messages) // 4 + 1


def _validated(schema: type[M], text: str) -> M:
    return schema.model_validate_json(_strip_fences(text))


def _answer_only(raw: str) -> str | None:
    """The model's answer without its reasoning, or None when there is no answer.

    A reply that opens a reasoning block and never closes it means the budget ran out while the model was still
    thinking: whatever follows is reasoning, not an answer.
    """
    text = _THINK_BLOCK.sub("", raw)
    if re.search(r"<think>", text, re.IGNORECASE):
        return None
    text = text.strip()
    return text or None


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
        self._vision_model = config.local_vision_model  # replaced once if that model turns out not to be installed
        self._vision_fallback_tried = False
        self._replay = (
            ReplayStore(config.replay_path, config.replay_mode)  # type: ignore[arg-type]
            if config.replay_path is not None and config.replay_mode != "off"
            else None
        )

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
            started = time.perf_counter()
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
            latency_ms = round((time.perf_counter() - started) * 1000)
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
            started = time.perf_counter()
            text = await self._ollama.chat_json(model, messages, json_schema)
            try:
                result = _validated(schema, text)
            except (ValidationError, ValueError) as exc:
                last_error = str(exc)
                self._log(task=task, backend="ollama", model=model, ok=False,
                          latency_ms=round((time.perf_counter() - started) * 1000))
                messages = _repair_messages(messages, text, last_error)
                continue
            self._log(task=task, backend="ollama", model=model, ok=True,
                      latency_ms=round((time.perf_counter() - started) * 1000))
            return result
        raise LLMError(f"local model gave no valid {schema.__name__}: {last_error[:200]}")

    # ----------------------------------------------------------------- public
    async def chat_json_ex(self, task: str, messages: list[dict], schema: type[M]) -> tuple[M, str]:
        """Like `chat_json` but also reports which backend answered: 'groq' or 'local'.

        With a replay store a recorded answer is returned with the backend that originally gave it.
        """
        store = self._replay
        key = store.key(task, messages, schema.__name__) if store is not None else None
        if store is not None and key is not None:  # note: an empty store is falsy, so never test `if store`
            hit = store.get(key)
            if hit is not None:
                try:
                    result = _validated(schema, hit["text"])
                    self._log(task=task, backend="replay", model=None, ok=True)
                    return result, hit["backend"]
                except (ValidationError, ValueError):
                    log.warning("recorded answer for %s no longer validates; treating as a miss", task)
            if store.mode == "replay":
                raise LLMError(f"no recorded answer for this {task} request (replay mode makes no live calls)")

        result, backend = await self._live_json(task, messages, schema)
        if store is not None and key is not None:
            store.put(key, task, result.model_dump_json(), backend)
        return result, backend

    async def _live_json(self, task: str, messages: list[dict], schema: type[M]) -> tuple[M, str]:
        if not self._onprem():
            try:
                return await self._groq_json(task, messages, schema), "groq"
            except _Unavailable as exc:
                log.info("groq unavailable for %s (%s); using local model", task, exc)
        return await self._local_json(task, messages, schema), "local"

    async def chat_json(self, task: str, messages: list[dict], schema: type[M]) -> M:
        return (await self.chat_json_ex(task, messages, schema))[0]

    @property
    def look_model(self) -> str:
        """The local model that answers questions about frames."""
        return self._cfg.local_look_model

    @staticmethod
    def _is_missing_model(exc: LLMError) -> bool:
        text = str(exc).lower()
        return "404" in text or "not found" in text

    async def _installed_vision_model(self) -> str | None:
        """An installed model that can see images, preferring the usual families; None if there is none."""
        names = await self._ollama.installed_models()
        ranked = sorted(names, key=lambda n: next((i for i, p in enumerate(VISION_MODEL_PREFERENCE) if p in n.lower()),
                                                  len(VISION_MODEL_PREFERENCE)))
        for name in ranked:
            if await self._ollama.can_see(name):
                return name
        return None

    async def _use_installed_vision_model(self, missing: str) -> bool:
        """After `missing` failed as not installed, switch to one that is (once). True when there is a new model."""
        if self._vision_fallback_tried:
            return False
        self._vision_fallback_tried = True
        found = await self._installed_vision_model()
        if found is None or found == missing:
            log.warning("no local vision model is installed (wanted %s); `ollama pull qwen3-vl:4b` enables captions "
                        "and clock reading", missing)
            return False
        log.warning("local vision model %s is not installed; using %s", missing, found)
        self._vision_model = found
        return True

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
            try:
                text = await self._ollama.vision_json(self._vision_model, image_jpeg, prompt)
            except LLMError as exc:
                if not (self._is_missing_model(exc) and await self._use_installed_vision_model(self._vision_model)):
                    raise
                text = await self._ollama.vision_json(self._vision_model, image_jpeg, prompt)
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

    async def notify(self, topic: str, title: str, message: str) -> None:
        """Send a text-only phone notification through ntfy (alerts, rule 9: all egress is here).

        Refuses in on-prem mode. The topic works like a password, so neither it nor the message is
        ever logged or put in an error: failures say only what went wrong.
        """
        if self._onprem():
            raise LLMError("notifications are disabled in on-prem mode")
        if not _NTFY_TOPIC.fullmatch(topic or ""):
            raise LLMError("invalid ntfy topic")
        safe_title = title.encode("latin-1", "replace").decode("latin-1")[:100]  # header values are latin-1
        try:
            resp = await self._client.post(
                f"{self._cfg.ntfy_base.rstrip('/')}/{topic}",
                content=message[:4000].encode("utf-8"),
                headers={"Title": safe_title, "Content-Type": "text/plain; charset=utf-8"},
                timeout=NTFY_TIMEOUT_S,
            )
        except httpx.HTTPError as exc:
            self._log(task="notify", backend="ntfy", ok=False, error=type(exc).__name__)
            raise LLMError("ntfy was unreachable") from None  # the exception text would contain the topic
        ok = resp.status_code < 400
        self._log(task="notify", backend="ntfy", ok=ok, status=resp.status_code)
        if not ok:
            raise LLMError(f"ntfy refused the notification (HTTP {resp.status_code})")

    async def vision_text(self, image_jpeg: bytes, prompt: str, *, local_only: bool = True,
                          max_tokens: int = 64, model: str | None = None) -> str | None:
        """One free-text answer about an image (captions, reading a burned-in clock, answering a question), or None.

        By default only the local vision model is used, never the cloud, because these images show people;
        `local_only=False` allows a Groq vision fallback and is never used in on-prem mode. The model reasons
        before it answers even with thinking off, so the output budget is `max_tokens` plus headroom for that,
        and any reasoning block is removed from the reply. `model` picks a local model for this call; if it is not
        installed the configured vision model is used instead. Neither the prompt nor the answer is logged.
        """
        if not image_jpeg or not prompt.strip():
            return None
        started = time.perf_counter()
        budget = max_tokens + VISION_REASONING_HEADROOM
        local = model or self._vision_model
        try:
            raw = None
            if model is not None and model != self._vision_model:
                try:
                    raw = await self._ollama.vision_text(model, image_jpeg, prompt, budget)
                except LLMError as exc:
                    if not self._is_missing_model(exc):
                        raise
                    local = self._vision_model  # the asked-for model is not installed: the configured one answers
            if raw is None:
                try:
                    raw = await self._ollama.vision_text(local, image_jpeg, prompt, budget)
                except LLMError as exc:
                    if not (self._is_missing_model(exc) and await self._use_installed_vision_model(local)):
                        raise
                    local = self._vision_model
                    raw = await self._ollama.vision_text(local, image_jpeg, prompt, budget)
            text = _answer_only(raw)
            self._log(task="vision_text", backend="ollama", model=local, ok=text is not None,
                      latency_ms=round((time.perf_counter() - started) * 1000))
            if text is not None or local_only or self._onprem():
                return text
        except LLMError as exc:
            self._log(task="vision_text", backend="ollama", model=local, ok=False, error=type(exc).__name__)
            if local_only or self._onprem():
                return None
        return await self._cloud_vision_text(image_jpeg, prompt, max_tokens)

    async def _cloud_vision_text(self, image_jpeg: bytes, prompt: str, max_tokens: int) -> str | None:
        model = self._cfg.groq_vision_model
        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_jpeg).decode("ascii")
        body = {
            "model": model, "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": prompt}, {"type": "image_url", "image_url": {"url": data_uri}}]}],
        }
        got = await self._groq_post("/chat/completions", model, 2048 + 300, "vision_text", json=body)
        if got is None or got[0].status_code != 200:
            return None
        resp, idx = got
        self._log_usage("vision_text", model, idx, resp, ok=True)
        try:
            return _answer_only(resp.json()["choices"][0]["message"]["content"] or "")
        except (KeyError, IndexError, ValueError):
            return None

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
