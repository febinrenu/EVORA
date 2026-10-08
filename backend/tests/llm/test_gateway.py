import json

import httpx
import pytest
from pydantic import BaseModel

from evora.llm.gateway import Gateway
from evora.llm.keypool import KeyPool
from evora.llm.schemas import GatewayConfig, LLMError


class Out(BaseModel):
    a: int


def groq_ok(content: str, headers: dict | None = None) -> httpx.Response:
    body = {
        "choices": [{"message": {"content": content}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 80}},
    }
    return httpx.Response(200, json=body, headers=headers or {})


def ollama_ok(content: str) -> httpx.Response:
    return httpx.Response(200, json={"message": {"content": content}})


class Recorder:
    def __init__(self, groq, ollama):
        self.groq, self.ollama = groq, ollama
        self.calls: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        handler = self.groq if request.url.host == "api.groq.com" else self.ollama
        return handler(request)

    def hosts(self):
        return [c.url.host for c in self.calls]

    def bodies(self, host):
        return [json.loads(c.content) for c in self.calls if c.url.host == host]


def make(rec: Recorder, keys=("gsk_key_aaaa", "gsk_key_bbbb"), onprem=False, tmp_path=None):
    cfg = GatewayConfig(log_path=(tmp_path / "llm.jsonl") if tmp_path else None)
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    return Gateway(cfg, KeyPool(list(keys)), client, onprem=lambda: onprem)


def no_ollama(_):
    raise AssertionError("ollama should not be called")


def no_groq(_):
    raise AssertionError("groq should not be called")


@pytest.mark.asyncio
async def test_groq_success_sends_schema_and_low_effort():
    rec = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    out, backend = await make(rec).chat_json_ex("planner", [{"role": "user", "content": "hi json"}], Out)
    assert out == Out(a=1) and backend == "groq"
    body = rec.bodies("api.groq.com")[0]
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["reasoning_effort"] == "low"
    assert body["response_format"]["type"] == "json_schema"


@pytest.mark.asyncio
async def test_describe_task_uses_large_model():
    rec = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    await make(rec).chat_json("describe", [{"role": "user", "content": "x"}], Out)
    assert rec.bodies("api.groq.com")[0]["model"] == "openai/gpt-oss-120b"


@pytest.mark.asyncio
async def test_429_rotates_to_next_key_in_same_call():
    def groq(req):
        if req.headers["authorization"].endswith("aaaa"):
            return httpx.Response(429, headers={"retry-after": "20"})
        return groq_ok('{"a": 2}')

    rec = Recorder(groq, no_ollama)
    gw = make(rec, keys=("gsk_key_aaaa", "gsk_key_bbbb"))
    out = await gw.chat_json("planner", [{"role": "user", "content": "x"}], Out)
    assert out.a == 2
    assert len(rec.calls) == 2
    # the limited key is sidelined for the next call
    await gw.chat_json("planner", [{"role": "user", "content": "x"}], Out)
    assert len(rec.calls) == 3
    assert rec.calls[-1].headers["authorization"].endswith("bbbb")


@pytest.mark.asyncio
async def test_all_keys_limited_falls_back_to_local():
    rec = Recorder(lambda r: httpx.Response(429), lambda r: ollama_ok('{"a": 3}'))
    out, backend = await make(rec).chat_json_ex("planner", [{"role": "user", "content": "x"}], Out)
    assert out.a == 3 and backend == "local"
    sent = rec.bodies("127.0.0.1")[0]
    assert sent["model"] == "qwen3.5:4b" and sent["stream"] is False
    assert isinstance(sent["format"], dict)


@pytest.mark.asyncio
async def test_invalid_reply_gets_one_repair_retry():
    replies = iter(['{"a": "not a number"}', '{"a": 7}'])
    rec = Recorder(lambda r: groq_ok(next(replies)), no_ollama)
    out = await make(rec).chat_json("planner", [{"role": "user", "content": "x"}], Out)
    assert out.a == 7
    second = rec.bodies("api.groq.com")[1]["messages"]
    assert second[-2]["role"] == "assistant" and "corrected JSON" in second[-1]["content"]


@pytest.mark.asyncio
async def test_still_invalid_after_repair_goes_local():
    rec = Recorder(lambda r: groq_ok("nonsense"), lambda r: ollama_ok('{"a": 9}'))
    out, backend = await make(rec).chat_json_ex("planner", [{"role": "user", "content": "x"}], Out)
    assert (out.a, backend) == (9, "local")
    assert rec.hosts().count("api.groq.com") == 2


@pytest.mark.asyncio
async def test_markdown_fenced_json_is_accepted():
    rec = Recorder(lambda r: groq_ok('```json\n{"a": 4}\n```'), no_ollama)
    assert (await make(rec).chat_json("planner", [{"role": "user", "content": "x"}], Out)).a == 4


@pytest.mark.asyncio
async def test_model_without_schema_mode_retries_with_json_object():
    def groq(req):
        body = json.loads(req.content)
        if body["response_format"]["type"] == "json_schema":
            return httpx.Response(400, text='{"error":"response_format json_schema unsupported"}')
        return groq_ok('{"a": 5}')

    rec = Recorder(groq, no_ollama)
    out = await make(rec).chat_json("planner", [{"role": "user", "content": "x"}], Out)
    assert out.a == 5
    assert rec.bodies("api.groq.com")[-1]["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_onprem_never_touches_groq():
    rec = Recorder(no_groq, lambda r: ollama_ok('{"a": 6}'))
    out, backend = await make(rec, onprem=True).chat_json_ex("planner", [{"role": "user", "content": "x"}], Out)
    assert (out.a, backend) == (6, "local")
    assert set(rec.hosts()) == {"127.0.0.1"}


@pytest.mark.asyncio
async def test_local_invalid_twice_raises():
    rec = Recorder(no_groq, lambda r: ollama_ok("garbage"))
    with pytest.raises(LLMError):
        await make(rec, onprem=True).chat_json("planner", [{"role": "user", "content": "x"}], Out)


@pytest.mark.asyncio
async def test_ollama_down_and_groq_down_raises():
    def boom(_):
        raise httpx.ConnectError("refused")

    rec = Recorder(boom, boom)
    with pytest.raises(LLMError):
        await make(rec).chat_json("planner", [{"role": "user", "content": "x"}], Out)


@pytest.mark.asyncio
async def test_log_records_cached_tokens_and_never_the_key(tmp_path):
    rec = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    await make(rec, tmp_path=tmp_path).chat_json("planner", [{"role": "user", "content": "x"}], Out)
    text = (tmp_path / "llm.jsonl").read_text()
    row = json.loads(text.splitlines()[-1])
    assert row["cached_tokens"] == 80 and row["backend"] == "groq" and row["ok"] is True
    assert "gsk_key" not in text


@pytest.mark.asyncio
async def test_headers_feed_the_pool():
    rec = Recorder(
        lambda r: groq_ok('{"a": 1}', {"x-ratelimit-remaining-tokens": "10", "x-ratelimit-reset-tokens": "40s"}),
        no_ollama,
    )
    gw = make(rec, keys=("gsk_key_aaaa", "gsk_key_bbbb"))
    await gw.chat_json("planner", [{"role": "user", "content": "x"}], Out)
    await gw.chat_json("planner", [{"role": "user", "content": "x"}], Out)
    # second call must prefer the key that still has headroom
    assert rec.calls[0].headers["authorization"] != rec.calls[1].headers["authorization"]


@pytest.mark.asyncio
async def test_vision_local_parses_answers():
    rec = Recorder(no_groq, lambda r: ollama_ok('{"answers": [true, "no", null]}'))
    got = await make(rec).vision_yesno(b"jpeg", ["a?", "b?", "c?"])
    assert got == [True, False, None]
    assert "images" in rec.bodies("127.0.0.1")[0]["messages"][0]


@pytest.mark.asyncio
async def test_vision_pads_short_answer_lists():
    rec = Recorder(no_groq, lambda r: ollama_ok('{"answers": [true]}'))
    assert await make(rec).vision_yesno(b"jpeg", ["a?", "b?"]) == [True, None]


@pytest.mark.asyncio
async def test_vision_falls_back_to_groq_when_local_down():
    def ollama_down(_):
        raise httpx.ConnectError("refused")

    rec = Recorder(lambda r: groq_ok('{"answers": [false, true]}'), ollama_down)
    got = await make(rec).vision_yesno(b"jpeg", ["a?", "b?"])
    assert got == [False, True]
    body = rec.bodies("api.groq.com")[0]
    assert body["model"] == "qwen/qwen3.8-27b"


@pytest.mark.asyncio
async def test_vision_onprem_with_local_down_returns_unknown():
    def ollama_down(_):
        raise httpx.ConnectError("refused")

    rec = Recorder(no_groq, ollama_down)
    assert await make(rec, onprem=True).vision_yesno(b"jpeg", ["a?"]) == [None]


@pytest.mark.asyncio
async def test_transcribe_groq_and_onprem_refusal():
    rec = Recorder(lambda r: httpx.Response(200, json={"text": " main gate "}), no_ollama)
    assert await make(rec).transcribe(b"audio") == "main gate"
    assert rec.calls[0].url.path.endswith("/audio/transcriptions")
    with pytest.raises(LLMError):
        await make(rec, onprem=True).transcribe(b"audio")
