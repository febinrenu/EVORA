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
    assert body["response_format"] == {"type": "json_object"}  # the planner prompt carries its own shape


@pytest.mark.asyncio
async def test_other_tasks_send_the_json_schema():
    rec = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    await make(rec).chat_json("describe", [{"role": "user", "content": "x"}], Out)
    fmt = rec.bodies("api.groq.com")[0]["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"]["properties"]["a"]["type"] == "integer"


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
    out = await make(rec).chat_json("describe", [{"role": "user", "content": "x"}], Out)
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
    big = [{"role": "user", "content": "x" * 8000}]  # about 2000 tokens, more than the 10 the first key has left
    await gw.chat_json("planner", big, Out)
    await gw.chat_json("planner", big, Out)
    # the second call must move to the key that still has headroom
    assert rec.calls[0].headers["authorization"] != rec.calls[1].headers["authorization"]


@pytest.mark.asyncio
async def test_sticks_to_one_key_while_it_has_room_so_the_prompt_cache_can_hit():
    rec = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    gw = make(rec, keys=("gsk_key_aaaa", "gsk_key_bbbb", "gsk_key_cccc"))
    for _ in range(5):
        await gw.chat_json("planner", [{"role": "user", "content": "x"}], Out)
    assert len({c.headers["authorization"] for c in rec.calls}) == 1


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


def make_replay(rec, tmp_path, mode):
    cfg = GatewayConfig(log_path=None, replay_path=tmp_path / "replay.jsonl", replay_mode=mode)
    client = httpx.AsyncClient(transport=httpx.MockTransport(rec))
    return Gateway(cfg, KeyPool(["gsk_key_aaaa"]), client)


MSG = [{"role": "user", "content": "same question"}]


@pytest.mark.asyncio
async def test_record_then_replay_makes_zero_live_calls(tmp_path):
    live = Recorder(lambda r: groq_ok('{"a": 11}'), no_ollama)
    assert (await make_replay(live, tmp_path, "record").chat_json("planner", MSG, Out)).a == 11
    assert len(live.calls) == 1

    offline = Recorder(no_groq, no_ollama)
    gw = make_replay(offline, tmp_path, "replay")
    out, backend = await gw.chat_json_ex("planner", MSG, Out)
    assert out.a == 11 and backend == "groq" and offline.calls == []  # same backend label as when recorded


@pytest.mark.asyncio
async def test_replay_mode_never_goes_live_on_a_miss(tmp_path):
    offline = Recorder(no_groq, no_ollama)
    with pytest.raises(LLMError, match="no recorded answer"):
        await make_replay(offline, tmp_path, "replay").chat_json("planner", MSG, Out)
    assert offline.calls == []


@pytest.mark.asyncio
async def test_auto_mode_replays_hits_and_records_misses(tmp_path):
    live = Recorder(lambda r: groq_ok('{"a": 5}'), no_ollama)
    gw = make_replay(live, tmp_path, "auto")
    await gw.chat_json("planner", MSG, Out)
    await gw.chat_json("planner", MSG, Out)
    assert len(live.calls) == 1  # the second was replayed
    await gw.chat_json("planner", [{"role": "user", "content": "different"}], Out)
    assert len(live.calls) == 2
    fresh = make_replay(Recorder(no_groq, no_ollama), tmp_path, "replay")  # a new process reads the file
    assert (await fresh.chat_json("planner", MSG, Out)).a == 5


@pytest.mark.asyncio
async def test_replay_key_depends_on_task_and_messages(tmp_path):
    live = Recorder(lambda r: groq_ok('{"a": 1}'), no_ollama)
    gw = make_replay(live, tmp_path, "auto")
    await gw.chat_json("planner", MSG, Out)
    await gw.chat_json("describe", MSG, Out)  # different task: not a hit
    assert len(live.calls) == 2


@pytest.mark.asyncio
async def test_off_by_default_and_bad_entries_are_ignored(tmp_path):
    live = Recorder(lambda r: groq_ok('{"a": 2}'), no_ollama)
    await make(live).chat_json("planner", MSG, Out)
    await make(live).chat_json("planner", MSG, Out)
    assert len(live.calls) == 2  # no replay unless configured
    path = tmp_path / "replay.jsonl"
    path.write_text("not json\n{\"key\": 1}\n")
    gw = make_replay(live, tmp_path, "auto")  # the damaged file does not stop startup
    await gw.chat_json("planner", MSG, Out)
    assert len(live.calls) == 3


# ------------------------------------------------------------------ notify
@pytest.mark.asyncio
async def test_notify_posts_the_message_to_the_topic(tmp_path):
    seen = []
    cfg = GatewayConfig(log_path=tmp_path / "llm.jsonl")
    transport = httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200))[1])
    gw = Gateway(cfg, KeyPool([]), httpx.AsyncClient(transport=transport))
    await gw.notify("secret-topic_123", "Alert: Gate", "Person at the gate — 09:14:03")
    (req,) = seen
    assert (req.method, req.url.host, req.url.path) == ("POST", "ntfy.sh", "/secret-topic_123")
    assert req.content.decode("utf-8") == "Person at the gate — 09:14:03"  # the UTF-8 body survives
    assert req.headers["title"] == "Alert: Gate"
    log_text = (tmp_path / "llm.jsonl").read_text()
    assert "secret-topic" not in log_text and "Person at the gate" not in log_text


@pytest.mark.asyncio
async def test_notify_goes_to_ntfy_not_to_groq_or_ollama():
    hosts = []

    def anywhere(request):
        hosts.append(request.url.host)
        return httpx.Response(200)

    cfg = GatewayConfig(log_path=None)
    gw = Gateway(cfg, KeyPool(["gsk_key_aaaa"]), httpx.AsyncClient(transport=httpx.MockTransport(anywhere)))
    await gw.notify("topic1", "t", "m")
    assert hosts == ["ntfy.sh"]  # never the model hosts, and never the Groq key
    cfg2 = GatewayConfig(log_path=None, ntfy_base="http://127.0.0.1:2586/")
    gw2 = Gateway(cfg2, KeyPool([]), httpx.AsyncClient(transport=httpx.MockTransport(anywhere)))
    await gw2.notify("topic1", "t", "m")
    assert hosts[-1] == "127.0.0.1"  # a self-hosted ntfy works too


@pytest.mark.asyncio
async def test_notify_is_refused_in_onprem_mode_without_any_request():
    def boom(_):
        raise AssertionError("no request may be made")

    gw = Gateway(GatewayConfig(log_path=None), KeyPool([]), httpx.AsyncClient(transport=httpx.MockTransport(boom)),
                 onprem=lambda: True)
    with pytest.raises(LLMError, match="on-prem"):
        await gw.notify("topic1", "t", "m")


@pytest.mark.asyncio
@pytest.mark.parametrize("topic", ["", "has space", "../etc", "a/b", "x" * 65, "bad?query=1"])
async def test_notify_rejects_topics_that_could_change_the_url(topic):
    def boom(_):
        raise AssertionError("no request may be made")

    gw = Gateway(GatewayConfig(log_path=None), KeyPool([]), httpx.AsyncClient(transport=httpx.MockTransport(boom)))
    with pytest.raises(LLMError, match="invalid ntfy topic"):
        await gw.notify(topic, "t", "m")


@pytest.mark.asyncio
async def test_notify_failures_never_contain_the_topic_or_message(tmp_path):
    def down(_):
        raise httpx.ConnectError("could not reach https://ntfy.sh/secret-topic")

    cfg = GatewayConfig(log_path=tmp_path / "llm.jsonl")
    gw = Gateway(cfg, KeyPool([]), httpx.AsyncClient(transport=httpx.MockTransport(down)))
    with pytest.raises(LLMError) as err:
        await gw.notify("secret-topic", "Alert", "private message text")
    assert "secret-topic" not in str(err.value) and err.value.__cause__ is None
    assert "secret-topic" not in (tmp_path / "llm.jsonl").read_text()

    gw500 = Gateway(cfg, KeyPool([]), httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(429))))
    with pytest.raises(LLMError, match="HTTP 429") as err2:
        await gw500.notify("secret-topic", "Alert", "private message text")
    assert "secret-topic" not in str(err2.value) and "private message" not in str(err2.value)


@pytest.mark.asyncio
async def test_notify_survives_titles_and_messages_that_are_awkward():
    seen = []
    gw = Gateway(GatewayConfig(log_path=None), KeyPool([]),
                 httpx.AsyncClient(transport=httpx.MockTransport(lambda r: (seen.append(r), httpx.Response(200))[1])))
    await gw.notify("t1", "Alert — 😀 " + "x" * 300, "m" * 9000)
    req = seen[0]
    assert req.headers["title"] == ("Alert ? ? " + "x" * 300)[:100]  # characters a header cannot carry become "?"
    assert len(req.content) == 4000


# -------------------------------------------------------------- vision_text
def ollama_text(content):
    return lambda r: httpx.Response(200, json={"message": {"content": content}})


@pytest.mark.asyncio
async def test_vision_text_asks_the_local_model_for_free_text_with_room_to_reason(tmp_path):
    seen = []
    rec = Recorder(no_groq, lambda r: (seen.append(json.loads(r.content)), ollama_text("A man in a red coat.")(r))[1])
    gw = make(rec, tmp_path=tmp_path)
    out = await gw.vision_text(b"jpeg-bytes", "Describe the person in one sentence.", max_tokens=64)
    assert out == "A man in a red coat."
    body = seen[0]
    assert body["model"] == "qwen3-vl:2b" and body["stream"] is False and body["think"] is False
    assert "format" not in body  # free text, not JSON
    assert body["options"]["num_predict"] >= 500  # with 32 the reply is empty: the model reasons before answering
    assert body["options"]["num_predict"] == 64 + 512
    msg = body["messages"][0]
    assert msg["content"] == "Describe the person in one sentence." and len(msg["images"]) == 1
    assert rec.hosts() == ["127.0.0.1"]  # nothing left the machine
    log = (tmp_path / "llm.jsonl").read_text()
    assert "Describe the person" not in log and "red coat" not in log  # prompts and answers are never logged


@pytest.mark.asyncio
async def test_vision_text_removes_the_reasoning_block():
    reply = "<think>The image shows... a clock reading 10:42.</think>\n\n  2018-03-09T10:42:07  "
    gw = make(Recorder(no_groq, ollama_text(reply)))
    assert await gw.vision_text(b"x", "Read the time.") == "2018-03-09T10:42:07"
    multi = "<think>a</think>first<think>b</think> second"
    assert await make(Recorder(no_groq, ollama_text(multi))).vision_text(b"x", "p") == "first second"


@pytest.mark.asyncio
async def test_a_reply_that_never_finishes_thinking_or_is_empty_is_none_not_reasoning():
    cut_off = "<think>Let me look at the image carefully. The top left shows"  # the budget ran out mid-thought
    assert await make(Recorder(no_groq, ollama_text(cut_off))).vision_text(b"x", "p") is None
    assert await make(Recorder(no_groq, ollama_text(""))).vision_text(b"x", "p") is None
    assert await make(Recorder(no_groq, ollama_text("<think>only thoughts</think>"))).vision_text(b"x", "p") is None


@pytest.mark.asyncio
async def test_vision_text_never_goes_to_the_cloud_by_default_even_when_the_local_model_fails():
    def down(_):
        raise httpx.ConnectError("refused")

    rec = Recorder(no_groq, down)
    assert await make(rec).vision_text(b"x", "p") is None  # returns None instead of raising or leaking the image
    assert set(rec.hosts()) == {"127.0.0.1"}
    rec500 = Recorder(no_groq, lambda r: httpx.Response(500, text="model not found"))
    assert await make(rec500).vision_text(b"x", "p") is None


@pytest.mark.asyncio
async def test_the_cloud_is_used_only_when_asked_for_and_never_on_prem():
    def down(_):
        raise httpx.ConnectError("refused")

    rec = Recorder(lambda r: groq_ok("A clock reading ten past ten."), down)
    out = await make(rec).vision_text(b"jpeg", "Read the clock.", local_only=False, max_tokens=40)
    assert out == "A clock reading ten past ten."
    groq_body = rec.bodies("api.groq.com")[0]
    assert groq_body["model"] == "qwen/qwen3.8-27b" and groq_body["max_tokens"] == 40
    assert groq_body["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    onprem = Recorder(no_groq, down)
    assert await make(onprem, onprem=True).vision_text(b"x", "p", local_only=False) is None
    assert set(onprem.hosts()) == {"127.0.0.1"}  # on-prem keeps every image on this machine


@pytest.mark.asyncio
async def test_vision_text_ignores_empty_input_without_any_request():
    def boom(_):
        raise AssertionError("no request may be made")

    gw = make(Recorder(boom, boom))
    assert await gw.vision_text(b"", "p") is None and await gw.vision_text(b"x", "   ") is None


def test_the_local_model_names_can_be_set_from_the_environment():
    cfg = GatewayConfig.from_env({"OLLAMA_VISION_MODEL": "qwen3-vl:4b", "OLLAMA_TEXT_MODEL": "qwen3.5:4b"})
    assert (cfg.local_vision_model, cfg.local_text_model) == ("qwen3-vl:4b", "qwen3.5:4b")
    default = GatewayConfig.from_env({})
    assert default.local_vision_model == "qwen3-vl:2b" and default.local_text_model == "qwen3.5:4b"


# ------------------------------------------- the configured vision model is not installed
class FakeOllama:
    """A tiny Ollama: installed models with their capabilities, 404 for anything else."""

    def __init__(self, installed: dict[str, list[str]], answer="a person"):
        self.installed, self.answer, self.paths, self.chat_models = installed, answer, [], []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in self.installed]})
        body = json.loads(request.content)
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"capabilities": self.installed.get(body["model"], [])})
        self.chat_models.append(body["model"])
        if body["model"] not in self.installed:
            return httpx.Response(404, json={"error": f"model '{body['model']}' not found"})
        return httpx.Response(200, json={"message": {"content": self.answer}})


def gateway_with(fake, onprem=False):
    return make(Recorder(no_groq, fake), onprem=onprem)


@pytest.mark.asyncio
async def test_a_missing_vision_model_is_replaced_once_by_an_installed_one_that_can_see():
    fake = FakeOllama({"qwen3.5:4b": ["completion", "vision"], "qwen2.5:0.5b": ["completion"]}, "A man in red.")
    gw = gateway_with(fake)
    assert await gw.vision_text(b"x", "Describe.") == "A man in red."  # default qwen3-vl:2b is not installed
    assert fake.chat_models == ["qwen3-vl:2b", "qwen3.5:4b"]  # tried the configured one, then the installed one
    before = len(fake.paths)
    assert await gw.vision_text(b"x", "Describe again.") == "A man in red."
    assert fake.chat_models[-1] == "qwen3.5:4b" and fake.paths[before:] == ["/api/chat"]  # no more probing


@pytest.mark.asyncio
async def test_the_usual_vision_families_are_preferred_when_several_are_installed():
    fake = FakeOllama({"llava:7b": ["vision"], "qwen3-vl:4b": ["vision"], "qwen3:4b": ["completion"]})
    gw = gateway_with(fake)
    assert await gw.vision_text(b"x", "p") == "a person"
    assert fake.chat_models[-1] == "qwen3-vl:4b"


@pytest.mark.asyncio
async def test_when_nothing_installed_can_see_the_answer_is_none_and_it_does_not_keep_probing():
    fake = FakeOllama({"qwen2.5:0.5b": ["completion"], "qwen3:4b": ["completion"]})
    gw = gateway_with(fake)
    assert await gw.vision_text(b"x", "p") is None
    probes = len([p for p in fake.paths if p != "/api/chat"])
    assert await gw.vision_text(b"x", "p") is None
    assert len([p for p in fake.paths if p != "/api/chat"]) == probes  # the search is done once, not on every call


@pytest.mark.asyncio
async def test_other_failures_do_not_trigger_a_model_search():
    rec500 = Recorder(no_groq, lambda r: httpx.Response(500, text="out of memory"))
    gw = make(rec500)
    assert await gw.vision_text(b"x", "p") is None
    assert [c.url.path for c in rec500.calls] == ["/api/chat"]  # a server error is not a missing model


@pytest.mark.asyncio
async def test_the_yes_no_vision_call_falls_back_the_same_way():
    fake = FakeOllama({"qwen3.5:4b": ["vision"]}, '{"answers": [true, false]}')
    gw = gateway_with(fake)
    assert await gw.vision_yesno(b"x", ["a?", "b?"]) == [True, False]
    assert fake.chat_models == ["qwen3-vl:2b", "qwen3.5:4b"]


@pytest.mark.asyncio
async def test_an_installed_configured_model_is_used_without_any_probing():
    fake = FakeOllama({"qwen3-vl:2b": ["vision"], "qwen3.5:4b": ["vision"]})
    gw = gateway_with(fake)
    assert await gw.vision_text(b"x", "p") == "a person"
    assert fake.paths == ["/api/chat"] and fake.chat_models == ["qwen3-vl:2b"]


@pytest.mark.asyncio
async def test_the_fallback_works_in_on_prem_mode_because_it_stays_on_this_machine():
    fake = FakeOllama({"qwen3.5:4b": ["vision"]})
    gw = gateway_with(fake, onprem=True)
    assert await gw.vision_text(b"x", "p", local_only=False) == "a person"


@pytest.mark.asyncio
async def test_a_reasoning_model_that_answers_in_the_thinking_field_still_gives_an_answer():
    reply = httpx.Response(200, json={"message": {"content": "", "thinking": '{"answers": ["true", "false"]}'}})
    rec = Recorder(no_groq, lambda r: reply)
    assert await make(rec, keys=()).vision_yesno(b"jpeg", ["a?", "b?"]) == [True, False]   # answers may be strings


@pytest.mark.asyncio
async def test_an_empty_reply_without_thinking_is_still_undecided():
    rec = Recorder(no_groq, lambda r: httpx.Response(200, json={"message": {"content": ""}}))
    assert await make(rec, keys=()).vision_yesno(b"jpeg", ["a?"]) == [None]
