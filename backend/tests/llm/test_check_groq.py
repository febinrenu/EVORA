import importlib.util
from pathlib import Path

import httpx

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_groq.py"
spec = importlib.util.spec_from_file_location("check_groq", SCRIPT)
check_groq = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check_groq)


def handler(request: httpx.Request) -> httpx.Response:
    key = request.headers["authorization"].removeprefix("Bearer ")
    if key.endswith("bad1"):
        return httpx.Response(401)
    if request.url.path.endswith("/models"):
        return httpx.Response(200, json={"data": [{"id": "openai/gpt-oss-20b"}, {"id": "whisper-large-v3"}]})
    return httpx.Response(
        200, json={}, headers={"x-ratelimit-remaining-tokens": "7900", "x-ratelimit-reset-tokens": "1.2s"}
    )


def client():
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_mask_hides_key():
    assert check_groq.mask("gsk_secretvalue9876") == "...9876"


def test_resolve_keys_prefers_cli_then_env_then_file():
    assert check_groq.resolve_keys("a, b", {"GROQ_KEYS": "x"}, {"GROQ_KEYS": "y"}) == ["a", "b"]
    assert check_groq.resolve_keys(None, {"GROQ_KEYS": "x"}, {"GROQ_KEYS": "y"}) == ["x"]
    assert check_groq.resolve_keys(None, {}, {"GROQ_KEYS": "y,,z"}) == ["y", "z"]
    assert check_groq.resolve_keys(None, {}, {}) == []


def test_env_file_parsing(tmp_path):
    f = tmp_path / ".env"
    f.write_text('# c\nGROQ_KEYS="k1,k2"  # mine\nOTHER=1\n')
    assert check_groq.load_env_file(f)["GROQ_KEYS"] == "k1,k2"
    assert check_groq.load_env_file(tmp_path / "missing") == {}


def test_report_flags_unlisted_models_and_bad_keys_without_leaking():
    rows = check_groq.run(["gsk_good_1111", "gsk_bad_bad1"], ["openai/gpt-oss-20b", "qwen/qwen3.8-27b"], client())
    text = check_groq.format_report(rows)
    assert rows[1]["models"] is None and "HTTP 401" in text
    assert "qwen/qwen3.8-27b: HTTP 200" in text and "[NOT LISTED for this key]" in text
    assert "remaining-tokens=7900" in text
    assert "gsk_" not in text
