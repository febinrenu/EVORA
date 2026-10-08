"""H0 check: which Groq models each key can see, and the rate-limit headers for each.

Usage:  uv run python scripts/check_groq.py [--keys k1,k2] [--models id1,id2]

Keys come from --keys or GROQ_KEYS (environment or .env). Keys are never printed in
full; only the last four characters are shown.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import httpx

BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODELS = [
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
    "qwen/qwen3.8-27b",
]
HEADER_NAMES = [
    "x-ratelimit-limit-requests",
    "x-ratelimit-remaining-requests",
    "x-ratelimit-limit-tokens",
    "x-ratelimit-remaining-tokens",
    "x-ratelimit-reset-tokens",
]


def mask(key: str) -> str:
    return f"...{key[-4:]}" if len(key) > 4 else "..."


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        values[name.strip()] = value.split("#", 1)[0].strip().strip('"').strip("'")
    return values


NUMBERED_KEY = re.compile(r"^GROQ_KEY[-_]?(\d+)$")


def resolve_keys(cli_value: str | None, env: dict[str, str], env_file: dict[str, str]) -> list[str]:
    """--keys wins; otherwise GROQ_KEYS (comma list) plus GROQ_KEY-1, GROQ_KEY_2 ... in numeric order."""
    if cli_value:
        return [k.strip() for k in cli_value.split(",") if k.strip()]
    for source in (env, env_file):  # the real environment wins over .env
        keys = [k.strip() for k in (source.get("GROQ_KEYS") or "").split(",") if k.strip()]
        numbered = sorted((int(m.group(1)), name) for name in source if (m := NUMBERED_KEY.match(name)))
        keys += [source[name].strip() for _, name in numbered if source[name].strip()]
        if keys:
            return list(dict.fromkeys(keys))
    return []


def list_models(client: httpx.Client, key: str) -> tuple[list[str] | None, str]:
    """Return (model ids, note). ids is None when the key was rejected or unreachable."""
    try:
        resp = client.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {key}"})
    except httpx.HTTPError as exc:
        return None, f"network error: {type(exc).__name__}"
    if resp.status_code != 200:
        return None, f"HTTP {resp.status_code}"
    try:
        return sorted(m["id"] for m in resp.json().get("data", [])), "ok"
    except (KeyError, ValueError, TypeError):
        return None, "unexpected reply"


def probe_model(client: httpx.Client, key: str, model: str) -> dict[str, str]:
    """One tiny completion; returns the rate-limit headers (and status) seen."""
    body = {"model": model, "messages": [{"role": "user", "content": "Reply with the word ok."}], "max_tokens": 16}
    if "gpt-oss" in model:
        body["reasoning_effort"] = "low"
        body["max_tokens"] = 64
    try:
        resp = client.post(f"{BASE_URL}/chat/completions", headers={"Authorization": f"Bearer {key}"}, json=body)
    except httpx.HTTPError as exc:
        return {"status": f"network error: {type(exc).__name__}"}
    out = {"status": str(resp.status_code)}
    for name in HEADER_NAMES:
        if name in resp.headers:
            out[name] = resp.headers[name]
    return out


def format_report(rows: list[dict]) -> str:
    lines: list[str] = []
    for row in rows:
        lines.append(f"key {row['key']}: {row['note']}")
        if row["models"] is None:
            continue
        lines.append(f"  models visible: {', '.join(row['models']) or '(none)'}")
        for model, info in row["probes"].items():
            seen = row["models"] is not None and model in row["models"]
            flag = "" if seen else "  [NOT LISTED for this key]"
            details = "  ".join(f"{k.replace('x-ratelimit-', '')}={v}" for k, v in info.items() if k != "status")
            lines.append(f"  {model}: HTTP {info['status']}  {details}{flag}")
    return "\n".join(lines)


def run(keys: list[str], models: list[str], client: httpx.Client) -> list[dict]:
    rows = []
    for key in keys:
        ids, note = list_models(client, key)
        probes = {m: probe_model(client, key, m) for m in models} if ids is not None else {}
        rows.append({"key": mask(key), "note": note, "models": ids, "probes": probes})
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--keys", help="comma separated keys (default: GROQ_KEYS)")
    parser.add_argument("--models", help="comma separated model ids to probe")
    args = parser.parse_args(argv)

    keys = resolve_keys(args.keys, dict(os.environ), load_env_file(Path(".env")))
    if not keys:
        print("No keys found. Set GROQ_KEYS in .env or pass --keys.", file=sys.stderr)
        return 2
    models = [m.strip() for m in args.models.split(",")] if args.models else DEFAULT_MODELS
    with httpx.Client(timeout=20.0) as client:
        rows = run(keys, models, client)
    print(format_report(rows))
    missing = [m for m in models if not any(r["models"] and m in r["models"] for r in rows)]
    if missing:
        print(f"\nMissing for every key: {', '.join(missing)}. "
              "Swap the id in config/default.yaml -> llm.models and log a Decision.")
    return 0 if all(r["models"] is not None for r in rows) and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
