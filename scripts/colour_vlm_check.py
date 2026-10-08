# ruff: noqa: E501
"""Second opinion on colours from the local vision model, on the same crops as the human labelling page.

Usage: python scripts/colour_vlm_check.py [--page data/share/colour_label/index.html] [--out data/share/colour_label/vlm_labels.json]
                                          [--model qwen3-vl:4b] [--limit N]

Writes a file in the same format as the human labels, so `python scripts/colour_eval.py <file> --predictor stored`
scores the system against the model. This measures AGREEMENT, not accuracy: the model makes its own mistakes, notably
between black and dark blue. It is most useful to find crops where the two disagree, which the human should check first.
This is a development script that talks to the local Ollama on this machine only; product code goes through the gateway.
Crops never leave the machine.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

COLOURS = ["black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "pink", "brown"]
SYNONYMS = {"gray": "grey", "navy": "blue", "dark blue": "blue", "light blue": "blue", "teal": "green", "olive": "green", "maroon": "red",
            "burgundy": "red", "beige": "brown", "tan": "brown", "khaki": "brown", "cream": "white", "silver": "grey", "charcoal": "grey",
            "dark gray": "grey", "dark grey": "grey", "light gray": "grey", "light grey": "grey", "violet": "purple", "magenta": "pink",
            "gold": "yellow"}
PERSON_PROMPT = (
    "Look at this person. Name the colour of the top (shirt, jacket or coat) and the colour of the trousers or skirt. "
    "Choose each colour from exactly this list: black, white, grey, red, orange, yellow, green, blue, purple, pink, brown. "
    'Reply as JSON: {"upper": "<colour>", "lower": "<colour>"}. If a part is not visible use "unknown".'
)
VEHICLE_PROMPT = (
    "Look at this vehicle. Name the colour of its body. Choose from exactly this list: black, white, grey, red, orange, yellow, green, "
    'blue, purple, pink, brown. Reply as JSON: {"color": "<colour>"}. If you cannot tell use "unknown".'
)


def normalise(word: str | None) -> str | None:
    if not word:
        return None
    w = word.strip().lower().rstrip(".")
    w = SYNONYMS.get(w, w)
    return w if w in COLOURS else None


def ask(host: str, model: str, jpeg_b64: str, prompt: str, timeout: float = 300.0) -> dict:
    body = {"model": model, "stream": False, "think": False, "format": "json", "options": {"temperature": 0, "num_predict": 600},
            "messages": [{"role": "user", "content": prompt, "images": [jpeg_b64]}]}
    req = urllib.request.Request(f"{host}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    content = json.loads(urllib.request.urlopen(req, timeout=timeout).read())["message"]["content"]
    match = re.search(r"\{.*\}", content, re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--page", type=Path, default=Path("data/share/colour_label/index.html"))
    ap.add_argument("--out", type=Path, default=Path("data/share/colour_label/vlm_labels.json"))
    ap.add_argument("--model", default="qwen3-vl:4b")
    ap.add_argument("--host", default="http://127.0.0.1:11434")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    html = args.page.read_text(encoding="utf-8")
    match = re.search(r"const ITEMS=(\[.*?\]), COLOURS=", html, re.S)
    if not match:
        print(f"no items found in {args.page}", file=sys.stderr)
        return 1
    items = json.loads(match.group(1))
    ws_match = re.search(r'workspace:"([^"]+)"', html)
    workspace = ws_match.group(1) if ws_match else "meva-school"
    if args.limit:
        items = items[: args.limit]
    started, out_items = time.time(), []
    for i, it in enumerate(items, 1):
        reply = ask(args.host, args.model, it.pop("jpeg"), PERSON_PROMPT if it["kind"] == "person" else VEHICLE_PROMPT)
        slots = ("upper", "lower") if it["kind"] == "person" else ("color",)
        for slot in slots:
            term = normalise(reply.get(slot))
            it[f"label_{slot}"] = term or "unsure"
        out_items.append(it)
        if i % 10 == 0 or i == len(items):
            print(f"{i}/{len(items)} done, {time.time() - started:.0f}s", flush=True)
            args.out.write_text(json.dumps({"tool_version": 1, "workspace": workspace, "source": args.model, "items": out_items}, indent=1), encoding="utf-8")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
