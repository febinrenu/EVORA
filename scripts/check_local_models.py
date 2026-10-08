"""Is this machine ready to look at footage? Checks the local models the way the product uses them.

    python scripts/check_local_models.py            # from the repo root, with the backend environment

It checks, and says what to run when something is wrong:
  1. Ollama answers and has the vision model the product wants (qwen3-vl:4b-instruct), or an installed fallback;
  2. a real call about a generated picture gets the right answer (count of red squares, a blue shape) and how long it took;
  3. the model runs on the GPU (an Ollama model that spilled to the processor is several times slower);
  4. PyTorch can use the GPU when the machine has one (the detector, embeddings and the object search depend on it);
  5. the open-vocabulary detector can start (weights, text encoder and the clip package are present).
Exit code 0 when nothing failed (warnings are allowed), 1 otherwise. No key or image leaves the machine.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

WANTED = "qwen3-vl:4b-instruct"
FALLBACKS = ("qwen3.5:4b", "qwen3-vl")
SLOW_CALL_S = 8.0
MIN_GPU_SHARE = 0.9


@dataclass
class Check:
    name: str
    status: str            # ok | warn | fail
    detail: str
    fix: str = ""

    def line(self) -> str:
        mark = {"ok": "ok   ", "warn": "WARN ", "fail": "FAIL "}[self.status]
        out = f"{mark} {self.name:34s} {self.detail}"
        return out + (f"\n      -> {self.fix}" if self.fix and self.status != "ok" else "")


def pick_model(installed: list[str]) -> str | None:
    """The model the product will use for looking: the wanted one, else a fallback that is installed."""
    if WANTED in installed:
        return WANTED
    for prefix in FALLBACKS:
        found = next((n for n in installed if n == prefix or n.startswith(prefix)), None)
        if found:
            return found
    return None


def gpu_share(entry: dict[str, Any]) -> float:
    """The share of a loaded Ollama model that sits in GPU memory (1.0 = all of it)."""
    size = float(entry.get("size") or 0)
    return float(entry.get("size_vram") or 0) / size if size else 0.0


def parse_number(text: str) -> int | None:
    match = re.search(r"\b(\d{1,3})\b", text or "")
    return int(match.group(1)) if match else None


def test_picture() -> bytes:
    """Three red squares and two blue ones on a grey background."""
    import cv2
    import numpy as np

    image = np.full((360, 640, 3), 120, np.uint8)
    for i in range(3):
        cv2.rectangle(image, (40 + i * 150, 60), (120 + i * 150, 140), (0, 0, 220), -1)      # red (BGR)
    for i in range(2):
        cv2.rectangle(image, (100 + i * 220, 220), (190 + i * 220, 300), (220, 60, 0), -1)   # blue
    return bytes(cv2.imencode(".jpg", image)[1])


def ask(client: httpx.Client, host: str, model: str, prompt: str, jpeg: bytes) -> tuple[str, float]:
    payload = {"model": model, "stream": False, "think": False, "options": {"temperature": 0, "num_predict": 600},
               "messages": [{"role": "user", "content": prompt, "images": [base64.b64encode(jpeg).decode()]}]}
    started = time.perf_counter()
    reply = client.post(f"{host}/api/chat", json=payload, timeout=300).json()
    message = reply.get("message", {})
    return ((message.get("content") or message.get("thinking") or "").strip(), time.perf_counter() - started)


def check_ollama(host: str, client: httpx.Client) -> tuple[list[Check], str | None]:
    try:
        installed = [m["name"] for m in client.get(f"{host}/api/tags", timeout=10).json().get("models", [])]
    except (httpx.HTTPError, ValueError, KeyError):
        return [Check("Ollama", "fail", f"no answer at {host}", "start it with: ollama serve")], None
    model = pick_model(installed)
    if model is None:
        return [Check("Vision model", "fail", f"none installed ({', '.join(installed) or 'no models'})",
                      f"ollama pull {WANTED}")], None
    if model != WANTED:
        return [Check("Vision model", "warn", f"{model} (the product prefers {WANTED})", f"ollama pull {WANTED}")], model
    return [Check("Vision model", "ok", model)], model


def check_picture(host: str, model: str, client: httpx.Client) -> list[Check]:
    jpeg = test_picture()
    out: list[Check] = []
    answer, seconds = ask(client, host, model, "How many red squares are visible in this image? Reply with only a number.", jpeg)
    out.append(Check("Counts what it sees", "ok" if parse_number(answer) == 3 else "fail",
                     f"asked for red squares, said {answer[:40]!r} (3 is right), {seconds:.1f}s",
                     "the model is not reading the picture correctly; try ollama pull " + WANTED))
    answer, seconds2 = ask(client, host, model, "Is there a blue shape in this image? Answer yes or no.", jpeg)
    out.append(Check("Reads colours", "ok" if answer.lower().startswith("yes") else "fail",
                     f"asked about a blue shape, said {answer[:40]!r}, {seconds2:.1f}s"))
    slowest = max(seconds, seconds2)
    out.append(Check("Answers quickly", "ok" if slowest <= SLOW_CALL_S else "warn", f"slowest call {slowest:.1f}s",
                     "the first call after loading a model is slow; if every call is, it is running on the processor"))
    try:
        loaded = [m for m in client.get(f"{host}/api/ps", timeout=10).json().get("models", []) if m.get("name") == model]
    except (httpx.HTTPError, ValueError):
        loaded = []
    if loaded:
        share = gpu_share(loaded[0])
        out.append(Check("Model runs on the GPU", "ok" if share >= MIN_GPU_SHARE else "warn", f"{share:.0%} of it in GPU memory",
                         "free GPU memory (close other GPU programs) or use a smaller model; set OLLAMA_MAX_LOADED_MODELS=1"))
    return out


def has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None and subprocess.run(
        ["nvidia-smi", "-L"], capture_output=True, text=True, check=False).returncode == 0


def check_torch() -> Check:
    try:
        import torch
    except ImportError:
        return Check("PyTorch", "fail", "not installed", "make setup-perception")
    if torch.cuda.is_available():
        return Check("PyTorch uses the GPU", "ok", f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})")
    if has_nvidia():
        return Check("PyTorch uses the GPU", "warn", f"an NVIDIA GPU is present but torch {torch.__version__} cannot use it",
                     "install the CUDA build: uv pip install --python backend/.venv/Scripts/python.exe --reinstall torch "
                     "torchvision --index-url https://download.pytorch.org/whl/cu128   (stop the app first)")
    return Check("PyTorch uses the GPU", "warn", f"no GPU found, running on the processor (torch {torch.__version__})",
                 "expected on a machine without an NVIDIA GPU; indexing and the object search are slower")


def check_open_vocabulary() -> Check:
    try:
        from evora.perception.openvocab import OpenVocabDetector, OpenVocabUnavailable
    except ImportError as exc:
        return Check("Object search (YOLOE)", "fail", f"the backend is not importable ({exc})",
                     "run from the repo root with PYTHONPATH=backend, or use the backend environment")
    try:
        OpenVocabDetector()
    except OpenVocabUnavailable as exc:
        return Check("Object search (YOLOE)", "fail", str(exc)[:110],
                     "python scripts/models_download.py --only yoloe; make setup-perception")
    return Check("Object search (YOLOE)", "ok", "weights, text encoder and clip package present")


def run(host: str) -> list[Check]:
    checks: list[Check] = []
    with httpx.Client() as client:
        found, model = check_ollama(host, client)
        checks += found
        if model:
            checks += check_picture(host, model, client)
    checks.append(check_torch())
    checks.append(check_open_vocabulary())
    return checks


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--host", default=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    sys.path[:0] = [str(Path(__file__).resolve().parents[1] / "backend")]
    checks = run(args.host)
    if args.json:
        print(json.dumps([c.__dict__ for c in checks], indent=2))
    else:
        for c in checks:
            print(c.line())
        failed = sum(c.status == "fail" for c in checks)
        warned = sum(c.status == "warn" for c in checks)
        print(f"\n{failed} failed, {warned} warnings" if failed or warned else "\nready")
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
