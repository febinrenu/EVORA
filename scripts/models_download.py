"""Download every local model artifact into ./models (used by `make models`).

Usage: python scripts/models_download.py [--only yolo siglip2 bge fastembed boxmot yunet ollama] [--gpu]

Heavy imports happen inside each step, so one missing package only fails its own step.
Ollama pulls are skipped with a warning when Ollama is not installed.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

log = logging.getLogger("models")
ROOT = Path(os.environ.get("EVORA_MODELS_DIR", "models")).resolve()
SIGLIP2 = "google/siglip2-base-patch16-224"
BGE = "BAAI/bge-small-en-v1.5"
YUNET_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx")
# Exact YOLO26 / YOLOE-26 checkpoint names must be confirmed against the Ultralytics docs (spike P2.3).
YOLO_WEIGHTS = ["yolo26n.pt", "yolo26s.pt", "yoloe-26n-seg.pt", "yoloe-26n-seg-pf.pt"]
BOXMOT_WEIGHTS = {"osnet_x0_25_msmt17.pt": "1sSwXSUlj4_tHZequ_iZ8w_Jh0VaRQMqF"}
ALL_STEPS = ["yolo", "siglip2", "bge", "fastembed", "boxmot", "yunet", "ollama"]


def step_yolo() -> None:
    from ultralytics.utils.downloads import attempt_download_asset

    dest = ROOT / "ultralytics"
    dest.mkdir(parents=True, exist_ok=True)
    missing = []
    for name in YOLO_WEIGHTS:
        got = Path(attempt_download_asset(dest / name))
        if got.exists() and got.stat().st_size > 0:
            log.info("yolo %s -> %s", name, got)
        else:
            missing.append(name)
            log.error("yolo %s not available", name)
    if missing:
        raise RuntimeError(f"missing weights: {', '.join(missing)}")


def step_hf(repo: str) -> None:
    from huggingface_hub import snapshot_download

    path = snapshot_download(repo_id=repo)
    log.info("hf %s -> %s", repo, path)


def step_fastembed() -> None:
    """bge-small in fastembed's ONNX format under models/fastembed, which the memory service loads local-only."""
    from fastembed import TextEmbedding

    cache = ROOT / "fastembed"
    cache.mkdir(parents=True, exist_ok=True)
    model = TextEmbedding(BGE, cache_dir=str(cache))
    dim = len(next(iter(model.embed(["warm-up"]))))
    log.info("fastembed %s -> %s (dim %d)", BGE, cache, dim)


def step_boxmot() -> None:
    """OSNet person ReID weights (BoxMOT model zoo, hosted on Google Drive)."""
    import gdown

    dest = ROOT / "boxmot"
    dest.mkdir(parents=True, exist_ok=True)
    for name, file_id in BOXMOT_WEIGHTS.items():
        target = dest / name
        if target.is_file() and target.stat().st_size > 0:
            log.info("boxmot %s present", name)
            continue
        if not gdown.download(id=file_id, output=str(target), quiet=True):
            raise RuntimeError(f"could not download {name} from Google Drive (quota?); try again later")
        log.info("boxmot %s -> %s", name, target)


def step_yunet() -> None:
    dest = ROOT / "yunet" / "face_detection_yunet_2023mar.onnx"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        log.info("yunet present")
        return
    urllib.request.urlretrieve(YUNET_URL, dest)
    log.info("yunet -> %s", dest)


def step_ollama(gpu: bool) -> None:
    if not shutil.which("ollama"):
        log.warning("ollama not installed; skipping")
        return
    failed = []
    for tag in ["qwen3-vl:4b" if gpu else "qwen3-vl:2b", "qwen3.5:4b"]:
        r = subprocess.run(["ollama", "pull", tag], capture_output=True, text=True)
        if r.returncode:
            log.error("ollama pull %s failed: %s", tag, (r.stderr or r.stdout).strip()[-300:])
            failed.append(tag)
        else:
            log.info("ollama %s ready", tag)
    if failed:
        raise RuntimeError(f"ollama pulls failed: {', '.join(failed)}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*", choices=ALL_STEPS)
    ap.add_argument("--gpu", action="store_true", help="pull the 4B VLM instead of 2B")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ROOT.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(ROOT / "hf"))

    steps = {
        "yolo": step_yolo,
        "siglip2": lambda: step_hf(SIGLIP2),
        "bge": lambda: step_hf(BGE),
        "fastembed": step_fastembed,
        "boxmot": step_boxmot,
        "yunet": step_yunet,
        "ollama": lambda: step_ollama(args.gpu),
    }
    failed = []
    for name in args.only or ALL_STEPS:
        try:
            steps[name]()
        except Exception as exc:  # noqa: BLE001 - keep going so one failure does not block the rest
            log.error("step %s failed: %s", name, exc)
            failed.append(name)
    if failed:
        log.error("failed steps: %s", ", ".join(failed))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
