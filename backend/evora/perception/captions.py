"""Layer L3: one factual sentence per person or vehicle, written by the local vision model.

The sentence feeds caption search (BM25 and the bge-small vector in the `captions` table). It is the lowest-priority
layer: it only runs when a vision client is registered (`perception.vision`), is capped by track count and time, and a
model failure for one track never stops the others. Crops go to the local model only; nothing is sent to the cloud.
The prompts ask for clothing, colours and carried objects, never faces, identity, age or gender, and the validator
rewrites any gendered noun to "person".
"""
from __future__ import annotations

import logging
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

import numpy as np
from contracts.models import CameraInfo, IngestJob
from contracts.vectors import TEXT_DIM, TEXT_MODEL

from evora.core.config import REPO_ROOT
from evora.core.db import Database
from evora.core.vectors import dims_from_meta, ensure_tables
from evora.core.workspace import Workspace
from evora.perception.locks import STORE_SETUP
from evora.perception.settings import IngestSettings
from evora.perception.vision import VisionClient

log = logging.getLogger("evora.perception.captions")

ProgressFn = Callable[[IngestJob], None]
PERSON_CLASSES = ("person",)
VEHICLE_CLASSES = ("car", "truck", "bus", "motorcycle", "bicycle")
PERSON_PROMPT = (
    "Describe this person's clothing in one short factual sentence: the colour of the top, the colour of the trousers or skirt, "
    "and any bag or umbrella they carry. Do not describe the face and do not guess identity, age or gender. "
    "Example: a person in a red jacket and dark blue trousers carrying a black backpack."
)
VEHICLE_PROMPT = (
    "Describe this vehicle in one short factual sentence: its colour and type "
    "(car, SUV, van, truck, bus, motorcycle or bicycle). "
    "Do not read any licence plate. Example: a white SUV."
)
MAX_CHARS = 160
MIN_WORDS = 3
_THINK = re.compile(r"<think>.*?</think>", re.S | re.I)
# "carrying no bag" would match a search for bags, so statements about what is absent are dropped
_NEGATION = re.compile(r"[,;]?\s*\b(?:carrying|holding|with|having)\s+(?:no|nothing|neither)\b[^.,;]*", re.I)
_GENDERED = re.compile(r"\b(men|man|women|woman|boys?|girls?|males?|females?|ladies|lady|guys?|gentlem[ae]n)\b", re.I)


class CaptionsUnavailable(RuntimeError):
    """No vision client, or the text embedding model is missing."""


class TextEmbedder(Protocol):
    def embed(self, texts: list[str]) -> np.ndarray:
        """(N, 384) float32, L2-normalised."""


class FastTextEmbedder:
    """bge-small-en-v1.5 through fastembed from `models/fastembed` (the same model the memory service uses)."""

    def __init__(self) -> None:
        cache = Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models")) / "fastembed"
        try:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(TEXT_MODEL, cache_dir=str(cache), local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - fastembed raises several unrelated types for a missing model
            raise CaptionsUnavailable(
                f"text embedding model missing: run scripts/models_download.py --only fastembed ({exc})"
            ) from exc

    def embed(self, texts: list[str]) -> np.ndarray:
        vecs = np.asarray(list(self._model.embed(texts)), dtype=np.float32)
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-12)


def clean_caption(raw: str | None) -> str | None:
    """A usable one-line caption from a model reply, or None. Gendered nouns become 'person'."""
    if not raw:
        return None
    text = _THINK.sub(" ", raw)
    text = re.sub(r"^[\s\"'`*\-]+|[\s\"'`*]+$", "", " ".join(text.split()))
    text = _NEGATION.sub("", text)
    text = _GENDERED.sub("person", text)
    text = re.sub(r"\bperson(\s+person)+\b", "person", text, flags=re.I)
    if len(text.split()) < MIN_WORDS or re.fullmatch(r"(none|n/a|unknown|cannot.*|i can.?t.*|sorry.*)", text, re.I):
        return None
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS].rsplit(" ", 1)[0].rstrip(",;:") + "."
    return text


def run_l3(cam: CameraInfo, ws: Workspace, db: Database, store, st: IngestSettings, vision: VisionClient | None,
           embedder: TextEmbedder | None, on_progress: ProgressFn) -> int | None:
    """Caption the camera's best tracks. Returns how many captions were written, or None when the layer could not run."""
    if vision is None:
        log.warning("%s: L3 skipped, no local vision model is registered", cam.id)
        return None
    embedder = embedder or FastTextEmbedder()
    started = time.monotonic()

    def report(frac: float) -> None:
        on_progress(IngestJob(id="", camera_id=cam.id, state="running", layer="L3", progress=frac,  # type: ignore[arg-type]
                              video_s_per_s=None))

    report(0.0)
    classes = PERSON_CLASSES + VEHICLE_CLASSES
    marks = ",".join("?" * len(classes))
    with db.read() as c:
        tracks = c.execute(
            f"SELECT id, cls, best_crop, best_t, quality FROM tracks WHERE camera_id=? AND cls IN ({marks}) "  # noqa: S608 - placeholders only
            "AND best_crop IS NOT NULL ORDER BY quality DESC LIMIT ?", (cam.id, *classes, st.l3_max_tracks)).fetchall()
    captions: list[tuple[str, str, float]] = []        # (track_id, text, time)
    failures = 0
    for i, t in enumerate(tracks):
        if time.monotonic() - started > st.l3_budget_s:
            log.info("%s: L3 time budget reached after %d of %d tracks", cam.id, i, len(tracks))
            break
        path = ws.media_dir / t["best_crop"]
        if not path.is_file():
            continue
        prompt = PERSON_PROMPT if t["cls"] in PERSON_CLASSES else VEHICLE_PROMPT
        try:
            reply = vision.describe(path.read_bytes(), prompt, max_tokens=st.l3_max_tokens)
        except Exception:  # noqa: BLE001 - one bad reply must not stop the layer
            log.exception("%s: caption call failed for %s", cam.id, t["id"])
            reply = None
        text = clean_caption(reply)
        if text is None:
            failures += 1
        else:
            captions.append((t["id"], text, t["best_t"] or 0.0))
        if i % 5 == 0:
            report(0.9 * i / max(len(tracks), 1))
    if tracks and not captions:
        log.warning("%s: L3 produced no captions (%d failed); the vision model may be down", cam.id, failures)
        return None
    with STORE_SETUP:
        ensure_tables(store, dims_from_meta(db), only={"captions"})
    table = store.open_table("captions")
    table.delete(f"camera_id = '{cam.id}'")
    if captions:
        vecs = embedder.embed([c[1] for c in captions])
        if vecs.shape[1] != TEXT_DIM:
            raise CaptionsUnavailable(f"text embedder returned {vecs.shape[1]} dimensions, the contract says {TEXT_DIM}")
        table.add([{"vector": v.tolist(), "text": text, "camera_id": cam.id, "t": t, "track_id": tid}
                   for v, (tid, text, t) in zip(vecs, captions, strict=True)])
    report(1.0)
    log.info("%s L3: %d captions (%d failed) in %.1fs", cam.id, len(captions), failures, time.monotonic() - started)
    return len(captions)
