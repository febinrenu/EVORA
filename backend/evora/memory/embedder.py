"""Text embedders for alias matching: bge-small via fastembed when cached, else a deterministic hashing fallback."""
from __future__ import annotations

import logging
import os
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from evora.core.config import REPO_ROOT

log = logging.getLogger("evora.memory.embedder")

DIM = 384
MODEL_ID = "BAAI/bge-small-en-v1.5"


class TextEmbedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray:
        """L2-normalised float32 matrix, one row per text, DIM columns."""
        ...


class EmbedderUnavailable(RuntimeError):
    pass


def _normalise(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return (mat / np.where(norms == 0, 1.0, norms)).astype(np.float32)


class HashingEmbedder:
    """Character 3-5-gram feature hashing. Deterministic, offline, catches typos and word forms."""

    name = "hash-ngram-384"

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), DIM), dtype=np.float32)
        for row, text in enumerate(texts):
            padded = f" {text.lower().strip()} "
            for n in (3, 4, 5):
                for i in range(len(padded) - n + 1):
                    h = zlib.crc32(padded[i : i + n].encode("utf-8"))
                    out[row, h % DIM] += 1.0 if (h >> 16) & 1 else -1.0
        return _normalise(out)


class FastEmbedder:
    """bge-small-en-v1.5 through fastembed, from the local cache only (`make models` fetches it)."""

    def __init__(self, cache_dir: Path | None = None):
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise EmbedderUnavailable("fastembed is not installed") from exc
        try:
            # `make models` stores the model in <repo>/models/fastembed, not in fastembed's default temp folder
            cache = cache_dir or Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models")) / "fastembed"
            self._model = TextEmbedding(MODEL_ID, cache_dir=str(cache), local_files_only=True)
        except Exception as exc:  # noqa: BLE001 - fastembed raises several unrelated types for a missing model
            raise EmbedderUnavailable(f"{MODEL_ID} is not cached: {exc}") from exc
        self.name = f"fastembed:{MODEL_ID}"

    def embed(self, texts: list[str]) -> np.ndarray:
        return _normalise(np.asarray(list(self._model.embed(texts)), dtype=np.float32))


def default_embedder(cache_dir: Path | None = None) -> TextEmbedder:
    try:
        return FastEmbedder(cache_dir)
    except EmbedderUnavailable as exc:
        log.warning("using the hashing embedder for aliases (%s)", exc)
        return HashingEmbedder()
