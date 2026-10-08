"""SigLIP2 image and text embeddings. All outputs are float32 and L2-normalised.

`SigLIP2Embedder` implements the `Embedder` protocol the retrieval and baseline code expects:
    embed_text(text) -> (D,)        embed_images(images) -> (N, D)
Images may be BGR numpy arrays (what the pipeline holds) or PIL images.
Weights come from the Hugging Face cache (`HF_HOME`), so after `make models` this works offline.
"""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Sequence
from typing import Any

import numpy as np

from evora.core.config import REPO_ROOT
from evora.perception.detect import resolve_device
from evora.perception.settings import IngestSettings

log = logging.getLogger("evora.perception.embed")


def _normalize(x: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return (x / np.maximum(norm, 1e-12)).astype(np.float32)


def _to_pil(img: Any):
    from PIL import Image

    if isinstance(img, np.ndarray):
        return Image.fromarray(img[:, :, ::-1])  # BGR -> RGB
    return img


class SigLIP2Embedder:
    def __init__(self, cfg: IngestSettings):
        # keep every model under ./models so the demo runs offline after `make models`
        os.environ.setdefault("HF_HOME", str(REPO_ROOT / "models" / "hf"))
        import torch
        from transformers import AutoModel, AutoProcessor

        self.cfg = cfg
        self.device = resolve_device(cfg.device)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.processor = AutoProcessor.from_pretrained(cfg.image_model)
        self.model = AutoModel.from_pretrained(cfg.image_model, torch_dtype=dtype).to(self.device).eval()
        self._lock = threading.Lock()
        self._torch = torch
        self.dim = int(self.model.config.vision_config.hidden_size)
        log.info("siglip2 %s on %s, dim %d", cfg.image_model, self.device, self.dim)

    def _features(self, out) -> np.ndarray:
        feats = out if hasattr(out, "shape") else out.pooler_output
        return feats.float().cpu().numpy()

    def embed_images(self, images: Sequence[Any]) -> np.ndarray:
        if len(images) == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        torch = self._torch
        chunks: list[np.ndarray] = []
        for i in range(0, len(images), self.cfg.embed_batch):
            batch = [_to_pil(im) for im in images[i : i + self.cfg.embed_batch]]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            if self.device == "cuda":
                inputs["pixel_values"] = inputs["pixel_values"].half()
            with self._lock, torch.inference_mode():
                chunks.append(self._features(self.model.get_image_features(**inputs)))
        return _normalize(np.concatenate(chunks, axis=0))

    def embed_texts(self, texts: Sequence[str]) -> np.ndarray:
        torch = self._torch
        # SigLIP text towers were trained with fixed-length padding
        inputs = self.processor(
            text=list(texts), padding="max_length", max_length=64, truncation=True, return_tensors="pt",
        ).to(self.device)
        with self._lock, torch.inference_mode():
            feats = self._features(self.model.get_text_features(**inputs))
        return _normalize(feats)

    def embed_text(self, text: str) -> np.ndarray:
        return self.embed_texts([text])[0]
