"""Open-vocabulary detection with YOLOE-26: "find red cars" without training a class for it.

    detector = OpenVocabDetector()
    boxes = detector.detect(frame_bgr, ["red car", "person with a backpack"])   # -> list[OpenVocabBox]

Needs three things on the machine (all fetched by `scripts/models_download.py --only yoloe`):
the checkpoint `models/ultralytics/yoloe-26n-seg.pt`, the text encoder `models/ultralytics/mobileclip2_b.ts`
(242 MB) and the Ultralytics `clip` package (`pip install git+https://github.com/ultralytics/CLIP.git`).
Without them `OpenVocabUnavailable` is raised. Boxes are normalised 0..1.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evora.core.config import REPO_ROOT
from evora.perception.detect import resolve_device

log = logging.getLogger("evora.perception.openvocab")
TEXT_ENCODER = "mobileclip2_b.ts"


class OpenVocabUnavailable(RuntimeError):
    """The YOLOE checkpoint, its text encoder or the clip package is missing."""


@dataclass(frozen=True)
class OpenVocabBox:
    label: str
    conf: float
    xyxy: tuple[float, float, float, float]   # normalised 0..1


def _models_dir() -> Path:
    return Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models")) / "ultralytics"


def _stage_text_encoder(models: Path) -> None:
    """Ultralytics looks for the text encoder in its own weights directory: put our copy there once."""
    from ultralytics.utils import SETTINGS

    target = Path(SETTINGS["weights_dir"]) / TEXT_ENCODER
    source = models / TEXT_ENCODER
    if not target.exists() and source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


class OpenVocabDetector:
    def __init__(self, weights: str = "yoloe-26n-seg.pt", device: str = "auto"):
        models = _models_dir()
        path = models / weights
        if not path.is_file():
            raise OpenVocabUnavailable(f"missing {path} (run scripts/models_download.py --only yoloe)")
        if not (models / TEXT_ENCODER).is_file():
            raise OpenVocabUnavailable(f"missing {models / TEXT_ENCODER} (run scripts/models_download.py --only yoloe)")
        try:
            import clip  # noqa: F401  (the Ultralytics fork is imported by get_text_pe)
            from ultralytics import YOLOE
        except ImportError as exc:
            raise OpenVocabUnavailable(
                "install the Ultralytics clip package: pip install git+https://github.com/ultralytics/CLIP.git"
            ) from exc
        _stage_text_encoder(models)
        self.device = resolve_device(device)
        self._model = YOLOE(str(path))
        self._classes: tuple[str, ...] = ()
        self._lock = threading.Lock()

    def detect(self, image_bgr: np.ndarray, texts: Sequence[str], conf: float = 0.15) -> list[OpenVocabBox]:
        """Boxes for any of `texts` in one BGR image, best first. Changing the text list re-encodes the prompts."""
        names = tuple(t.strip() for t in texts if t.strip())
        if not names:
            return []
        h, w = image_bgr.shape[:2]
        with self._lock:
            if names != self._classes:
                self._model.set_classes(list(names), self._model.get_text_pe(list(names)))
                self._classes = names
            result = self._model.predict(image_bgr, conf=conf, device=self.device, verbose=False)[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        xyxy, confs, clss = result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy()
        boxes = [
            OpenVocabBox(names[int(c)], float(p), (float(b[0] / w), float(b[1] / h), float(b[2] / w), float(b[3] / h)))
            for b, p, c in zip(xyxy, confs, clss, strict=True)
        ]
        return sorted(boxes, key=lambda b: -b.conf)
