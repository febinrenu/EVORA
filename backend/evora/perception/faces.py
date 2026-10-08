"""Face blurring with OpenCV YuNet. Detection only: nothing here recognises or identifies anyone.

    blur_faces(jpeg_or_frame) -> bytes      JPEG with every detected face Gaussian-blurred

Accepts JPEG/PNG bytes or a BGR numpy frame. The YuNet model comes from `models/yunet/` (fetched by
`scripts/models_download.py`). When the model file is missing a `FaceBlurUnavailable` is raised, so a
caller can never mistake an unblurred frame for a blurred one.
"""
from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import cv2
import numpy as np

from evora.core.config import REPO_ROOT

log = logging.getLogger("evora.perception.faces")

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
SCORE_THRESHOLD = 0.6
NMS_THRESHOLD = 0.3
TOP_K = 200
PAD = 0.35          # grow each face box by this fraction before blurring
JPEG_QUALITY = 90


class FaceBlurUnavailable(RuntimeError):
    """The YuNet model is not present, so frames cannot be blurred."""


_local = threading.local()   # an OpenCV detector keeps its input size, so each thread owns one and clips blur in parallel


def model_path() -> Path:
    root = Path(os.environ.get("EVORA_MODELS_DIR", REPO_ROOT / "models"))
    return root / "yunet" / MODEL_NAME


def _get_detector(width: int, height: int) -> cv2.FaceDetectorYN:
    detector = getattr(_local, "detector", None)
    if detector is None:
        path = model_path()
        if not path.is_file():
            raise FaceBlurUnavailable(f"face model missing: {path} (run scripts/models_download.py --only yunet)")
        detector = cv2.FaceDetectorYN.create(str(path), "", (width, height), SCORE_THRESHOLD, NMS_THRESHOLD, TOP_K)
        _local.detector = detector
    detector.setInputSize((width, height))
    return detector


def detect_faces(frame: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Face boxes as pixel (x, y, w, h) in `frame`."""
    h, w = frame.shape[:2]
    _, faces = _get_detector(w, h).detect(frame)
    if faces is None:
        return []
    return [(int(f[0]), int(f[1]), int(f[2]), int(f[3])) for f in faces]


def blur_frame(frame: np.ndarray) -> np.ndarray:
    """Return a copy of a BGR frame with every detected face blurred."""
    out = frame.copy()
    h, w = out.shape[:2]
    for x, y, bw, bh in detect_faces(frame):
        px, py = int(bw * PAD), int(bh * PAD)
        x1, y1, x2, y2 = max(x - px, 0), max(y - py, 0), min(x + bw + px, w), min(y + bh + py, h)
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        k = max(15, (max(x2 - x1, y2 - y1) // 2) | 1)
        out[y1:y2, x1:x2] = cv2.GaussianBlur(out[y1:y2, x1:x2], (k, k), 0)
    return out


def blur_faces(jpeg_or_frame: bytes | np.ndarray) -> bytes:
    """Blur faces in an encoded image or a BGR frame and return JPEG bytes."""
    if isinstance(jpeg_or_frame, np.ndarray):
        frame = jpeg_or_frame
    else:
        frame = cv2.imdecode(np.frombuffer(jpeg_or_frame, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("not a decodable image")
    ok, buf = cv2.imencode(".jpg", blur_frame(frame), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise OSError("could not encode the blurred frame")
    return buf.tobytes()
