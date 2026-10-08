"""Verification: a second, independent look at the top evidence (contribution C3).

Retrieval ranks by similarity; this asks a vision model a plain yes/no question about each
candidate. To keep one model call per batch, candidates go into a single numbered contact
sheet and the model answers one question per number. Results stream after the answer, so
a slow or missing model never delays or loses the answer.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Protocol

from contracts.models import Evidence, QueryPlan

from evora.llm.schemas import LLMError

log = logging.getLogger("evora.query.verify")

MAX_ITEMS = 9   # a 3 x 3 sheet
CELL_PX = 224   # each candidate is shown at least this large


class SheetUnavailable(RuntimeError):
    """The contact sheet cannot be built (no image library, or no usable image)."""


class VisionGateway(Protocol):
    async def vision_yesno(self, image_jpeg: bytes, questions: list[str]) -> list[bool | None]: ...


class ImageSource(Protocol):
    def crop_for(self, ev: Evidence) -> bytes | None:
        """JPEG of the evidence (its best crop, or the frame region around its box); None if unavailable."""


@dataclass(frozen=True)
class VerifyConfig:
    max_items: int = MAX_ITEMS
    cell_px: int = CELL_PX


def describe_target(plan: QueryPlan) -> str:
    """'a red car', from the planner's own visual description."""
    if not plan.targets:
        return "the object being searched for"
    text = plan.targets[0].embed_text.removeprefix("a photo of").strip()
    return text or plan.targets[0].noun


def build_contact_sheet(images: Sequence[bytes], cell_px: int = CELL_PX, cols: int = 3) -> bytes:
    """Tile the images into one numbered JPEG (1 at the top left, reading order)."""
    try:
        import cv2
        import numpy as np
    except ImportError as exc:  # pragma: no cover - exercised only where OpenCV is absent
        raise SheetUnavailable("OpenCV is required to build the verification sheet") from exc
    if not images:
        raise SheetUnavailable("no images to put on the sheet")
    rows = (len(images) + cols - 1) // cols
    sheet = np.full((rows * cell_px, cols * cell_px, 3), 255, dtype=np.uint8)
    for i, raw in enumerate(images):
        frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise SheetUnavailable(f"image {i + 1} could not be decoded")
        h, w = frame.shape[:2]
        scale = min((cell_px - 6) / w, (cell_px - 6) / h)
        resized = cv2.resize(frame, (max(1, int(w * scale)), max(1, int(h * scale))))
        r, c = divmod(i, cols)
        y0 = r * cell_px + (cell_px - resized.shape[0]) // 2
        x0 = c * cell_px + (cell_px - resized.shape[1]) // 2
        sheet[y0:y0 + resized.shape[0], x0:x0 + resized.shape[1]] = resized
        label = str(i + 1)
        cv2.rectangle(sheet, (c * cell_px, r * cell_px), (c * cell_px + 34, r * cell_px + 34), (0, 0, 0), -1)
        cv2.putText(sheet, label, (c * cell_px + 6, r * cell_px + 27), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (255, 255, 255), 2, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise SheetUnavailable("could not encode the sheet")
    return bytes(buf)


def questions_for(plan: QueryPlan, count: int) -> list[str]:
    what = describe_target(plan)
    return [f"Does image {n} clearly show {what}?" for n in range(1, count + 1)]


class Verifier:
    """Implements the router's Verifier protocol."""

    def __init__(self, gateway: VisionGateway, images: ImageSource, cfg: VerifyConfig | None = None,
                 sheet_builder=build_contact_sheet) -> None:  # noqa: ANN001 - callable seam for tests
        self._gateway, self._images, self.cfg = gateway, images, cfg or VerifyConfig()
        self._sheet = sheet_builder

    async def verify(self, plan: QueryPlan, evidence: Sequence[Evidence]) -> AsyncIterator[tuple[str, bool | None]]:
        batch = list(evidence)[: self.cfg.max_items]
        pictures: list[tuple[Evidence, bytes]] = []
        results: dict[str, bool | None] = {}
        for ev in batch:
            data = self._images.crop_for(ev)
            if data:
                pictures.append((ev, data))
            else:
                results[ev.id] = None  # nothing to look at

        if pictures:
            try:
                sheet = self._sheet([d for _, d in pictures], self.cfg.cell_px)
                answers = await self._gateway.vision_yesno(sheet, questions_for(plan, len(pictures)))
            except (SheetUnavailable, LLMError) as exc:
                log.info("verification skipped: %s", exc)
                answers = [None] * len(pictures)
            for (ev, _), ok in zip(pictures, answers, strict=False):
                results[ev.id] = ok

        for ev in batch:  # stream in ranking order
            yield ev.id, results.get(ev.id)
