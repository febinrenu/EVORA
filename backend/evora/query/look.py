"""Questions only the picture can answer: ask the local vision model to look at the frames.

The tracker and the detector know where people and objects are; they do not know that someone is sitting, on the
phone, or what colour a carpet is, and a "what is in the room" built from track events says "multiple people
appeared". For those questions a few frames of the camera, laid out in time order on one labelled sheet, are shown
to the local vision model together with the question. The frames are returned as the evidence, so every answer can
be checked against what the footage shows, and the answer says it comes from a model looking at them.

Only the local model is used (the images show people).
"""
from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, tzinfo
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from contracts.models import QueryPlan

log = logging.getLogger("evora.query.look")

# what a person is doing or how they are positioned: not something the tracker or the detector measures
VISUAL_ACTIONS = re.compile(
    r"\b(sit|sits|sitting|seated|stand|stands|standing|talk|talks|talking|speak|speaking|phone|calling|texting|eat|eating|"
    r"drink|drinking|wave|waving|hold|holding|read|reading|typing|writing|dance|dancing|point|pointing|hug|hugging|"
    r"smile|smiling|sleep|sleeping|kneel|kneeling|bend|bending|lean|leaning|jump|jumping|argue|arguing|fight|fighting|"
    r"doing|happening|going on|laugh|laughing|clap|clapping)\b",
    re.IGNORECASE,
)
FRAMES_PER_CAMERA = 4
CELL = (480, 270)            # one frame on the sheet
ANSWER_TOKENS = 160
NOT_VISIBLE = "I can't tell from these frames"


class Gateway(Protocol):
    async def vision_text(self, image_jpeg: bytes, prompt: str, *, local_only: bool = True, max_tokens: int = 64,
                          model: str | None = None) -> str | None: ...


def wants_look(plan: QueryPlan, text: str) -> bool:
    """Does this question need the picture rather than the tracks?"""
    if plan.intent == "describe":
        return True
    if plan.intent in ("count", "path", "standing"):
        return plan.intent == "count" and bool(VISUAL_ACTIONS.search(text)) and not plan.targets
    if not plan.targets:
        return True                                    # a scene or activity question with no object type
    return bool(VISUAL_ACTIONS.search(text))           # "is anyone sitting", "who is talking on the phone"


@dataclass(frozen=True)
class Frame:
    camera_id: str
    t: float
    path: str


def contact_sheet(images: Sequence[np.ndarray], labels: Sequence[str]) -> bytes | None:
    """The frames in a grid, each marked with its letter and clock time, as one JPEG."""
    import cv2

    if not images:
        return None
    cols = 2 if len(images) > 1 else 1
    rows = (len(images) + cols - 1) // cols
    sheet = np.zeros((rows * CELL[1], cols * CELL[0], 3), np.uint8)
    for i, (image, label) in enumerate(zip(images, labels, strict=True)):
        tile = cv2.resize(image, CELL, interpolation=cv2.INTER_AREA)
        cv2.rectangle(tile, (0, 0), (CELL[0], 24), (0, 0, 0), -1)
        cv2.putText(tile, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        r, c = divmod(i, cols)
        sheet[r * CELL[1]:(r + 1) * CELL[1], c * CELL[0]:(c + 1) * CELL[0]] = tile
    ok, buf = cv2.imencode(".jpg", sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes() if ok else None


def prompt_for(question: str, camera_name: str, labels: Sequence[str]) -> str:
    return (
        f"These {len(labels)} frames come from one camera ({camera_name}), in time order, marked {', '.join(labels)}. "
        f"Question: {question.strip()}\n"
        "Answer in one or two plain sentences using only what you can see in the frames. Say how many when counting. "
        f"If the frames do not show it, answer: {NOT_VISIBLE}. Do not name people or guess who they are."
    )


def yes_no(answer: str) -> bool | None:
    """True or False when the answer opens with yes or no, else None."""
    head = answer.strip().lower()
    if head.startswith(("yes", "yeah")):
        return True
    if head.startswith(("no,", "no.", "no ", "nobody", "no one", "none", "not ")) or head == "no":
        return False
    return None


def cannot_tell(answer: str) -> bool:
    return answer.strip().lower().startswith(NOT_VISIBLE.lower()) or "cannot tell" in answer.lower()[:80]


class LookAnswerer:
    """Shows frames of a camera to the local vision model and returns its answer."""

    def __init__(self, gateway: Gateway, media_dir: Path, model: str | None = None) -> None:
        self._gateway = gateway
        self._media = Path(media_dir)
        self._model = model

    async def ask(self, question: str, camera_name: str, frames: Sequence[Frame], tz: tzinfo = UTC) -> str | None:
        import cv2

        images, labels = [], []
        for letter, frame in zip("ABCD", frames, strict=False):
            image = cv2.imread(str(self._media / frame.path))
            if image is None:
                continue
            images.append(image)
            labels.append(f"{letter} {datetime.fromtimestamp(frame.t, tz).strftime('%H:%M:%S')}")
        sheet = contact_sheet(images, labels)
        if sheet is None:
            return None
        answer = await self._gateway.vision_text(sheet, prompt_for(question, camera_name, labels), local_only=True,
                                                 max_tokens=ANSWER_TOKENS, model=self._model)
        return answer.strip() if answer else None


def pick_frames(rows: Sequence[tuple[float, str]], camera_id: str, n: int = FRAMES_PER_CAMERA) -> list[Frame]:
    """n frames spread over the available ones, first and last included."""
    from evora.query.objects import sample_evenly

    return [Frame(camera_id, t, path) for t, path in sample_evenly(list(rows), n)]


def as_dict(frames: Sequence[Frame]) -> list[dict[str, Any]]:
    return [{"camera_id": f.camera_id, "t": f.t, "path": f.path} for f in frames]
