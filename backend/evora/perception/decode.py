"""Video decoding with PyAV.

Frames carry their presentation time (PTS) in seconds; frame index times fps is never used,
so variable frame rate files keep correct timestamps. Rotation metadata (phone video) is applied.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import av
import cv2
import numpy as np

log = logging.getLogger("evora.perception.decode")
MAX_CONSECUTIVE_ERRORS = 50


class DecodeError(RuntimeError):
    pass


@dataclass(frozen=True)
class VideoInfo:
    width: int            # after rotation, before downscale
    height: int
    fps: float | None
    duration_s: float | None
    rotation: int         # degrees clockwise to apply for upright display: 0, 90, 180, 270


@dataclass(frozen=True)
class Frame:
    index: int
    pts_s: float          # seconds from the start of the file
    image: np.ndarray     # BGR uint8, rotated upright and downscaled to max_width


def _clockwise_from_ccw(ccw_degrees: float) -> int:
    """FFmpeg reports display rotation counter-clockwise; return the clockwise quarter turns to apply, in degrees."""
    return (-int(round(ccw_degrees / 90.0)) * 90) % 360


def _frame_rotation(frame: av.VideoFrame) -> int:
    """Clockwise degrees (0, 90, 180, 270) needed to make a decoded frame upright."""
    ccw = getattr(frame, "rotation", None)
    if ccw is None:
        for side in getattr(frame, "side_data", None) or []:
            if "DISPLAYMATRIX" in str(getattr(side, "type", "")).upper():
                m = np.frombuffer(bytes(side), dtype="<i4")
                if m.size >= 2:
                    ccw = math.degrees(math.atan2(m[3], m[0]))
    return _clockwise_from_ccw(float(ccw)) if ccw else 0


def _stream_rotation_tag(stream: av.video.stream.VideoStream) -> int:
    """Older phone files carry a `rotate` metadata tag (clockwise degrees) instead of a display matrix."""
    tag = stream.metadata.get("rotate") if stream.metadata else None
    try:
        return int(float(tag)) % 360 if tag else 0
    except ValueError:
        return 0


def _apply_rotation(img: np.ndarray, rotation: int) -> np.ndarray:
    if rotation == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if rotation == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return img


def probe_video(path: str | Path) -> VideoInfo:
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                raise DecodeError(f"no video stream in {path}")
            stream = container.streams.video[0]
            rotation = _stream_rotation_tag(stream)
            if rotation == 0:
                for first in container.decode(stream):
                    rotation = _frame_rotation(first)
                    break
            w, h = stream.codec_context.width, stream.codec_context.height
            if rotation in (90, 270):
                w, h = h, w
            fps = float(stream.average_rate) if stream.average_rate else None
            if stream.duration is not None and stream.time_base is not None:
                duration = float(stream.duration * stream.time_base)
            elif container.duration is not None:
                duration = container.duration / av.time_base
            else:
                duration = None
            return VideoInfo(width=w, height=h, fps=fps, duration_s=duration, rotation=rotation)
    except (av.error.FFmpegError, OSError) as exc:
        raise DecodeError(f"cannot open {path}: {exc}") from exc


def read_frames(path: str | Path, max_width: int = 1280, *, start_s: float = 0.0, end_s: float | None = None) -> Iterator[Frame]:
    """Yield decoded frames in presentation order as upright BGR images no wider than `max_width`.

    A corrupted packet is logged and skipped; the generator keeps going until the stream ends.
    """
    try:
        container = av.open(str(path))
    except (av.error.FFmpegError, OSError) as exc:
        raise DecodeError(f"cannot open {path}: {exc}") from exc
    with container:
        if not container.streams.video:
            raise DecodeError(f"no video stream in {path}")
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        tag_rotation = _stream_rotation_tag(stream)
        fps = float(stream.average_rate) if stream.average_rate else None
        index = -1
        errors = 0
        packets = container.demux(stream)
        while True:
            try:
                packet = next(packets)
            except StopIteration:
                break
            except (av.error.FFmpegError, OSError) as exc:
                errors += 1
                log.warning("demux error in %s: %s", path, exc)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise DecodeError(f"giving up on {path} after {errors} read errors") from exc
                continue
            try:
                decoded = packet.decode()
            except (av.error.FFmpegError, OSError) as exc:
                errors += 1
                log.warning("decode error in %s: %s", path, exc)
                if errors >= MAX_CONSECUTIVE_ERRORS:
                    raise DecodeError(f"giving up on {path} after {errors} read errors") from exc
                continue
            errors = 0
            for frame in decoded:
                rotation = tag_rotation or _frame_rotation(frame)
                index += 1
                if frame.time is not None:
                    pts = float(frame.time)
                elif fps:
                    pts = index / fps
                else:
                    pts = float(index)
                if pts < start_s:
                    continue
                if end_s is not None and pts > end_s:
                    return
                img = frame.to_ndarray(format="bgr24")
                img = _apply_rotation(img, rotation)
                h, w = img.shape[:2]
                if w > max_width:
                    scale = max_width / w
                    img = cv2.resize(img, (max_width, int(round(h * scale)) // 2 * 2), interpolation=cv2.INTER_AREA)
                yield Frame(index=index, pts_s=pts, image=img)
