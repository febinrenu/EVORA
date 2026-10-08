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


def _even(x: float) -> int:
    return max(2, int(round(x)) // 2 * 2)


class Frame:
    """One decoded frame. Pixels are converted only when `image` is first read.

    Converting to BGR is the most expensive step after decoding, and most frames are skipped by the
    samplers, so `pts_s` and the cheap grey `proxy()` are available without paying for it.
    """

    __slots__ = ("index", "pts_s", "_av", "_rotation", "_max_width", "_image")

    def __init__(self, index: int, pts_s: float, av_frame: av.VideoFrame | None, rotation: int, max_width: int,
                 image: np.ndarray | None = None):
        self.index, self.pts_s = index, pts_s
        self._av, self._rotation, self._max_width, self._image = av_frame, rotation, max_width, image

    @classmethod
    def from_image(cls, index: int, pts_s: float, image: np.ndarray) -> Frame:
        return cls(index, pts_s, None, 0, image.shape[1], image)

    @property
    def image(self) -> np.ndarray:
        """BGR uint8, rotated upright and no wider than `max_width` (scaled during conversion)."""
        if self._image is None:
            f = self._av
            upright_w = f.height if self._rotation in (90, 270) else f.width
            if upright_w > self._max_width:
                scale = self._max_width / upright_w
                arr = f.reformat(width=_even(f.width * scale), height=_even(f.height * scale), format="bgr24",
                                 interpolation="AREA").to_ndarray()
            else:
                arr = f.to_ndarray(format="bgr24")
            self._image = _apply_rotation(arr, self._rotation)
            self._av = None   # the decoder buffer is no longer needed
        return self._image

    def proxy(self, width: int) -> np.ndarray:
        """Small grey copy (about `width` px wide, not rotated) for motion measurement."""
        if self._image is not None:
            img = self._image
            h, w = img.shape[:2]
            small = cv2.resize(img, (width, max(1, int(round(h * width / w)))), interpolation=cv2.INTER_AREA)
            return cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        f = self._av
        name = f.format.name
        if (name.startswith("yuv") or name.startswith("nv")) and not any(t in name for t in ("10", "12", "16")):
            # 8-bit YUV keeps brightness in plane 0: read it in place and take every n-th pixel (no conversion)
            plane = f.planes[0]
            luma = np.frombuffer(plane, dtype=np.uint8).reshape(f.height, plane.line_size)[:, : f.width]
            step = max(1, f.width // width)
            small = np.ascontiguousarray(luma[::step, ::step])
            return cv2.resize(small, (width, max(1, int(round(f.height * width / f.width)))), interpolation=cv2.INTER_AREA)
        return f.reformat(width=width, height=max(2, int(round(f.height * width / f.width))), format="gray",
                          interpolation="AREA").to_ndarray()


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
    """Yield decoded frames in presentation order; `Frame.image` is an upright BGR image no wider than `max_width`.

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
        rotation: int | None = None
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
                if rotation is None:
                    rotation = tag_rotation or _frame_rotation(frame)   # the display matrix is constant for a stream
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
                yield Frame(index, pts, frame, rotation, max_width)
