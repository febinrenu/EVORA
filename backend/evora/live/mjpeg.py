"""Live tiles: a low-rate MJPEG stream of a camera, with every frame face-blurred before it is sent."""
from __future__ import annotations

import asyncio
import logging
import shutil
import threading
from collections.abc import AsyncIterator, Callable

from fastapi.concurrency import run_in_threadpool

from evora.core.media_service import BlurFailed, BlurFn
from evora.live.restream import LiveError

log = logging.getLogger("evora.live.mjpeg")

SOI, EOI = b"\xff\xd8", b"\xff\xd9"
BOUNDARY = "frame"
MIN_FPS, MAX_FPS = 1, 5
TILE_WIDTH = 640


class JpegSplitter:
    """Cuts a byte stream of back-to-back JPEGs into frames, whatever the chunk boundaries are."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, data: bytes) -> list[bytes]:
        self._buf += data
        frames: list[bytes] = []
        while True:
            start = self._buf.find(SOI)
            if start < 0:
                keep = bytes(self._buf[-1:]) if self._buf.endswith(SOI[:1]) else b""  # may be half of the next frame's start
                self._buf.clear()
                self._buf += keep  # everything else is garbage
                break
            if start:
                del self._buf[:start]
            end = self._buf.find(EOI, 2)  # entropy-coded data never contains FFD9, so the first one ends the frame
            if end < 0:
                break
            frames.append(bytes(self._buf[: end + 2]))
            del self._buf[: end + 2]
        return frames


def tile_command(ffmpeg: str, url: str, fps: int) -> list[str]:
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-rtsp_transport", "tcp", "-fflags", "nobuffer",
        "-flags", "low_delay", "-i", url, "-an", "-vf", f"fps={fps},scale={TILE_WIDTH}:-2", "-q:v", "5",
        "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1",
    ]


# a seam for tests: replaced by a command that emits scripted frames
COMMAND_FACTORY: Callable[[str, str, int], list[str]] = tile_command


class TileLimiter:
    """At most `maximum` tiles at once: each one is a decoder process."""

    def __init__(self, maximum: int = 8) -> None:
        self.maximum, self._n, self._lock = maximum, 0, threading.Lock()

    def acquire(self) -> bool:
        with self._lock:
            if self._n >= self.maximum:
                return False
            self._n += 1
            return True

    def release(self) -> None:
        with self._lock:
            self._n = max(self._n - 1, 0)

    @property
    def active(self) -> int:
        return self._n


def clamp_fps(fps: int) -> int:
    return min(max(fps, MIN_FPS), MAX_FPS)


async def stream_frames(
    url: str, fps: int, blur: BlurFn | None, blur_one: Callable[[BlurFn, bytes], bytes],
) -> AsyncIterator[bytes]:
    """Multipart chunks for one viewer. The decoder is killed when the viewer goes away."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None and COMMAND_FACTORY is tile_command:
        raise LiveError(500, "ffmpeg is not installed")
    cmd = COMMAND_FACTORY(ffmpeg or "ffmpeg", url, clamp_fps(fps))
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL, stdin=asyncio.subprocess.DEVNULL,
    )
    splitter = JpegSplitter()
    try:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(65536)
            if not chunk:
                break
            for frame in splitter.feed(chunk):
                if blur is not None:
                    try:
                        frame = await run_in_threadpool(blur_one, blur, frame)
                    except BlurFailed:
                        continue  # blur was promised: a frame that cannot be blurred is dropped, never sent raw
                yield (
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(frame)}\r\n\r\n".encode()
                    + frame + b"\r\n"
                )
    finally:
        if proc.returncode is None:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
