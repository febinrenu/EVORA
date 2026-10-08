"""Render frames, thumbnails and clips from stored footage, with a disk cache and face blur."""
from __future__ import annotations

import functools
import logging
import os
import secrets
import shutil
import subprocess
import tempfile
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path

from contracts.models import CameraInfo

from evora.core.workspace import Workspace
from evora.evidence.store import EvidenceRecord

log = logging.getLogger("evora.media_service")

@functools.cache
def probe_jpeg() -> bytes:
    """A small valid JPEG for checking that the blur model actually runs (a 1x1 file is rejected by OpenCV)."""
    import cv2
    import numpy as np

    ok, buf = cv2.imencode(".jpg", np.full((96, 96, 3), 128, dtype=np.uint8))
    if not ok:
        raise RuntimeError("could not build the blur probe image")
    return buf.tobytes()


FFMPEG_TIMEOUT_S = 120
UNBLUR_TTL_S = 300

BlurFn = Callable[[bytes], bytes]


class MediaError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class BlurFailed(Exception):
    """The face-blur function exists but could not run (for example its model file is missing)."""


class UnblurTokens:
    """Short-lived tokens that allow one viewer to see unblurred media. In memory on purpose."""

    def __init__(self, ttl_s: int = UNBLUR_TTL_S, clock: Callable[[], float] = time.time):
        self._ttl, self._clock = ttl_s, clock
        self._tokens: dict[str, tuple[float, str]] = {}  # token -> (expires, stated reason)
        self._lock = threading.Lock()

    def issue(self, reason: str = "") -> tuple[str, float]:
        token, expires = secrets.token_urlsafe(16), self._clock() + self._ttl
        with self._lock:
            self._tokens = {t: v for t, v in self._tokens.items() if v[0] > self._clock()}
            self._tokens[token] = (expires, reason)
        return token, expires

    def valid(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock:
            return self._tokens.get(token, (0.0, ""))[0] > self._clock()

    def reason_of(self, token: str | None) -> str | None:
        """The reason given when the token was issued, while it is still valid."""
        if not self.valid(token):
            return None
        with self._lock:
            return self._tokens[token][1]  # type: ignore[index]


class MediaService:
    def __init__(
        self, ws: Workspace, cfg: dict, blur_provider: Callable[[], BlurFn | None],
        offset_fn: Callable[[CameraInfo, float], float | None] | None = None,
    ):
        self.ws = ws
        self._offset_fn = offset_fn  # wall-clock time -> position in the file, for footage replayed as live
        self.pre_roll = float(cfg["media"]["pre_roll_s"])
        self.post_roll = float(cfg["media"]["post_roll_s"])
        self._blur_provider = blur_provider
        self.cache_max_bytes = int(cfg["media"].get("cache_max_bytes", 0))
        self._locks: dict[Path, threading.Lock] = {}
        self._guard = threading.Lock()
        self.ffmpeg_calls = 0
        self._probe: tuple[float, int, bool] | None = None  # (when, id of the blur function, did it work)
        (ws.media_dir / "thumbs").mkdir(parents=True, exist_ok=True)
        ws.clips_dir.mkdir(parents=True, exist_ok=True)

    # --- helpers ---
    def _lock_for(self, path: Path) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(path, threading.Lock())

    def _ffmpeg(self, args: list[str]) -> bytes:
        exe = shutil.which("ffmpeg")
        if exe is None:
            raise MediaError(500, "ffmpeg is not installed")
        with self._guard:
            self.ffmpeg_calls += 1
        try:
            out = subprocess.run(
                [exe, "-v", "error", "-nostdin", *args], capture_output=True, timeout=FFMPEG_TIMEOUT_S, check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise MediaError(502, "rendering timed out") from exc
        if out.returncode != 0:
            log.error("ffmpeg failed: %s", out.stderr.decode(errors="replace")[:500])
            raise MediaError(502, "could not render this footage")
        return out.stdout

    @staticmethod
    def source_of(cam: CameraInfo) -> Path:
        if cam.kind != "file":
            raise MediaError(409, "this camera has no recorded file")
        path = Path(cam.source_uri)
        if not path.is_file():
            raise MediaError(404, "source video is missing")
        return path

    def blur_status(self, want_blur: bool) -> str:
        """`off`, `applied` or `unavailable`, by actually running the blur on a tiny image (cached for 30 s)."""
        if not want_blur:
            return "off"
        fn = self._blur_provider()
        if fn is None:
            return "unavailable"
        now = time.monotonic()
        if self._probe is None or self._probe[1] != id(fn) or now - self._probe[0] > 30:
            try:
                fn(probe_jpeg())
                ok = True
            except Exception:  # noqa: BLE001 - any failure means blur cannot be promised
                ok = False
            self._probe = (now, id(fn), ok)
        return "applied" if self._probe[2] else "unavailable"

    def blur_function(self) -> BlurFn | None:
        """The face-blur function if the model is installed (no logging; callers decide what to do without it)."""
        return self._blur_provider()

    def blur_state(self, want_blur: bool) -> tuple[BlurFn | None, str]:
        """(function to apply or None, header value)."""
        if not want_blur:
            return None, "off"
        fn = self._blur_provider()
        if fn is None:
            log.warning("face blur requested but perception.blur_faces is not installed; serving unblurred media")
            return None, "unavailable"
        return fn, "applied"

    def _mapped(self, cam: CameraInfo, t: float) -> float | None:
        return self._offset_fn(cam, t) if self._offset_fn is not None else None

    def _rel(self, cam: CameraInfo, t: float) -> float:
        top = max((cam.duration_s or 0) - 1.5 / (cam.fps or 25.0), 0.0)  # a seek past the last frame yields nothing
        mapped = self._mapped(cam, t)
        return min(max(mapped if mapped is not None else t - cam.t0, 0.0), top)

    def _clip_window(self, cam: CameraInfo, rec: EvidenceRecord) -> tuple[float, float]:
        """(start, end) in file seconds. Replayed-as-live evidence maps through its session and may cross the loop seam."""
        duration = cam.duration_s or 0.0
        lo, hi = rec.t_start - self.pre_roll, rec.t_end + self.post_roll
        peak = self._mapped(cam, rec.t_peak)
        if peak is None:  # recorded footage
            return max(lo - cam.t0, 0.0), min(hi - cam.t0, duration)
        start, end = self._mapped(cam, lo), self._mapped(cam, hi)
        if start is None:
            start = max(peak - (rec.t_peak - lo), 0.0)
        if end is None:
            end = min(peak + (hi - rec.t_peak), duration)
        if end <= start:  # the window crosses the loop seam: cut it at the end of the loop
            end = duration
        return start, end

    # --- frames and thumbnails ---
    def frame(
        self, cam: CameraInfo, t: float, want_blur: bool, bbox: tuple[float, float, float, float] | None = None,
    ) -> tuple[bytes, str]:
        src = self.source_of(cam)
        args = ["-ss", f"{self._rel(cam, t):.3f}", "-i", str(src), "-frames:v", "1"]
        if bbox is not None:
            x1, y1, x2, y2 = (min(max(v, 0.0), 1.0) for v in bbox)
            w, h = max(x2 - x1, 0.001), max(y2 - y1, 0.001)
            args += ["-vf", f"drawbox=x=iw*{x1:.4f}:y=ih*{y1:.4f}:w=iw*{w:.4f}:h=ih*{h:.4f}:color=yellow:t=3"]
        jpeg = self._ffmpeg([*args, "-pix_fmt", "yuvj420p", "-q:v", "3", "-f", "image2pipe", "-c:v", "mjpeg", "pipe:1"])
        if not jpeg:
            raise MediaError(502, "no frame at that time")
        fn, status = self.blur_state(want_blur)
        if fn is None:
            return jpeg, status
        try:
            return self._blur(fn, jpeg), status
        except BlurFailed:
            return jpeg, "unavailable"  # reported honestly; the UI warns and nothing pretends to be blurred

    def _blur(self, fn: BlurFn, data: bytes) -> bytes:
        try:
            return fn(data)
        except Exception as exc:  # noqa: BLE001 - any failure of the blur model must degrade, never crash a media request
            log.warning("face blur failed (%s): %s", exc.__class__.__name__, exc)
            raise BlurFailed from exc

    def thumb(self, cam: CameraInfo, rec: EvidenceRecord, want_blur: bool) -> tuple[Path, str]:
        fn, status = self.blur_state(want_blur)
        thumbs = self.ws.media_dir / "thumbs"
        path = thumbs / f"{rec.id}_{'blur' if fn else 'raw'}.jpg"
        with self._lock_for(path):
            if path.is_file():
                os.utime(path)
                return path, status
            data, actual = self.frame(cam, rec.t_peak, want_blur, rec.bbox)
            if fn is not None and actual != "applied":  # the blur failed: never file an unblurred image as the blurred one
                path = thumbs / f"{rec.id}_raw.jpg"
            self._atomic_write(path, data)
            self.trim_cache(keep=path)
            return path, actual

    # --- clips ---
    def clip(self, cam: CameraInfo, rec: EvidenceRecord, want_blur: bool) -> tuple[Path, str]:
        fn, status = self.blur_state(want_blur)
        raw = self.ws.clips_dir / f"{rec.id}_raw.mp4"
        with self._lock_for(raw):
            if raw.is_file():
                os.utime(raw)
            else:
                self._render_clip(cam, rec, raw)
                self.trim_cache(keep=raw)
        if fn is None:
            return raw, status
        blurred = self.ws.clips_dir / f"{rec.id}_blur.mp4"
        with self._lock_for(blurred):
            if blurred.is_file():
                os.utime(blurred)
            else:
                try:
                    self._blur_clip(cam, raw, blurred, fn)
                except BlurFailed:
                    return raw, "unavailable"
                self.trim_cache(keep=blurred, also_keep=raw)
        return blurred, status

    def _render_clip(self, cam: CameraInfo, rec: EvidenceRecord, out: Path) -> None:
        src = self.source_of(cam)
        start, end = self._clip_window(cam, rec)
        if end - start <= 0.05:
            raise MediaError(422, "that moment is outside the recorded footage")
        tmp = out.with_name(f"{out.stem}.{uuid.uuid4().hex[:6]}.tmp.mp4")
        try:
            self._ffmpeg([
                "-ss", f"{start:.3f}", "-i", str(src), "-t", f"{end - start:.3f}", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-y", str(tmp),
            ])
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)

    def _blur_clip(self, cam: CameraInfo, raw: Path, out: Path, fn: BlurFn) -> None:
        """Decode to JPEG frames, blur each, re-encode. Disk based so no pixel library is needed."""
        fps = cam.fps or 25.0
        tmp_out = out.with_name(f"{out.stem}.{uuid.uuid4().hex[:6]}.tmp.mp4")
        with tempfile.TemporaryDirectory(dir=self.ws.clips_dir) as td:
            frames = Path(td)
            try:
                self._ffmpeg(["-i", str(raw), "-q:v", "3", str(frames / "%06d.jpg")])
                for jpg in sorted(frames.glob("*.jpg")):
                    jpg.write_bytes(self._blur(fn, jpg.read_bytes()))
                self._ffmpeg([
                    "-framerate", f"{fps:.3f}", "-i", str(frames / "%06d.jpg"), "-c:v", "libx264",
                    "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    "-y", str(tmp_out),
                ])
                os.replace(tmp_out, out)
            finally:
                tmp_out.unlink(missing_ok=True)

    def trim_cache(self, keep: Path | None = None, also_keep: Path | None = None) -> int:
        """Delete least recently used cached files until the cache fits its cap. Returns files removed."""
        if self.cache_max_bytes <= 0:
            return 0
        files = [
            p for d in (self.ws.media_dir / "thumbs", self.ws.clips_dir) for p in d.glob("*")
            if p.is_file() and ".tmp" not in p.name
        ]
        stats = {p: p.stat() for p in files}
        total = sum(st.st_size for st in stats.values())
        removed = 0
        for p in sorted(files, key=lambda f: stats[f].st_mtime):
            if total <= self.cache_max_bytes:
                break
            if p in (keep, also_keep):
                continue
            try:
                p.unlink()
            except OSError:  # still being served (Windows) or already gone: try again next time
                continue
            total -= stats[p].st_size
            removed += 1
        return removed

    @staticmethod
    def _atomic_write(path: Path, data: bytes) -> None:
        tmp = path.with_name(f"{path.stem}.{uuid.uuid4().hex[:6]}.tmp")
        try:
            tmp.write_bytes(data)
            os.replace(tmp, path)
        finally:
            tmp.unlink(missing_ok=True)
