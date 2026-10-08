"""Rolling recorder: keeps the last few minutes of a real RTSP camera on disk, so alerts on it have playable evidence.

One supervised ffmpeg per camera copies the stream (no re-encoding) into short MPEG-TS segments named after the wall-clock
second they started (`20261008_214956.ts`, local time). The index is just the folder: a segment spans from its name to the
moment it was last written, so it survives restarts and needs no database. Old segments are deleted by age and by a disk cap.
The recordings are raw video on this machine; they are only ever served through the blur-by-default media routes.
"""
from __future__ import annotations

import logging
import re
import shutil
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from evora.live.restream import LiveError, ProcessLike, ReplayManager, Spawn, _tail

log = logging.getLogger("evora.live.recorder")

STAMP = "%Y%m%d_%H%M%S"
_NAME = re.compile(r"^(\d{8}_\d{6})\.ts$")
_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
GAP_TOLERANCE_S = 2.0  # a moment this close after a segment's last write still belongs to it
MAX_BACKOFF_S = 10.0
STALL_SEGMENTS = 3  # no new data for this many segment lengths while "recording" means the stream went quiet


@dataclass(frozen=True)
class Segment:
    camera_id: str
    path: Path
    start: float  # wall clock, epoch seconds
    end: float


def segment_command(ffmpeg: str, url: str, directory: Path, segment_s: float) -> list[str]:
    """Copy `url` into `directory` as segments named by their start time. An argument list, never a shell string."""
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", "-rtsp_transport", "tcp", "-i", url, "-an", "-c:v", "copy",
        "-f", "segment", "-segment_time", f"{segment_s:g}", "-segment_format", "mpegts", "-reset_timestamps", "1",
        "-strftime", "1", str(directory / f"{STAMP}.ts"),
    ]


class RecordingIndex:
    """What is on disk for each camera, and the retention rules."""

    def __init__(self, root: Path, segment_s: float = 10.0, keep_s: float = 1800.0, max_bytes: int = 2_000_000_000) -> None:
        self.root, self.segment_s, self.keep_s, self.max_bytes = root, segment_s, keep_s, max_bytes

    def directory(self, camera_id: str) -> Path:
        if not _CAMERA_ID.match(camera_id):
            raise ValueError(f"invalid camera id: {camera_id!r}")
        return self.root / camera_id

    def segments(self, camera_id: str) -> list[Segment]:
        folder = self.directory(camera_id)
        out: list[Segment] = []
        for path in folder.glob("*.ts") if folder.is_dir() else []:
            m = _NAME.match(path.name)
            if m is None:
                continue
            try:
                start = datetime.strptime(m.group(1), STAMP).timestamp()  # local wall clock, as ffmpeg named it
                stat = path.stat()
            except (ValueError, OSError):
                continue
            if stat.st_size == 0:  # ffmpeg opened it but failed before writing anything (it is restarted): nothing to play
                continue
            end = stat.st_mtime
            out.append(Segment(camera_id, path, start, max(end, start)))
        out.sort(key=lambda s: s.start)
        if not out:
            return []
        # a segment ends no later than the next one starts
        return [Segment(s.camera_id, s.path, s.start, min(s.end, nxt.start) if nxt else s.end)
                for s, nxt in zip(out, [*out[1:], None], strict=True)]

    def has_recording(self, camera_id: str) -> bool:
        try:
            return bool(self.segments(camera_id))
        except ValueError:
            return False

    def at(self, camera_id: str, t: float) -> Segment | None:
        """The segment holding wall-clock time `t` (a moment just after its last write still counts)."""
        found = None
        for seg in self.segments(camera_id):
            if seg.start <= t:
                found = seg
            else:
                break
        if found is not None and t <= found.end + GAP_TOLERANCE_S:
            return found
        return None

    def window(self, camera_id: str, lo: float, hi: float) -> list[Segment]:
        """Segments that overlap [lo, hi], oldest first."""
        return [s for s in self.segments(camera_id) if s.end >= lo - GAP_TOLERANCE_S and s.start <= hi]

    def buffered_s(self, camera_id: str) -> float:
        segs = self.segments(camera_id)
        return round(sum(s.end - s.start for s in segs), 1)

    def stats(self, camera_id: str) -> dict[str, Any]:
        segs = self.segments(camera_id)
        return {
            "segments": len(segs), "buffered_s": round(sum(s.end - s.start for s in segs), 1),
            "bytes": sum(s.path.stat().st_size for s in segs if s.path.exists()),
            "oldest": segs[0].start if segs else None, "newest": segs[-1].end if segs else None,
        }

    def prune(self, camera_id: str, now: float | None = None) -> int:
        """Delete segments older than the retention window, then the oldest ones while over the disk cap.

        The newest segment is never deleted: it is the one being written.
        """
        now = time.time() if now is None else now
        segs = self.segments(camera_id)
        removed = 0
        keep: list[Segment] = []
        for i, seg in enumerate(segs):
            if i < len(segs) - 1 and seg.end < now - self.keep_s:
                removed += self._delete(seg)
            else:
                keep.append(seg)
        total = sum(s.path.stat().st_size for s in keep if s.path.exists())
        for seg in keep[:-1]:
            if total <= self.max_bytes:
                break
            size = seg.path.stat().st_size if seg.path.exists() else 0
            if self._delete(seg):
                removed += 1
                total -= size
        return removed

    @staticmethod
    def _delete(seg: Segment) -> int:
        try:
            seg.path.unlink()
            return 1
        except FileNotFoundError:
            return 0
        except OSError as exc:  # a player holding the file open on Windows: try again next time
            log.warning("could not delete %s: %s", seg.path.name, exc)
            return 0


@dataclass
class Recording:
    camera_id: str
    url: str
    state: str = "starting"           # starting | recording | retrying | stopped
    restarts: int = 0
    error: str | None = None
    since: float = 0.0
    proc: ProcessLike | None = field(default=None, repr=False)
    stop: threading.Event = field(default_factory=threading.Event, repr=False)


class Recorder:
    def __init__(
        self, index: RecordingIndex, log_dir: Path, *, ffmpeg: str | None = None, spawn: Spawn | None = None,
        on_state: Callable[[str, str], None] | None = None, backoff_s: float = 1.0, join_timeout_s: float = 5.0,
    ) -> None:
        self.index, self.log_dir = index, log_dir
        self._ffmpeg, self._spawn = ffmpeg, spawn or ReplayManager._popen
        self._on_state, self._backoff, self._join = on_state, backoff_s, join_timeout_s
        self._lock = threading.RLock()
        self._recs: dict[str, Recording] = {}
        self._threads: dict[str, threading.Thread] = {}

    # --- control ---
    def start(self, camera_id: str, url: str) -> None:
        """Start (or keep) recording `url` for a camera. Already recording the same address changes nothing."""
        folder = self.index.directory(camera_id)
        ffmpeg = self._ffmpeg or shutil.which("ffmpeg")
        if ffmpeg is None:
            raise LiveError(500, "ffmpeg is not installed")
        with self._lock:
            existing = self._recs.get(camera_id)
            if existing and existing.state != "stopped" and existing.url == url:
                return
            self._stop_one(camera_id)
            folder.mkdir(parents=True, exist_ok=True)
            rec = Recording(camera_id, url)
            self._recs[camera_id] = rec
            self._launch(rec, ffmpeg)
            thread = threading.Thread(target=self._watch, args=(rec, ffmpeg), name=f"record-{camera_id}", daemon=True)
            self._threads[camera_id] = thread
            thread.start()

    def stop(self, camera_ids: list[str] | None = None) -> None:
        with self._lock:
            targets = [c for c in (camera_ids if camera_ids is not None else list(self._recs)) if c in self._recs]
            for cid in targets:
                self._stop_one(cid)
            threads = [self._threads[c] for c in targets if c in self._threads]
        for t in threads:
            t.join(self._join)

    def shutdown(self) -> None:
        self.stop(None)

    def status(self) -> list[dict[str, Any]]:
        with self._lock:
            recs = list(self._recs.values())
        out = []
        for r in recs:
            stats = self.index.stats(r.camera_id)
            age = round(max(time.time() - stats["newest"], 0.0), 1) if stats["newest"] is not None else None
            # recording, yet nothing written for a few segments: the camera is connected but silent
            stalled = r.state == "recording" and age is not None and age > STALL_SEGMENTS * self.index.segment_s
            out.append({"camera_id": r.camera_id, "state": r.state, "restarts": r.restarts, "error": r.error,
                        "segments": stats["segments"], "buffered_s": stats["buffered_s"], "bytes": stats["bytes"],
                        "last_segment_age_s": age, "stalled": stalled})
        return out

    def is_recording(self, camera_id: str) -> bool:
        with self._lock:
            rec = self._recs.get(camera_id)
            return rec is not None and rec.state in ("starting", "recording", "retrying")

    # --- internals ---
    def _announce(self, rec: Recording, state: str) -> None:
        rec.state = state
        if self._on_state is not None:
            try:
                self._on_state(rec.camera_id, state)
            except Exception:  # noqa: BLE001 - a UI note must never break recording
                log.exception("recording state callback failed")

    def _launch(self, rec: Recording, ffmpeg: str) -> None:
        args = segment_command(ffmpeg, rec.url, self.index.directory(rec.camera_id), self.index.segment_s)
        rec.proc = self._spawn(args, self.log_dir / f"rec_{rec.camera_id}.log")
        rec.since = time.time()

    def _stop_one(self, camera_id: str) -> None:
        rec = self._recs.get(camera_id)
        if rec is None or rec.state == "stopped":
            return
        rec.stop.set()
        ReplayManager._kill(rec.proc)
        self._announce(rec, "stopped")

    def _watch(self, rec: Recording, ffmpeg: str) -> None:
        """Keep ffmpeg running (a camera that drops comes back by itself) and prune old segments."""
        launched = time.monotonic()
        last_prune = 0.0
        failures = 0
        while not rec.stop.is_set():
            proc = rec.proc
            if proc is not None and proc.poll() is None:
                if rec.state != "recording" and time.monotonic() - launched > 1.0:
                    rec.error, failures = None, 0
                    self._announce(rec, "recording")
                if time.monotonic() - last_prune >= max(self.index.segment_s, 1.0):
                    last_prune = time.monotonic()
                    try:
                        self.index.prune(rec.camera_id)
                    except OSError:
                        log.exception("pruning %s failed", rec.camera_id)
                rec.stop.wait(0.3)
                continue
            if rec.stop.is_set():
                return
            rec.error = _tail(self.log_dir / f"rec_{rec.camera_id}.log") or "ffmpeg stopped"
            failures += 1
            rec.restarts += 1
            self._announce(rec, "retrying")
            if rec.stop.wait(min(self._backoff * 2 ** (failures - 1), MAX_BACKOFF_S)):
                return
            try:
                self._launch(rec, ffmpeg)
            except (OSError, LiveError) as exc:
                rec.error = str(exc)[:200]
                continue
            launched = time.monotonic()


def disk_cap_bytes(gigabytes: float) -> int:
    return int(max(gigabytes, 0.0) * 1_000_000_000)


__all__ = ["Recorder", "RecordingIndex", "Segment", "disk_cap_bytes", "segment_command"]
