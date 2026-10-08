"""Replay-as-live: a local MediaMTX server and one ffmpeg per camera that loops a recorded file as an RTSP stream."""
from __future__ import annotations

import atexit
import logging
import os
import re
import shutil
import socket
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from evora.core import cameras as cams
from evora.core.db import Database

log = logging.getLogger("evora.live")

RTSP_HOST = "127.0.0.1"  # everything stays on this machine; the privacy guard is not needed for subprocesses
MIN_SPEED, MAX_SPEED = 0.25, 8.0
QUICK_EXIT_S = 4.0       # an ffmpeg that dies sooner than this with `-c copy` is retried as a transcode
MAX_RESTARTS = 3
_CAMERA_ID = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
INSTALL_HINT = "Install it with `winget install bluenviron.mediamtx` (Windows) or from github.com/bluenviron/mediamtx/releases."


class LiveError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


class ProcessLike(Protocol):
    def poll(self) -> int | None: ...
    def terminate(self) -> None: ...
    def kill(self) -> None: ...
    def wait(self, timeout: float | None = None) -> int: ...


Spawn = Callable[..., ProcessLike]


def find_mediamtx(configured: str | None = None) -> Path | None:
    """Configured path, then PATH, then where winget puts it."""
    if configured:
        p = Path(configured)
        return p if p.is_file() else None
    found = shutil.which("mediamtx")
    if found:
        return Path(found)
    local = os.environ.get("LOCALAPPDATA")
    if local:
        packages = Path(local) / "Microsoft" / "WinGet" / "Packages"
        if packages.is_dir():
            for exe in sorted(packages.glob("bluenviron.mediamtx*/mediamtx.exe")):
                return exe
    return None


def mediamtx_config(port: int) -> str:
    """RTSP on loopback only; every other protocol and the control APIs are off."""
    return (
        "logLevel: info\n"
        "logDestinations: [stdout]\n"
        "api: false\nmetrics: false\npprof: false\nplayback: false\n"
        "rtmp: false\nhls: false\nwebrtc: false\nsrt: false\n"
        "rtsp: true\n"
        "rtspTransports: [tcp]\n"
        f"rtspAddress: {RTSP_HOST}:{port}\n"
        "rtspEncryption: \"no\"\n"
        "paths:\n  all_others:\n"
    )


def ffmpeg_command(ffmpeg: str, source: str, url: str, speed: float, transcode: bool) -> list[str]:
    """Loop `source` at `speed` times real time into `url`. Always an argument list, never a shell string."""
    pace = ["-re"] if speed == 1.0 else ["-readrate", f"{speed:g}"]
    video = (
        ["-c:v", "libx264", "-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p", "-g", "25"]
        if transcode else ["-c:v", "copy"]
    )
    return [
        ffmpeg, "-hide_banner", "-loglevel", "error", "-nostdin", *pace, "-stream_loop", "-1", "-i", source,
        "-an", *video, "-f", "rtsp", "-rtsp_transport", "tcp", url,
    ]


def port_open(port: int, host: str = RTSP_HOST, timeout: float = 0.3) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def _tail(path: Path, lines: int = 3) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " | ".join(text[-lines:])


@dataclass
class Stream:
    camera_id: str
    source: str
    url: str
    speed: float
    state: str = "starting"           # starting | running | retrying | failed | stopped
    transcode: bool = False
    restarts: int = 0
    started_at: float = 0.0           # wall clock when ffmpeg last started (the file position is zero at this moment)
    error: str | None = None
    proc: ProcessLike | None = field(default=None, repr=False)
    stop: threading.Event = field(default_factory=threading.Event, repr=False)


class ReplayManager:
    def __init__(
        self, db: Database, live_dir: Path, *, mediamtx_path: str | None = None, port: int = 8554,
        max_streams: int = 16, default_speed: float = 1.0, spawn: Spawn | None = None,
        on_state: Callable[[str, str], None] | None = None, ffmpeg: str | None = None,
    ) -> None:
        self.db, self.live_dir, self.port = db, live_dir, port
        self.max_streams, self.default_speed = max_streams, default_speed
        self._configured = mediamtx_path
        self._spawn: Spawn = spawn or self._popen
        self._on_state = on_state
        self._ffmpeg = ffmpeg
        self._lock = threading.RLock()
        self._streams: dict[str, Stream] = {}
        self._server: ProcessLike | None = None
        self._server_binary: Path | None = None
        atexit.register(self.shutdown)

    # --- processes ---
    @staticmethod
    def _popen(args: list[str], log_file: Path) -> ProcessLike:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        flags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        with log_file.open("ab") as fh:
            return subprocess.Popen(
                args, stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=flags,
                cwd=log_file.parent,  # MediaMTX writes auto.crt and auto.key next to where it starts: keep them in the workspace
            )

    @staticmethod
    def _kill(proc: ProcessLike | None) -> None:
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3)
        except (subprocess.TimeoutExpired, OSError):
            proc.kill()

    def _announce(self, stream: Stream, state: str) -> None:
        stream.state = state
        if self._on_state is not None:
            try:
                self._on_state(stream.camera_id, state)
            except Exception:  # noqa: BLE001 - a UI notification must never break streaming
                log.exception("state callback failed")

    # --- server ---
    def server_ready(self) -> bool:
        return self._server is not None and self._server.poll() is None and port_open(self.port)

    def _ensure_server(self, wait_s: float = 8.0) -> None:
        if self.server_ready():
            return
        binary = find_mediamtx(self._configured)
        if binary is None:
            raise LiveError(503, f"MediaMTX is not installed, so streams cannot be served. {INSTALL_HINT}")
        if port_open(self.port):
            raise LiveError(409, f"Port {self.port} is already in use by another program; stop it or change live.rtsp_port.")
        self.live_dir.mkdir(parents=True, exist_ok=True)
        config = self.live_dir / "mediamtx.yml"
        config.write_text(mediamtx_config(self.port), encoding="utf-8")
        log_file = self.live_dir / "mediamtx.log"
        self._server = self._spawn([str(binary), str(config)], log_file)
        self._server_binary = binary
        end = time.monotonic() + wait_s
        while time.monotonic() < end:
            if self._server.poll() is not None:
                raise LiveError(502, f"MediaMTX stopped right after starting: {_tail(log_file)}")
            if port_open(self.port):
                return
            time.sleep(0.1)
        self._kill(self._server)
        self._server = None
        raise LiveError(502, f"MediaMTX did not start listening within {wait_s:g} s: {_tail(log_file)}")

    # --- streams ---
    def _launch(self, stream: Stream) -> None:
        ffmpeg = self._ffmpeg or shutil.which("ffmpeg")
        if ffmpeg is None:
            raise LiveError(500, "ffmpeg is not installed")
        args = ffmpeg_command(ffmpeg, stream.source, stream.url, stream.speed, stream.transcode)
        stream.proc = self._spawn(args, self.live_dir / f"{stream.camera_id}.log")
        stream.started_at = time.time()

    def _watch(self, stream: Stream) -> None:
        """Restart a stream that dies on its own: once as a transcode if it died instantly, then a few more times."""
        launched = time.monotonic()
        while not stream.stop.is_set():
            proc = stream.proc
            if proc is not None and proc.poll() is None:
                if stream.state != "running" and time.monotonic() - launched > 1.0:
                    self._announce(stream, "running")
                stream.stop.wait(0.3)
                continue
            if stream.stop.is_set():
                return
            lived = time.monotonic() - launched
            detail = _tail(self.live_dir / f"{stream.camera_id}.log")
            if lived < QUICK_EXIT_S and not stream.transcode:
                stream.transcode = True
                log.info("%s: copy failed quickly, retrying as a transcode (%s)", stream.camera_id, detail)
            elif stream.restarts >= MAX_RESTARTS:
                stream.error = detail or "ffmpeg keeps stopping"
                self._announce(stream, "failed")
                return
            else:
                stream.restarts += 1
            self._announce(stream, "retrying")
            try:
                self._launch(stream)
            except LiveError as exc:
                stream.error = exc.message
                self._announce(stream, "failed")
                return
            launched = time.monotonic()
            stream.stop.wait(0.2)

    def start(self, camera_ids: list[str], speed: float | None = None) -> dict[str, Any]:
        speed = self.default_speed if speed is None else float(speed)
        if not MIN_SPEED <= speed <= MAX_SPEED:
            raise LiveError(422, f"speed must be between {MIN_SPEED:g} and {MAX_SPEED:g}")
        if not camera_ids:
            raise LiveError(422, "choose at least one camera")
        plan: list[tuple[str, str]] = []
        for cid in dict.fromkeys(camera_ids):  # validate everything before starting anything
            if not _CAMERA_ID.match(cid):
                raise LiveError(422, f"invalid camera id: {cid!r}")
            try:
                cam = cams.get_camera(self.db, cid)
            except cams.CameraNotFound:
                raise LiveError(404, f"unknown camera: {cid}") from None
            if cam.kind != "file":
                raise LiveError(409, f"{cam.name} is already a live source; only recorded files can be replayed")
            if not Path(cam.source_uri).is_file():
                raise LiveError(404, f"the recording for {cam.name} is missing")
            plan.append((cid, cam.source_uri))
        with self._lock:
            active = {c for c, s in self._streams.items() if s.state not in ("stopped", "failed")}
            if len(active | {c for c, _ in plan}) > self.max_streams:
                raise LiveError(409, f"at most {self.max_streams} streams can run at once")
            self._ensure_server()
            for cid, source in plan:
                existing = self._streams.get(cid)
                if existing and existing.state in ("starting", "running", "retrying") and existing.speed == speed:
                    continue
                self._stop_one(cid)
                stream = Stream(cid, source, f"rtsp://{RTSP_HOST}:{self.port}/{cid}", speed)
                self._streams[cid] = stream
                self._launch(stream)
                threading.Thread(target=self._watch, args=(stream,), name=f"replay-{cid}", daemon=True).start()
        return self.status()

    def _stop_one(self, camera_id: str) -> bool:
        stream = self._streams.get(camera_id)
        if stream is None or stream.state == "stopped":
            return False
        stream.stop.set()
        self._kill(stream.proc)
        self._announce(stream, "stopped")
        return True

    def stop(self, camera_ids: list[str] | None = None) -> dict[str, Any]:
        with self._lock:
            for cid in list(camera_ids if camera_ids is not None else self._streams):
                self._stop_one(cid)
            if camera_ids is None or not any(s.state not in ("stopped", "failed") for s in self._streams.values()):
                self._kill(self._server)
                self._server = None
        return self.status()

    def status(self) -> dict[str, Any]:
        with self._lock:
            return {
                "server": {
                    "ready": self.server_ready(), "port": self.port, "binary_found": find_mediamtx(self._configured) is not None,
                },
                "streams": [
                    {
                        "camera_id": s.camera_id, "url": s.url, "state": s.state, "speed": s.speed, "restarts": s.restarts,
                        "transcoding": s.transcode, "started_at": s.started_at or None, "error": s.error,
                    }
                    for s in self._streams.values()
                ],
            }

    def shutdown(self) -> None:
        with self._lock:
            for cid in list(self._streams):
                self._stop_one(cid)
            self._kill(self._server)
            self._server = None
