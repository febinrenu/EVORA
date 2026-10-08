"""Upload handling: safe filenames, capped streaming writes with SHA-256, and ffprobe validation."""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

PROBE_TIMEOUT_S = 30
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class UploadError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(frozen=True)
class ProbeResult:
    codec: str
    width: int
    height: int
    fps: float | None
    rotation: int
    duration_s: float
    creation_time: str | None


def safe_filename(raw: str, allowed_ext: list[str]) -> str:
    """Basename only, no leading dots, limited charset; the extension must be on the allow-list."""
    base = Path(raw.replace("\\", "/")).name
    stem, ext = Path(base).stem, Path(base).suffix.lower()
    if ext not in allowed_ext:
        raise UploadError(422, f"unsupported file type '{ext or base}'; allowed: {', '.join(allowed_ext)}")
    stem = _UNSAFE.sub("_", stem).strip("._") or "video"
    return f"{stem[:80]}{ext}"


class UploadSink:
    """Writes one upload to `<uploads_dir>/<uuid>_<name>` in chunks, enforcing the size cap."""

    def __init__(self, uploads_dir: Path, filename: str, max_bytes: int, allowed_ext: list[str]):
        self.name = safe_filename(filename, allowed_ext)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        self.path = uploads_dir / f"{uuid.uuid4().hex[:8]}_{self.name}"
        self._max = max_bytes
        self._size = 0
        self._sha = hashlib.sha256()
        self._fh = self.path.open("wb")

    def write(self, chunk: bytes) -> None:
        self._size += len(chunk)
        if self._size > self._max:
            self.discard()
            raise UploadError(413, f"file exceeds the {self._max} byte limit")
        self._sha.update(chunk)
        self._fh.write(chunk)

    def finish(self) -> tuple[Path, str]:
        self._fh.close()
        if self._size == 0:
            self.discard()
            raise UploadError(422, "empty file")
        return self.path, self._sha.hexdigest()

    def discard(self) -> None:
        if not self._fh.closed:
            self._fh.close()
        self.path.unlink(missing_ok=True)


def _fraction(text: str | None) -> float | None:
    if not text or "/" not in text:
        return None
    num, den = text.split("/", 1)
    try:
        return float(num) / float(den) if float(den) else None
    except ValueError:
        return None


def probe(path: Path) -> ProbeResult:
    """Validate a video with ffprobe. Raises UploadError(422) when it is not a decodable video."""
    exe = shutil.which("ffprobe")
    if exe is None:
        raise UploadError(500, "ffprobe is not installed")
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-print_format", "json", "-show_streams", "-show_format", str(path)],
            capture_output=True, text=True, timeout=PROBE_TIMEOUT_S, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise UploadError(422, "ffprobe timed out; the file may be corrupt") from exc
    if out.returncode != 0:
        raise UploadError(422, "not a readable video file")
    try:
        info = json.loads(out.stdout)
    except json.JSONDecodeError as exc:
        raise UploadError(422, "not a readable video file") from exc
    video = next((s for s in info.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None or not video.get("codec_name"):
        raise UploadError(422, "no video stream found")
    width, height = int(video.get("width") or 0), int(video.get("height") or 0)
    if width <= 0 or height <= 0:
        raise UploadError(422, "video has no frame size")
    fmt = info.get("format", {})
    try:
        duration = float(video.get("duration") or fmt.get("duration") or 0)
    except ValueError:
        duration = 0.0
    if duration <= 0:
        raise UploadError(422, "video has no duration")
    rotation = 0
    tag = (video.get("tags") or {}).get("rotate")
    side = next((d for d in video.get("side_data_list", []) if "rotation" in d), None)
    try:
        rotation = int(float(tag)) if tag is not None else int(float(side["rotation"])) if side else 0
    except (TypeError, ValueError):
        rotation = 0
    fps = _fraction(video.get("avg_frame_rate")) or _fraction(video.get("r_frame_rate"))
    created = (fmt.get("tags") or {}).get("creation_time") or (video.get("tags") or {}).get("creation_time")
    return ProbeResult(video["codec_name"], width, height, fps, rotation % 360, duration, created)


def transcode_to_h264(src: Path, dst: Path, timeout_s: int) -> None:
    """Convert an unusual codec (Indeo, MPEG-2, ...) to H.264 so every later stage can read it."""
    exe = shutil.which("ffmpeg")
    if exe is None:
        raise UploadError(500, "ffmpeg is not installed")
    tmp = dst.with_name(f"{dst.stem}.tmp{dst.suffix}")
    try:
        out = subprocess.run(
            [exe, "-v", "error", "-nostdin", "-i", str(src), "-an", "-c:v", "libx264", "-preset", "veryfast",
             "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-y", str(tmp)],
            capture_output=True, timeout=timeout_s, check=False,
        )
        if out.returncode != 0:
            raise UploadError(422, "this video's codec could not be converted")
        tmp.replace(dst)
    except subprocess.TimeoutExpired as exc:
        raise UploadError(422, "converting this video took too long") from exc
    finally:
        tmp.unlink(missing_ok=True)


def validate_rtsp_uri(uri: str) -> str:
    uri = uri.strip()
    parsed = urlparse(uri)
    if parsed.scheme != "rtsp" or not parsed.hostname or any(ch.isspace() for ch in uri):
        raise UploadError(422, "uri must be a valid rtsp:// address")
    return uri
