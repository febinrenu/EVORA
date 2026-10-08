"""Evidence pack: clip, frames, provenance and a SHA-256 manifest that anyone can verify offline.

`python -m evora.evidence.pack verify <pack.zip>` checks a pack without the application.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from evora import __version__
from evora.core import cameras as cams
from evora.core.db import Database
from evora.core.media_service import MediaError, MediaService
from evora.core.workspace import Workspace
from evora.evidence import audit, signing
from evora.evidence import store as evidence_store
from evora.query.planner import workspace_tz

ZIP_TIME = (1980, 1, 1, 0, 0, 0)  # fixed, so identical content always gives identical bytes
SUMS_NAME, MANIFEST_NAME = "SHA256SUMS", "manifest.json"
SIG_NAME, PUB_NAME = "manifest.sig", "signer.pub"
_SUM_LINE = re.compile(r"^([0-9a-f]{64})  (\S+)$")
README = (
    "evora evidence pack\n"
    "Verify the files: unzip this archive, then run `sha256sum -c SHA256SUMS` in the folder.\n"
    "manifest.json lists the same hashes plus the hashes of the original source video, which is not included.\n"
    "manifest.sig is an Ed25519 signature of manifest.json made with the key in signer.pub.\n"
    "Check everything, signature included: python -m evora.evidence.pack verify <this zip> [--fingerprint <hex>]\n"
    "A valid signature shows the manifest was not changed after export. To know the pack came from the installation you\n"
    "trust, compare the signer fingerprint (in manifest.json) with the one that installation published.\n"
)


class PackRefused(Exception):
    """Exporting is not allowed in the current privacy state."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


@dataclass
class PackResult:
    path: Path
    sha256: str
    files: dict[str, dict[str, Any]]
    faces_blurred: bool
    context: dict[str, Any] = field(default_factory=dict)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stored_sha256(db: Database, camera_id: str, path: Path) -> str:
    """Hash of the file that was actually decoded; cached by size and modification time."""
    stat = path.stat()
    key = f"sha.{camera_id}"
    cached = db.get_meta(key)
    if cached:
        entry = json.loads(cached)
        if entry.get("size") == stat.st_size and entry.get("mtime_ns") == stat.st_mtime_ns:
            return entry["sha256"]
    digest = sha256_file(path)
    db.set_meta(key, json.dumps({"size": stat.st_size, "mtime_ns": stat.st_mtime_ns, "sha256": digest}))
    return digest


def _context(db: Database, evidence_id: str) -> dict[str, Any]:
    """Which question or watch produced this evidence, from stored answers and alerts."""
    with db.read() as c:
        for row in c.execute("SELECT id, text, plan, answer, timings, created_at FROM query_log ORDER BY created_at DESC"):
            if row["answer"] and evidence_id in row["answer"]:
                return {
                    "kind": "query", "query_id": row["id"], "question": row["text"], "asked_at": row["created_at"],
                    "plan": json.loads(row["plan"] or "null"), "timings_ms": json.loads(row["timings"] or "null"),
                }
        for row in c.execute("SELECT a.id, a.sq_id, a.evidence, s.text, s.rule FROM alerts a LEFT JOIN standing_queries s "
                             "ON s.id = a.sq_id"):
            if evidence_id in (row["evidence"] or ""):
                return {
                    "kind": "alert", "alert_id": row["id"],
                    "standing_query": {"id": row["sq_id"], "text": row["text"], "rule": json.loads(row["rule"] or "null")},
                }
    return {"kind": "unknown"}


def _times(db: Database, t: float) -> dict[str, Any]:
    tz = workspace_tz(db)
    return {
        "epoch_s": t, "iso_utc": datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds"),
        "iso_local": datetime.fromtimestamp(t, tz).isoformat(timespec="milliseconds"),
    }


def _transcoded_from(db: Database, source_sha256: str | None) -> dict[str, Any] | None:
    if not source_sha256:
        return None
    for entry in audit.entries(db, "transcode"):
        if entry["detail"].get("original_sha256") == source_sha256:
            return {"codec": entry["detail"].get("from"), "original_sha256": source_sha256}
    return None


def _json(obj: Any) -> bytes:
    return (json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _live_source(media: MediaService, cam: Any, rec: Any) -> dict[str, Any]:
    """Provenance of footage from a real camera: the hashes of the recording segments the clip was cut from.

    The camera's address is never written here (it may carry a password).
    """
    segments = [
        {"name": s.path.name, "sha256": sha256_file(s.path), "size": s.path.stat().st_size,
         "start_epoch_s": round(s.start, 3), "end_epoch_s": round(s.end, 3)}
        for s in media.live_segments(cam, rec) if s.path.is_file()
    ]
    note = (
        "A live camera: the clip was cut from the rolling recording segments listed here. The segments are not part of this "
        "pack and are deleted after the buffer window, so compare these hashes while they still exist."
        if segments else
        "A live camera: the clip was cut from the rolling recording while it existed; those segments have since been deleted."
    )
    return {"kind": "live_recording", "camera_name": cam.name, "segments": segments, "note": note}


def build_pack(
    db: Database, ws: Workspace, media: MediaService, evidence_id: str, *, blur_setting: bool,
    unblur_token_valid: bool, unblur_reason: str | None = None, now: float | None = None,
) -> PackResult:
    rec = evidence_store.get(db, evidence_id)  # raises EvidenceError / EvidenceNotFound
    cam = cams.get_camera(db, rec.camera_id)
    live = media.has_live_recording(cam)  # a real camera: the evidence comes from its rolling recording
    source = None if live else media.source_of(cam)  # raises MediaError (RTSP without a recording, missing file)

    # privacy: blurred, or not exported
    if unblur_token_valid:
        want_blur, why_unblurred = False, f"unblur token: {unblur_reason or 'no reason recorded'}"
    elif not blur_setting:
        want_blur, why_unblurred = False, "face blur is switched off"
    else:
        want_blur, why_unblurred = True, None
        if media.blur_status(True) != "applied":
            raise PackRefused(
                "Face blur is unavailable, so this evidence cannot be exported with faces hidden. "
                "Install the face model, or request an unblur token with a reason."
            )

    clip_path, clip_status = media.clip(cam, rec, want_blur)
    frames: list[tuple[str, bytes]] = []
    for name, t, box in (("frame_1_start.jpg", rec.t_start, None), ("frame_2_peak.jpg", rec.t_peak, rec.bbox),
                         ("frame_3_end.jpg", rec.t_end, None)):
        data, status = media.frame(cam, t, want_blur, box)
        if want_blur and status != "applied":
            raise PackRefused("Face blur failed while rendering the frames; nothing was exported.")
        frames.append((name, data))
    if want_blur and clip_status != "applied":
        raise PackRefused("Face blur failed while rendering the clip; nothing was exported.")
    faces_blurred = want_blur

    full = evidence_store.find_evidence(db, evidence_id)
    context = _context(db, evidence_id)
    evidence_doc = {
        "evidence_id": evidence_id,
        "evidence": full.model_dump(mode="json") if full else None,
        "camera": {"id": cam.id, "name": cam.name, "clock_source": cam.t0_source, "clock_t0": _times(db, cam.t0)},
        "times": {"start": _times(db, rec.t_start), "peak": _times(db, rec.t_peak), "end": _times(db, rec.t_end),
                  "offset_in_file_s": None if live else max(rec.t_peak - cam.t0, 0.0)},
        "bbox_normalized": list(rec.bbox) if rec.bbox else None,
        "produced_by": context,
    }

    members: dict[str, bytes] = {"clip.mp4": clip_path.read_bytes()}
    members.update(dict(frames))
    members["evidence.json"] = _json(evidence_doc)
    members["README.txt"] = README.encode("utf-8")

    generated = now if now is not None else time.time()
    upload_sha = cams.source_sha256(db, cam.id)
    signer = signing.load_or_create(ws.root)
    manifest = {
        "format": "evora-evidence-pack/1", "software_version": __version__, "workspace": ws.slug,
        "generated_at": _times(db, generated), "evidence_id": evidence_id, "faces_blurred": faces_blurred,
        "unblurred_because": why_unblurred,
        "source": _live_source(media, cam, rec) if source is None else {
            "file_name": Path(cam.source_uri).name, "source_sha256": upload_sha,
            "stored_sha256": stored_sha256(db, cam.id, source), "transcoded_from": _transcoded_from(db, upload_sha),
            "fps": cam.fps, "width": cam.width, "height": cam.height, "duration_s": cam.duration_s,
            "note": "The original video is not part of this pack; compare these hashes with the file you hold.",
        },
        "signer": {"algorithm": signing.ALGORITHM, "fingerprint": signer.fingerprint},
        "files": {n: {"sha256": sha256_bytes(b), "size": len(b)} for n, b in sorted(members.items())},
    }
    members[MANIFEST_NAME] = _json(manifest)
    members[SIG_NAME] = signer.sign(members[MANIFEST_NAME])
    members[PUB_NAME] = signer.public_pem
    sums = "".join(f"{sha256_bytes(b)}  {n}\n" for n, b in sorted(members.items()))
    members[SUMS_NAME] = sums.encode("ascii")

    out_dir = ws.root / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{evidence_id}_{'blur' if faces_blurred else 'raw'}.zip"
    tmp = out.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w") as zf:
        for name in sorted(members):
            info = zipfile.ZipInfo(name, ZIP_TIME)
            info.compress_type = zipfile.ZIP_STORED if name.endswith((".mp4", ".jpg")) else zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, members[name])
    os.replace(tmp, out)
    result = PackResult(out, sha256_file(out), manifest["files"], faces_blurred, context)
    audit.record(db, "export", {
        "evidence_id": evidence_id, "files": {n: v["sha256"] for n, v in manifest["files"].items()},
        "pack_sha256": result.sha256, "faces_blurred": faces_blurred, "signer_fingerprint": signer.fingerprint,
        "unblurred_because": why_unblurred,
    })
    return result


def inspect_signature(source: Path | bytes) -> tuple[str, str | None]:
    """("valid" | "unsigned" | "invalid", signer fingerprint) for a pack's manifest signature."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source)
    except (zipfile.BadZipFile, OSError):
        return "invalid", None
    with zf:
        names = set(zf.namelist())
        if not {SIG_NAME, PUB_NAME} & names:
            return "unsigned", None
        if not {SIG_NAME, PUB_NAME, MANIFEST_NAME} <= names:
            return "invalid", None
        public = zf.read(PUB_NAME)
        try:
            fingerprint = signing.fingerprint_of(public)
        except signing.SigningError:
            return "invalid", None
        ok = signing.verify(public, zf.read(MANIFEST_NAME), zf.read(SIG_NAME))
    return ("valid" if ok else "invalid"), fingerprint


def verify_pack(source: Path | bytes, fingerprint: str | None = None) -> list[str]:
    """Problems found in a pack; an empty list means every file matches its recorded hash and the signature holds.

    An unsigned (older) pack is not a problem by itself, unless `fingerprint` says whose signature is required.
    """
    problems: list[str] = []
    try:
        zf = zipfile.ZipFile(io.BytesIO(source) if isinstance(source, bytes) else source)
    except (zipfile.BadZipFile, OSError) as exc:
        return [f"not a readable zip file: {exc}"]
    with zf:
        names = [i.filename for i in zf.infolist()]
        if len(set(names)) != len(names):
            problems.append("duplicate entries in the archive")
        for n in names:
            if n.startswith(("/", "\\")) or ".." in n.split("/") or "/" in n or "\\" in n or ":" in n:
                problems.append(f"unsafe path in archive: {n!r}")
        safe = [n for n in names if n == os.path.basename(n) and ".." not in n]
        if SUMS_NAME not in safe:
            return [*problems, f"{SUMS_NAME} is missing"]
        listed: dict[str, str] = {}
        for line in zf.read(SUMS_NAME).decode("ascii", "replace").splitlines():
            m = _SUM_LINE.match(line)
            if not m:
                problems.append(f"malformed line in {SUMS_NAME}: {line[:60]!r}")
                continue
            listed[m.group(2)] = m.group(1)
        for n in safe:
            if n != SUMS_NAME and n not in listed:
                problems.append(f"{n} is in the archive but not listed in {SUMS_NAME}")
        for n, digest in listed.items():
            if n not in safe:
                problems.append(f"{n} is listed in {SUMS_NAME} but missing from the archive")
            elif sha256_bytes(zf.read(n)) != digest:
                problems.append(f"{n} does not match its recorded SHA-256 (changed after export)")
        if MANIFEST_NAME in safe:
            try:
                manifest = json.loads(zf.read(MANIFEST_NAME))
            except json.JSONDecodeError:
                problems.append(f"{MANIFEST_NAME} is not valid JSON")
            else:
                for n, meta in manifest.get("files", {}).items():
                    if n not in safe:
                        problems.append(f"{n} is described in {MANIFEST_NAME} but missing from the archive")
                    elif sha256_bytes(zf.read(n)) != meta.get("sha256"):
                        problems.append(f"{n} does not match {MANIFEST_NAME}")
        else:
            problems.append(f"{MANIFEST_NAME} is missing")
    state, signer = inspect_signature(source)
    if state == "invalid":
        problems.append(f"{SIG_NAME} does not match {MANIFEST_NAME} (the manifest was changed, or the signature is damaged)")
    if fingerprint is not None:
        if state == "unsigned":
            problems.append(f"this pack is not signed, so it cannot be tied to signer {fingerprint}")
        elif state == "valid" and signer != fingerprint.lower():
            problems.append(f"signed by {signer}, not by the expected {fingerprint}")
    return problems


def _main(argv: list[str]) -> int:
    usage = "usage: python -m evora.evidence.pack verify <pack.zip> [--fingerprint <hex>]"
    args = argv[1:]
    fingerprint: str | None = None
    if "--fingerprint" in args:
        k = args.index("--fingerprint")
        if k + 1 >= len(args):
            print(usage)
            return 2
        fingerprint = args[k + 1].strip().lower()
        del args[k:k + 2]
    if len(args) != 2 or args[0] != "verify":
        print(usage)
        return 2
    problems = verify_pack(Path(args[1]), fingerprint)
    if problems:
        print("PACK DOES NOT VERIFY")
        for p in problems:
            print(" -", p)
        return 1
    state, signer = inspect_signature(Path(args[1]))
    print("OK: every file matches its SHA-256")
    if state == "valid":
        print(f"OK: the manifest is signed by {signer}" + (" (as expected)" if fingerprint else
              "; compare this fingerprint with the one the operator published to know where the pack came from"))
    else:
        print("NOTE: this pack is not signed (exported by an older version): only its file hashes were checked")
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))


__all__ = ["MediaError", "PackRefused", "PackResult", "build_pack", "sha256_file", "verify_pack"]
