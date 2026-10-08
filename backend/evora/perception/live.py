"""Live ingest from an RTSP stream (PLAN P2.17): tracks, crops and events as they happen.

    live_ingest(cam, profile, stop, on_event)     runs until `stop` is set

A grabber thread reads the stream and keeps only the newest frame, so a slow detector drops frames
instead of falling behind; it reconnects with backoff when the stream breaks. The processing loop uses
the same motion-gated sampler, detector, tracker and crop bookkeeping as file ingest.

* Times are wall-clock epoch seconds at the moment a frame arrives.
* A track gets its `tracks` row as soon as it is confirmed, so the alert engine can look up its class and
  trajectory when an event arrives; rows and points are refreshed twice a second.
* Events (appear, cross_line, enter_zone, exit_zone, dwell, disappear) are written to the `events` table
  and passed to `on_event` as dicts shaped exactly like `events` rows, once each, as soon as they occur.
* A track unseen for `track_lost_s` is finalised: best crops are embedded and saved like in file ingest.
Attributes (colour, carrying) and ReID features are not computed for live tracks yet.
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from collections.abc import Callable, Iterator

from contracts.models import CameraInfo, Zone

from evora.core.config import load_config
from evora.core.db import Database, open_db
from evora.core.vectors import dims_from_meta, ensure_tables, open_store
from evora.core.workspace import Workspace
from evora.perception.crops import ActiveTrack, TrackBook
from evora.perception.decode import Frame
from evora.perception.detect import load_detector
from evora.perception.embed import get_embedder
from evora.perception.events import Sample, foot, track_events, zones_of
from evora.perception.locks import STORE_SETUP
from evora.perception.motion import AdaptiveSampler
from evora.perception.pipeline import _persist, _RowBuffer, resolve_workspace
from evora.perception.settings import IngestSettings, load_settings
from evora.perception.track import FrameTracker

log = logging.getLogger("evora.perception.live")

EventFn = Callable[[dict], None]
FrameSource = Callable[[threading.Event], Iterator[tuple[float, Frame]]]
HOUSEKEEPING_S = 0.5
ZONE_REFRESH_S = 5.0
BACKOFF_START_S, BACKOFF_MAX_S = 1.0, 10.0
RTSP_OPTIONS = {"rtsp_transport": "tcp", "fflags": "nobuffer", "flags": "low_delay", "max_delay": "500000", "stimeout": "5000000"}
_TRACK_SEQ = re.compile(r":t(\d+)$")


class FrameSlot:
    """A mailbox for the newest frame: putting a new one replaces an unread old one."""

    def __init__(self) -> None:
        self._item: tuple[float, Frame] | None = None
        self._cond = threading.Condition()
        self.dropped = 0

    def put(self, item: tuple[float, Frame]) -> None:
        with self._cond:
            if self._item is not None:
                self.dropped += 1
            self._item = item
            self._cond.notify()

    def take(self, timeout: float) -> tuple[float, Frame] | None:
        with self._cond:
            if self._item is None:
                self._cond.wait(timeout)
            item, self._item = self._item, None
            return item


def rtsp_frames(url: str, stop: threading.Event, max_width: int) -> Iterator[tuple[float, Frame]]:
    """Frames from an RTSP stream as (arrival time, lazily converted Frame). Raises when the stream ends or breaks."""
    import av

    container = av.open(url, options=RTSP_OPTIONS, timeout=(5.0, 5.0))
    try:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        index = 0
        for av_frame in container.decode(stream):
            if stop.is_set():
                return
            yield time.time(), Frame(index, 0.0, av_frame, 0, max_width)
            index += 1
    finally:
        container.close()


def _grab(source: FrameSource, slot: FrameSlot, stop: threading.Event, state: Callable[[str], None]) -> None:
    """Fill the slot until stopped, reconnecting with exponential backoff."""
    delay = BACKOFF_START_S
    while not stop.is_set():
        try:
            state("running")
            for item in source(stop):
                slot.put(item)
                delay = BACKOFF_START_S
            if not stop.is_set():
                raise ConnectionError("stream ended")
        except Exception as exc:  # noqa: BLE001 - any stream failure means: wait and reconnect
            if stop.is_set():
                return
            log.warning("stream interrupted (%s); retrying in %.0f s", exc, delay)
            state("retrying")
            if stop.wait(delay):
                return
            delay = min(delay * 2, BACKOFF_MAX_S)


def _first_seq(db: Database, camera_id: str) -> int:
    with db.read() as c:
        ids = [r[0] for r in c.execute("SELECT id FROM tracks WHERE camera_id=?", (camera_id,))]
    seqs = [int(m.group(1)) for i in ids if (m := _TRACK_SEQ.search(i))]
    return max(seqs, default=0) + 1


class _Session:
    """Everything a live camera keeps between frames: stored rows, emitted events, synced points."""

    def __init__(self, cam: CameraInfo, ws: Workspace, db: Database, st: IngestSettings, on_event: EventFn, t_base: float):
        self.cam, self.ws, self.db, self.st, self.on_event, self.t_base = cam, ws, db, st, on_event, t_base
        self.emitted: set[str] = set()
        self.synced: dict[int, int] = {}          # seq -> number of points already written
        self.zones: list[Zone] = []
        self.zones_at = -1e9
        self.dropped_logged = 0

    def track_id(self, seq: int) -> str:
        return f"{self.cam.id}:t{seq:06d}"

    def emit(self, row: tuple) -> None:
        eid, camera_id, track_id, kind, zone_id, t, payload = row
        if eid in self.emitted:
            return
        self.emitted.add(eid)
        with self.db.write() as c:
            c.execute("INSERT OR REPLACE INTO events(id,camera_id,track_id,kind,zone_id,t,payload) VALUES(?,?,?,?,?,?,?)", row)
        try:
            self.on_event({"id": eid, "camera_id": camera_id, "track_id": track_id, "kind": kind, "zone_id": zone_id,
                           "t": t, "payload": payload})
        except Exception:  # noqa: BLE001 - a failing alert handler must not stop ingestion
            log.exception("on_event failed for %s", eid)

    def refresh_zones(self, now: float) -> None:
        if now - self.zones_at >= ZONE_REFRESH_S:
            self.zones, self.zones_at = zones_of(self.db, self.cam.id), now

    def sync_active(self, active: list[ActiveTrack]) -> None:
        """Keep a tracks row and its points current for every confirmed track, and emit its new events."""
        for tr in active:
            if tr.n_obs < self.st.min_track_obs or not tr.points:
                continue
            tid = self.track_id(tr.seq)
            t0 = self.t_base
            with self.db.write() as c:
                c.execute(
                    "INSERT INTO tracks(id,camera_id,cls,cls_conf,t_start,t_end,n_obs,attrs) VALUES(?,?,?,?,?,?,?,'{}') "
                    "ON CONFLICT(id) DO UPDATE SET cls=excluded.cls, cls_conf=excluded.cls_conf, t_end=excluded.t_end, "
                    "n_obs=excluded.n_obs",
                    (tid, self.cam.id, tr.cls, tr.cls_conf, t0 + tr.t_start, t0 + tr.t_end, tr.n_obs))
                done = self.synced.get(tr.seq, 0)
                c.executemany("INSERT INTO track_points(track_id,t,x1,y1,x2,y2,conf) VALUES(?,?,?,?,?,?,?)",
                              [(tid, t0 + p[0], p[1], p[2], p[3], p[4], p[5]) for p in tr.points[done:]])
                self.synced[tr.seq] = len(tr.points)
            x, y = foot(*tr.points[0][1:5])
            self.emit((f"{tid}:appear:-:0", self.cam.id, tid, "appear", None, t0 + tr.t_start,
                       json.dumps({"x": round(x, 4), "y": round(y, 4)})))
            samples = [Sample(t0 + p[0], *foot(*p[1:5])) for p in tr.points]
            for row in track_events(tid, self.cam.id, samples, self.zones, t_end=t0 + tr.t_end, dwell_s=self.st.dwell_s,
                                    hysteresis=self.st.line_hysteresis, debounce=self.st.zone_debounce):
                self.emit(row)

    def finalise(self, tracks: list, still_active: set[int], buffer: _RowBuffer, embedder) -> None:
        """Replace the provisional rows of finished tracks by the full ones (crops, embeddings, direction).

        A track that was stored but then dropped as noise (no usable crop) loses its provisional rows too.
        """
        for seq in [s for s in self.synced if s not in still_active and s not in {t.seq for t in tracks}]:
            tid = self.track_id(seq)
            with self.db.write() as c:
                c.execute("DELETE FROM track_points WHERE track_id=?", (tid,))
                c.execute("DELETE FROM tracks WHERE id=?", (tid,))
            self.synced.pop(seq, None)
        if not tracks:
            return
        for trk in tracks:
            tid = self.track_id(trk.seq)
            with self.db.write() as c:
                c.execute("DELETE FROM track_points WHERE track_id=?", (tid,))
                c.execute("DELETE FROM tracks WHERE id=?", (tid,))
            self.synced.pop(trk.seq, None)
        _persist(self.db, buffer, self.ws, self.cam.model_copy(update={"t0": self.t_base}), tracks, self.st, embedder)
        buffer.flush()
        for trk in tracks:      # only now: the full track row exists when the alert engine looks it up
            if trk.points:
                tid = self.track_id(trk.seq)
                x, y = foot(*trk.points[-1][1:5])
                self.emit((f"{tid}:disappear:-:0", self.cam.id, tid, "disappear", None, self.t_base + trk.t_end,
                           json.dumps({"x": round(x, 4), "y": round(y, 4)})))


def live_ingest(
    cam: CameraInfo, profile: str, stop: threading.Event, on_event: EventFn, *,
    ws: Workspace | None = None, settings: IngestSettings | None = None, frame_source: FrameSource | None = None,
    on_state: Callable[[str], None] | None = None,
) -> None:
    """Ingest a live camera until `stop` is set (PLAN.md section 5.6).

    `on_state` (optional) is told "running" when frames are flowing and "retrying" when the stream broke.
    `frame_source` replaces the RTSP reader, which is how the tests feed frames.
    """
    cfg = load_config(profile)
    st = settings or load_settings(cfg)
    ws = ws or resolve_workspace()
    db = open_db(ws.db_path)
    embedder = get_embedder(st)
    store = open_store(ws.vectors_dir)
    with STORE_SETUP:
        db.set_meta("embed_dim_image", str(embedder.dim))
        ensure_tables(store, dims_from_meta(db), only={"crops", "scenes"})
    if frame_source is None:
        url = cam.source_uri
        frame_source = lambda s: rtsp_frames(url, s, st.max_width)  # noqa: E731
    state = on_state or (lambda _s: None)

    det = load_detector(st)
    tracker, sampler = FrameTracker(det, st), AdaptiveSampler(st)
    t_base = time.time()
    book = TrackBook(st, first_seq=_first_seq(db, cam.id))
    session = _Session(cam, ws, db, st, on_event, t_base)
    buffer = _RowBuffer(store.open_table("crops"), limit=1)
    slot = FrameSlot()
    grabber = threading.Thread(target=_grab, args=(frame_source, slot, stop, state), name=f"grab-{cam.id}", daemon=True)
    grabber.start()
    log.info("%s live ingest started from %s", cam.id, cam.source_uri)
    next_housekeeping = 0.0
    try:
        while not stop.is_set():
            item = slot.take(timeout=HOUSEKEEPING_S)
            if item is not None:
                t_wall, frame = item
                rel = t_wall - t_base
                if sampler.should_process(rel, frame):
                    book.observe(rel, tracker.update(frame.image), frame.image)
            now = time.time()
            if now >= next_housekeeping:
                next_housekeeping = now + HOUSEKEEPING_S
                session.refresh_zones(now)
                session.sync_active(book.active())
                finished = book.finalize_stale(now - t_base)
                session.finalise(finished, {t.seq for t in book.active()}, buffer, embedder)
                if slot.dropped - session.dropped_logged >= 100:
                    session.dropped_logged = slot.dropped
                    log.info("%s: %d frames dropped so far (detector slower than the stream)", cam.id, slot.dropped)
    finally:
        stop.set()
        grabber.join(timeout=5.0)
        session.sync_active(book.active())
        session.finalise(book.finalize_all(), set(), buffer, embedder)
        log.info("%s live ingest stopped", cam.id)
