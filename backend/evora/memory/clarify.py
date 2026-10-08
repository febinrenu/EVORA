"""Clarify-once state machine: persist the paused query, turn the user's answer into a memory fact."""
from __future__ import annotations

import json
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

from contracts.models import CameraInfo, CameraOption, ClarifyRequest, ClarifyResponse, MemoryFact, QueryPlan, Referent, Zone

from evora.core.cameras import CameraNotFound, get_camera, list_cameras
from evora.core.db import Database
from evora.memory.kb import FactNotFound, KnowledgeBase, normalize
from evora.memory.resolve import Ambiguous, Resolution
from evora.memory.tod import TodError, parse_tod_range

_TOD = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")
_FILLER = {"the", "one", "camera", "cam", "it", "is", "that", "this", "a", "an", "at", "in", "of", "on"}


class ClarifyError(ValueError):
    """The answer could not be used. The pending question stays open so the UI can ask again."""


class QuestionClosed(ClarifyError):
    """The question was already answered, or never existed."""


@dataclass(frozen=True)
class Pending:
    query_id: str
    text: str
    plan: dict[str, Any]
    clarify: ClarifyRequest


@dataclass(frozen=True)
class ClarifyOutcome:
    fact: MemoryFact
    pending: Pending


def _question(ref: Referent, resolution: Resolution) -> tuple[str, str]:
    if isinstance(resolution, Ambiguous):
        return "choose_known", f"Which of these did you mean by {ref.text}?"
    if ref.role == "object":
        return "choose_track", f"Which one is {ref.text}? Pick it from the footage."
    if ref.role == "time":
        return "time_range", f"What hours do you mean by {ref.text}?"
    return "choose_camera", f"Which camera shows {ref.text}?"


def parse_camera_text(text: str, cameras: list[CameraInfo]) -> CameraInfo:
    """'camera 2', 'cam_02', 'the lobby one' -> one camera, or ClarifyError."""
    norm = normalize(text)
    by_id = {c.id: c for c in cameras}
    if norm.replace(" ", "_") in by_id:
        return by_id[norm.replace(" ", "_")]
    m = re.fullmatch(r"(?:camera|cam)\s*(\d+)", norm)
    if m:
        cid = f"cam_{int(m.group(1)):02d}"
        if cid not in by_id:
            raise ClarifyError(f"there is no camera {int(m.group(1))}")
        return by_id[cid]
    words = {w for w in norm.split() if w not in _FILLER}
    if not words:
        raise ClarifyError("please name a camera, for example 'camera 2' or 'the lobby one'")
    matches = []
    for cam in cameras:
        name = {w for w in normalize(cam.name).split() if w not in _FILLER}
        if name and (name <= words or words <= name):
            matches.append(cam)
    if len(matches) != 1:
        raise ClarifyError("I could not tell which camera you meant; please pick one from the list")
    return matches[0]


class Clarifier:
    def __init__(self, db: Database, kb: KnowledgeBase):
        self.db, self.kb = db, kb

    # --- ask ---
    def ask(
        self, query_id: str, text: str, plan: QueryPlan | dict[str, Any], ref: Referent, resolution: Resolution,
    ) -> ClarifyRequest:
        kind, question = _question(ref, resolution)
        options = [
            CameraOption(
                camera_id=c.id, camera_name=c.name,
                thumb_url=f"/api/cameras/{c.id}/frame?t={c.t0 + min(5.0, (c.duration_s or 10.0) / 2):.1f}",
            )
            for c in list_cameras(self.db)
        ] if kind == "choose_camera" else []
        req = ClarifyRequest(
            query_id=query_id, referent=ref, question=question, kind=kind, options=options,  # type: ignore[arg-type]
            known_candidates=[f.id for f in resolution.facts] if isinstance(resolution, Ambiguous) else [],
            allow_region=ref.role == "place",
        )
        plan_json = plan.model_dump_json() if isinstance(plan, QueryPlan) else json.dumps(plan)
        with self.db.write() as c:
            c.execute(
                "INSERT OR REPLACE INTO pending_queries(query_id,text,plan,clarify,created_at) VALUES(?,?,?,?,?)",
                (query_id, text, plan_json, req.model_dump_json(), time.time()),
            )
        return req

    def pending(self, query_id: str) -> Pending:
        with self.db.read() as c:
            row = c.execute("SELECT * FROM pending_queries WHERE query_id=?", (query_id,)).fetchone()
        if row is None:
            raise QuestionClosed("that question is no longer open")
        return Pending(row["query_id"], row["text"], json.loads(row["plan"]), ClarifyRequest.model_validate_json(row["clarify"]))

    # --- answer ---
    def apply(self, resp: ClarifyResponse) -> ClarifyOutcome:
        pending = self.pending(resp.query_id)
        ref = pending.clarify.referent
        if resp.fact_id:
            fact = self._bind_known(ref, resp.fact_id)
        elif ref.role == "place":
            fact = self._bind_place(ref, resp)
        elif ref.role == "object":
            fact = self._bind_object(ref, resp)
        else:
            fact = self._bind_time(ref, resp)
        with self.db.write() as c:
            c.execute("DELETE FROM pending_queries WHERE query_id=?", (resp.query_id,))
        return ClarifyOutcome(fact, pending)

    def _remember(self, ref: Referent, binding: dict[str, Any]) -> MemoryFact:
        existing = self.kb.find_exact(ref.role, ref.text)
        if existing:  # asked again about a known name: this is a correction
            return self.kb.supersede(existing[0].id, binding, "clarification")
        return self.kb.create(ref.role, normalize(ref.text), binding, "clarification")

    def _bind_known(self, ref: Referent, fact_id: str) -> MemoryFact:
        try:
            fact = self.kb.get(fact_id)
        except FactNotFound:
            raise ClarifyError("that known place no longer exists") from None
        if fact.superseded_by or fact.kind != ref.role:
            raise ClarifyError("that choice does not match what was asked")
        self.kb.add_alias(fact.id, ref.text)
        self.kb.touch(fact.id)
        return self.kb.get(fact.id)

    def _camera(self, camera_id: str) -> CameraInfo:
        try:
            return get_camera(self.db, camera_id)
        except CameraNotFound:
            raise ClarifyError(f"unknown camera {camera_id}") from None

    def _bind_place(self, ref: Referent, resp: ClarifyResponse) -> MemoryFact:
        if resp.camera_id:
            cam = self._camera(resp.camera_id)
        elif resp.text:
            cam = parse_camera_text(resp.text, list_cameras(self.db))
        else:
            raise ClarifyError("choose a camera for this place")
        binding: dict[str, Any] = {"camera_id": cam.id}
        zone_id = self._insert_zone(cam.id, resp.zone) if resp.zone else None
        if zone_id:
            binding["zone_id"] = zone_id
        fact = self._remember(ref, binding)
        if zone_id:
            with self.db.write() as c:
                c.execute("UPDATE zones SET fact_id=? WHERE id=?", (fact.id, zone_id))
        return fact

    def _insert_zone(self, camera_id: str, zone: Zone) -> str:
        if zone.camera_id != camera_id:
            raise ClarifyError("the drawn region belongs to a different camera")
        if any(not (0.0 <= v <= 1.0) for pt in zone.points for v in pt):
            raise ClarifyError("region points must be inside the frame")
        if zone.kind == "line" and len(zone.points) != 2:
            raise ClarifyError("a line needs exactly two points")
        if zone.kind == "polygon" and len(zone.points) < 3:
            raise ClarifyError("a polygon needs at least three points")
        zone_id = f"z_{uuid.uuid4().hex[:8]}"
        with self.db.write() as c:
            c.execute(
                "INSERT INTO zones(id,camera_id,kind,points,direction,created_at) VALUES(?,?,?,?,?,?)",
                (zone_id, camera_id, zone.kind, json.dumps(zone.points), zone.direction, time.time()),
            )
        return zone_id

    def _bind_object(self, ref: Referent, resp: ClarifyResponse) -> MemoryFact:
        if not resp.track_id:
            raise ClarifyError("pick the object from the footage")
        with self.db.read() as c:
            row = c.execute("SELECT global_id FROM tracks WHERE id=?", (resp.track_id,)).fetchone()
        if row is None:
            raise ClarifyError("unknown track")
        return self._remember(ref, {"track_id": resp.track_id, "global_id": row["global_id"]})

    def _bind_time(self, ref: Referent, resp: ClarifyResponse) -> MemoryFact:
        after, before = resp.tod_after, resp.tod_before
        if not after and not before and resp.text:  # typed: "8pm to 6am"
            try:
                after, before = parse_tod_range(resp.text)
            except TodError as exc:
                raise ClarifyError(str(exc)) from None
        if not after or not before or not _TOD.match(after) or not _TOD.match(before):
            raise ClarifyError("give the hours as HH:MM to HH:MM")
        return self._remember(ref, {"tod_after": after, "tod_before": before})
