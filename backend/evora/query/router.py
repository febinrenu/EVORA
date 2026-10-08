"""The query router: plan -> referents -> retrieval -> logic -> answer, as a stream of events.

Event order (contracts/api.md): `plan`, then either `clarify` (and stop) or `evidence`*,
`answer`, `verified`*, `done`. `answer` is sent before verification so the UI never waits
on the slow part. Memory and verification are injected so this module owns the order of
steps and nothing else:

  resolver   M1's memory.resolve: Bound | Ambiguous | Unknown for a place, object or time
  clarifier  M1's memory.clarify: builds and persists the one clarification, then resumes
  verifier   optional second look at the top evidence (P3.11); streams `verified`
"""
from __future__ import annotations

import inspect
import json
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, tzinfo
from pathlib import Path
from typing import Any, Literal, Protocol

from contracts.models import (
    Answer,
    CameraOption,
    ClarifyRequest,
    ClarifyResponse,
    Evidence,
    MemoryFact,
    PathHop,
    QueryPlan,
    Referent,
    StreamEvent,
    TimeWindow,
    Zone,
)

from evora.core.db import Database
from evora.evidence.store import EvidenceError, register
from evora.query.compose import Sentence, compose_checked, make_notes, stamp, validate
from evora.query.describe import deterministic_sentences, gather_facts, narrate
from evora.query.logic import (
    Candidate,
    EventRec,
    Match,
    apply_action,
    best_per_track,
    count_distinct,
    order_matches,
)
from evora.query.planner import Planner, PlanningError, reference_now, workspace_tz
from evora.query.retrieve import Retriever, SearchScope

log = logging.getLogger("evora.query.router")

TRACK_PAD_S = 5.0     # evidence window around the best frame of a track
EVENT_PAD_S = 1.5     # evidence window around a crossing / entry / dwell
MAX_DESCRIBE_EVIDENCE = 8
EVENT_CHUNK = 400     # SQLite parameter budget when loading events


# ----------------------------------------------------------------- protocols
class Resolution(Protocol):
    status: Literal["bound", "ambiguous", "unknown"]
    facts: Sequence[MemoryFact]


class Resolver(Protocol):
    def resolve(self, ref: Referent) -> Resolution: ...


class Clarifier(Protocol):
    def ask(self, query_id: str, text: str, plan: QueryPlan, ref: Referent, resolution: Resolution,
            options: list[CameraOption]) -> ClarifyRequest: ...

    def resume(self, resp: ClarifyResponse) -> tuple[str, QueryPlan] | None: ...


class Verifier(Protocol):
    def verify(self, plan: QueryPlan, evidence: Sequence[Evidence]) -> AsyncIterator[tuple[str, bool | None]]: ...


@dataclass(frozen=True)
class RouterConfig:
    accept: float = 0.40          # a match scoring below this is a near miss, not an answer (calibrated on dev)
    verify: str = "filter"        # "filter": the visual check can set candidates aside; "annotate": flags only
    thumb_fmt: str = "/api/media/thumb/{id}.jpg"
    clip_fmt: str = "/api/media/clip/{id}.mp4"


@dataclass
class _Camera:
    id: str
    name: str
    t0: float
    layers: list[str]
    ir_fraction: float | None
    source_name: str | None
    duration_s: float | None = None


@dataclass
class _Bound:
    """What the remembered facts add to a plan."""

    camera_ids: set[str] = field(default_factory=set)
    zones: dict[str, Zone] = field(default_factory=dict)
    global_ids: set[str] = field(default_factory=set)
    window: TimeWindow | None = None
    notes: list[str] = field(default_factory=list)


def _event(kind: str, data: dict[str, Any]) -> StreamEvent:
    return StreamEvent(type=kind, data=data)  # type: ignore[arg-type]


class Router:
    def __init__(
        self,
        db: Database,
        planner: Planner,
        retriever: Retriever,
        resolver: Resolver,
        clarifier: Clarifier,
        verifier: Verifier | None = None,
        cfg: RouterConfig | None = None,
        reference_override=lambda: None,  # noqa: B008 - callable returning settings.reference_now or None
        gateway: Any = None,
        path_for: Callable[[str], list[PathHop]] | None = None,
        file_offset: Callable[[str, float | None, float], float | None] | None = None,
    ) -> None:
        # (camera id, camera duration, wall-clock time) -> seconds into the recorded file, or None to use t - t0.
        # Replay-as-live footage loops, so its position in the file is not simply t - t0.
        self._file_offset = file_offset
        self._gateway = gateway  # phrases `describe` answers; without it they are plain deterministic summaries
        self._path_for = path_for or self._reid_path
        self._db, self._planner, self._retriever = db, planner, retriever
        self._resolver, self._clarifier, self._verifier = resolver, clarifier, verifier
        self.cfg = cfg or RouterConfig()
        self._reference_override = reference_override

    # ------------------------------------------------------------------ public
    async def answer(self, text: str, session_id: str) -> AsyncIterator[StreamEvent]:
        query_id = f"q_{uuid.uuid4().hex[:10]}"
        started = time.monotonic()
        cameras = self._cameras()
        ref_now, tz = self._clock()
        try:
            planned = await self._planner.plan(text, cameras, ref_now, tz)
        except PlanningError as exc:
            yield _event("error", {"message": str(exc), "query_id": query_id})
            yield _event("done", {"query_id": query_id})
            return
        timings = {**planned.timings_ms, "plan_total": _ms(started)}
        async for ev in self._run(query_id, text, planned.plan, list(planned.notes), timings, started, cameras, tz,
                                  ref_now):
            yield ev

    async def resume(self, resp: ClarifyResponse) -> AsyncIterator[StreamEvent]:
        started = time.monotonic()
        pending = self._clarifier.resume(resp)
        if pending is None:
            yield _event("error", {"message": "That question is no longer waiting for an answer; ask it again.",
                                   "query_id": resp.query_id})
            yield _event("done", {"query_id": resp.query_id})
            return
        text, plan = pending
        cameras = self._cameras()
        ref_now, tz = self._clock()
        async for ev in self._run(resp.query_id, text, plan, [], {}, started, cameras, tz, ref_now):
            yield ev

    # ---------------------------------------------------------------- pipeline
    async def _run(self, query_id: str, text: str, plan: QueryPlan, notes: list[str], timings: dict[str, float],
                   started: float, cameras: list[_Camera], tz: tzinfo, ref_now: float) -> AsyncIterator[StreamEvent]:
        yield _event("plan", plan.model_dump(mode="json"))

        # 1. referents: ask once for the first one memory cannot place, otherwise bind them all
        bound = _Bound()
        for ref in plan.unresolved:
            resolution = self._resolver.resolve(ref)
            if inspect.isawaitable(resolution):  # M1's resolver is async (its equivalence step calls the gateway)
                resolution = await resolution
            if resolution.status == "bound":
                self._apply_fact(resolution.facts[0], ref, bound)
                continue
            options = [CameraOption(camera_id=c.id, camera_name=c.name,
                                    thumb_url=f"/api/cameras/{c.id}/frame?t={c.t0}") for c in cameras]
            request = self._clarifier.ask(query_id, text, plan, ref, resolution, options)
            yield _event("clarify", request.model_dump(mode="json"))
            return
        notes += bound.notes

        if plan.intent == "standing":
            yield _event("note", {"text": "Watch requests are created from the Watch panel, not the question bar."})
            yield _event("done", {"query_id": query_id})
            return

        plan = self._with_bound_time(plan, bound)

        if plan.intent == "describe":
            async for ev in self._describe(query_id, text, plan, notes, timings, started, cameras, tz, ref_now, bound):
                yield ev
            return

        # 2. retrieval
        t = time.monotonic()
        scope_cams = frozenset(set(plan.camera_ids) | bound.camera_ids)
        found = await self._retriever.search(plan, SearchScope(scope_cams, plan.time, tz))
        timings["retrieve"] = _ms(t)
        notes += [n for n in found.notes if n not in notes]

        # 3. logic: does each candidate actually satisfy the action, where and when
        t = time.monotonic()
        cands = found.candidates
        if bound.global_ids:
            cands = [c for c in cands if c.track.global_id in bound.global_ids]
        matches, dropped = self._matches(plan, cands, bound, tz)
        if dropped:
            notes.append("Coarse scene matches were left out because they cannot show the requested action.")
        accepted = [m for m in matches if m.score >= self.cfg.accept]
        near = [m for m in matches if m.score < self.cfg.accept]
        timings["logic"] = _ms(t)

        # 4. evidence and answer
        t = time.monotonic()
        camera_by_id = {c.id: c for c in cameras}
        hops: list[PathHop] = []
        if plan.intent == "path" and accepted:
            evidence, hops = self._path_evidence(query_id, max(accepted, key=lambda m: m.score), camera_by_id, notes)
        else:
            shown = self._select(plan, accepted)
            evidence = [self._evidence(m, camera_by_id, f"{query_id}_{i}") for i, m in enumerate(shown, start=1)]
        miss = None
        if not accepted and near:
            best = max(near, key=lambda m: m.score)
            miss = self._evidence(best, camera_by_id, f"{query_id}_miss")
        for ev in evidence + ([miss] if miss else []):
            self._register(ev)
        for ev in evidence:
            yield _event("evidence", ev.model_dump(mode="json"))

        count = count_distinct(accepted) if plan.intent == "count" else None
        composed = compose_checked(
            plan, evidence, count=count, nearest_miss=miss, path=hops, cameras=_as_compose_cameras(cameras),
            source_names={c.id: c.source_name for c in cameras if c.source_name}, tz=tz, reference_now=ref_now,
            partial=_is_partial(cameras, plan, evidence),
        )
        timings["compose"] = _ms(t)
        timings["ttfa"] = _ms(started)
        all_notes = list(dict.fromkeys([*notes, *composed.notes]))
        answer = Answer(
            query_id=query_id, text=composed.text, verdict=composed.verdict, count=composed.count,
            evidence=evidence, path=hops, nearest_miss=miss, confidence=_confidence(evidence, miss),
            plan=plan, timings_ms=timings, notes=all_notes,
        )
        yield _event("answer", answer.model_dump(mode="json"))

        # 5. verification: a second, independent look at the top evidence. The first answer above is already out
        # (time to first answer); if the look sets candidates aside, a revised answer follows (time to verified answer).
        if self._verifier is not None and evidence and self._worth_verifying(plan):
            checked: dict[str, bool | None] = {}
            t = time.monotonic()
            try:
                async for evidence_id, ok in self._verifier.verify(plan, evidence):
                    checked[evidence_id] = ok
                    yield _event("verified", {"evidence_id": evidence_id, "verified": ok})
            except Exception as exc:  # noqa: BLE001 - verification is optional; never lose the answer over it
                log.warning("verification failed: %s", exc)
                yield _event("note", {"text": "Verification was unavailable for this answer."})
            timings["verify"] = _ms(t)
            if self.cfg.verify == "filter":
                revised = self._revise(query_id, plan, evidence, checked, cameras, camera_by_id, tz, ref_now,
                                       all_notes, timings)
                if revised is not None:
                    answer = revised
                    yield _event("answer", answer.model_dump(mode="json"))
        timings["ttva"] = _ms(started)
        answer.timings_ms = timings
        self._log_query(query_id, text, plan, answer)
        yield _event("done", {"query_id": query_id})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _worth_verifying(plan: QueryPlan) -> bool:
        """Only questions whose answer depends on how something looks need a look.

        "a person at the gate after 8pm" is settled by the detector, the zone and the clock; "a red car" or
        "a person carrying a large bag" is not, so those get checked. Counts and first/last are left alone:
        the check covers only the top few, so it cannot settle them honestly.
        """
        return plan.intent in ("exists", "list") and bool(plan.targets) and bool(plan.targets[0].attributes)

    def _revise(self, query_id: str, plan: QueryPlan, evidence: list[Evidence], checked: dict[str, bool | None],
                cameras: list[_Camera], camera_by_id: dict[str, _Camera], tz: tzinfo, ref_now: float,
                notes: list[str], timings: dict[str, float]) -> Answer | None:
        """The answer after the visual check, or None when the check changed nothing.

        Candidates the check said no to are set aside; confirmed ones rank first, undecided ones keep their
        order after them. If nothing is left, the answer is a grounded not-found with the best rejected
        candidate as the nearest miss.
        """
        failed = [e for e in evidence if checked.get(e.id) is False]
        if not failed:
            return None
        kept = [e for e in evidence if checked.get(e.id) is not False]
        kept.sort(key=lambda e: 0 if checked.get(e.id) else 1)  # stable: confirmed before undecided
        kept = [e.model_copy(update={"verified": checked.get(e.id)}) for e in kept]
        miss = None
        if not kept:
            best = max(failed, key=lambda e: e.score)
            miss = best.model_copy(update={"verified": False, "why": [*best.why, "the visual check did not show it"]})
        what = plan.targets[0].embed_text.removeprefix("a photo of ").strip() or "the object"
        composed = compose_checked(
            plan, kept, nearest_miss=miss, cameras=_as_compose_cameras(cameras),
            source_names={c.id: c.source_name for c in cameras if c.source_name}, tz=tz, reference_now=ref_now,
            partial=_is_partial(cameras, plan, kept),
        )
        revised_notes = list(dict.fromkeys([*notes, *composed.notes,
                                            f"The visual check set aside {len(failed)} of {len(evidence)} "
                                            f"candidate(s) that did not clearly show {what}."]))
        return Answer(query_id=query_id, text=composed.text, verdict=composed.verdict, count=composed.count,
                      evidence=kept, nearest_miss=miss, confidence=_confidence(kept, miss), plan=plan,
                      timings_ms=timings, notes=revised_notes)

    def _clock(self) -> tuple[float, tzinfo]:
        override = self._reference_override()
        return (override if override is not None else reference_now(self._db)), workspace_tz(self._db)

    def _cameras(self) -> list[_Camera]:
        with self._db.read() as conn:
            rows = list(conn.execute("SELECT id, name, t0, layers, ir_fraction, source_uri, duration_s FROM cameras ORDER BY id"))
        out = []
        for r in rows:
            try:
                layers = json.loads(r["layers"] or "[]")
            except json.JSONDecodeError:
                layers = []
            out.append(_Camera(r["id"], r["name"], r["t0"], layers, r["ir_fraction"],
                               Path(r["source_uri"]).name if r["source_uri"] else None, r["duration_s"]))
        return out

    def _apply_fact(self, fact: MemoryFact, ref: Referent, bound: _Bound) -> None:
        b = fact.binding
        if fact.kind == "place":
            cam = b.get("camera_id")
            if not cam:
                return
            bound.camera_ids.add(cam)
            zone = self._zone(b.get("zone_id"), cam)
            bound.zones[cam] = zone
        elif fact.kind == "time":
            window = bound.window or TimeWindow()
            bound.window = window.model_copy(update={
                "tod_after": b.get("tod_after") or window.tod_after,
                "tod_before": b.get("tod_before") or window.tod_before})
        elif fact.kind == "object":
            gid = b.get("global_id")
            if gid:
                bound.global_ids.add(gid)
            else:
                bound.notes.append(f'"{ref.text}" is remembered but has no identity to match against yet.')

    def _zone(self, zone_id: str | None, camera_id: str) -> Zone:
        if zone_id:
            with self._db.read() as conn:
                row = conn.execute("SELECT id, camera_id, kind, points, direction FROM zones WHERE id=?",
                                   (zone_id,)).fetchone()
            if row is not None:
                return Zone(id=row["id"], camera_id=row["camera_id"], kind=row["kind"],
                            points=[tuple(p) for p in json.loads(row["points"] or "[]")], direction=row["direction"])
        return Zone(id=zone_id or f"frame_{camera_id}", camera_id=camera_id, kind="frame")

    @staticmethod
    def _with_bound_time(plan: QueryPlan, bound: _Bound) -> QueryPlan:
        if bound.window is None:
            return plan
        base = plan.time or TimeWindow()
        merged = base.model_copy(update={"tod_after": bound.window.tod_after or base.tod_after,
                                         "tod_before": bound.window.tod_before or base.tod_before})
        return plan.model_copy(update={"time": merged})

    def _reid_path(self, global_id: str) -> list[PathHop]:
        from evora.reid.paths import path_for

        return path_for(global_id, db=self._db)

    def _path_evidence(self, query_id: str, anchor: Match, cameras: dict[str, _Camera],
                       notes: list[str]) -> tuple[list[Evidence], list[PathHop]]:
        """The cameras one identity passed through, as evidence in time order.

        The anchor is the best match for the question; its identity's other tracks come from the re-ID
        links. A track with no identity gives a one-hop path, and the answer says so.
        """
        hops = self._path_for(anchor.global_id) if anchor.global_id else []
        if not hops:
            cam = cameras[anchor.camera_id]
            hops = [PathHop(camera_id=anchor.camera_id, camera_name=cam.name, t_in=anchor.t_start, t_out=anchor.t_end,
                            evidence_id=anchor.track_id.replace(":", "_"))]
            notes.append("This one was not linked to any other camera, so the path has a single stop.")
        elif len(hops) == 1:
            notes.append("Only one camera saw this identity, so the path has a single stop.")
        evidence: list[Evidence] = []
        final: list[PathHop] = []
        for i, hop in enumerate(hops, start=1):
            with self._db.read() as conn:
                row = conn.execute("SELECT id, camera_id, cls, t_start, t_end, best_t, global_id FROM tracks "
                                   "WHERE REPLACE(id, ':', '_') = ?", (hop.evidence_id,)).fetchone()
            if row is None or row["camera_id"] not in cameras:
                continue
            is_anchor = row["id"] == anchor.track_id
            peak = row["best_t"] if row["best_t"] is not None else hop.t_in
            why = anchor.why if is_anchor else (f"same identity as the best match ({row['global_id']})",)
            match = Match(row["id"], row["camera_id"], row["t_start"], peak, row["t_end"], anchor.score,
                          row["global_id"], None, why)
            ev = self._evidence(match, cameras, f"{query_id}_p{i}")
            evidence.append(ev)
            final.append(hop.model_copy(update={"evidence_id": ev.id}))
        return evidence, final

    async def _describe(self, query_id: str, text: str, plan: QueryPlan, notes: list[str], timings: dict[str, float],
                        started: float, cameras: list[_Camera], tz: tzinfo, ref_now: float,
                        bound: _Bound) -> AsyncIterator[StreamEvent]:
        """What happened: a grounded summary of the events in scope, with no retrieval involved."""
        t = time.monotonic()
        names = {c.id: c.name for c in cameras}
        camera_by_id = {c.id: c for c in cameras}
        scope = set(plan.camera_ids) | bound.camera_ids
        facts, overview = gather_facts(self._db, scope, plan.time, tz)
        where = (f"the {plan.place.text}" if plan.place else
                 " or ".join(f"the {names[c]} camera" for c in sorted(scope) if c in names))
        when = plan.time.phrase.strip() if plan.time and plan.time.phrase else ""
        days = {datetime.fromtimestamp(f.t, tz).date() for f in facts}
        with_date = len(days) > 1 or (bool(days) and days != {datetime.fromtimestamp(ref_now, tz).date()})
        narrated = await narrate(self._gateway, facts, overview, where, when)
        sources = {c.id: c.source_name for c in cameras if c.source_name}

        by_id = {f.id: f for f in facts}
        sentences = narrated or deterministic_sentences(facts, overview, where, when, tz, names, sources, with_date)
        cited = list(dict.fromkeys(fid for s in sentences for fid in s.fact_ids))[:MAX_DESCRIBE_EVIDENCE]
        evidence_for: dict[str, Evidence] = {}
        for n, fid in enumerate(cited, start=1):
            f = by_id[fid]
            match = Match(f.track_id, f.camera_id, f.track_start, f.t, f.track_end, 1.0, f.global_id, fid, (f.text,))
            evidence_for[fid] = self._evidence(match, camera_by_id, f"{query_id}_{n}")
        for ev in evidence_for.values():
            self._register(ev)
            yield _event("evidence", ev.model_dump(mode="json"))

        final: list[Sentence] = []
        for s in sentences:
            ids = tuple(evidence_for[f].id for f in s.fact_ids if f in evidence_for)
            body = s.text
            if narrated and ids:  # the model never writes times: the code adds them from the cited evidence
                first = evidence_for[next(f for f in s.fact_ids if f in evidence_for)]
                where_when = stamp(first, tz, with_date, sources.get(first.camera_id))
                body = f"{body[:-1] if body.endswith('.') else body}. At {where_when}."
            final.append(Sentence(body, ids, "fact" if ids else "negative"))
        evidence = list(evidence_for.values())
        validate(final, {e.id for e in evidence})
        timings["describe"] = _ms(t)
        timings["ttfa"] = _ms(started)
        if narrated:
            notes.append("Worded by the language model from the listed events; every sentence cites its evidence.")
        verdict = "found" if facts else "not_found"
        if facts and _is_partial(cameras, plan, evidence):
            verdict = "partial"
        all_notes = list(dict.fromkeys([*notes, *make_notes(plan, _as_compose_cameras(cameras), evidence)]))
        answer = Answer(query_id=query_id, text=" ".join(s.text for s in final), verdict=verdict, evidence=evidence,
                        confidence=1.0 if facts else 0.5, plan=plan, timings_ms=timings, notes=all_notes)
        yield _event("answer", answer.model_dump(mode="json"))
        timings["ttva"] = _ms(started)
        self._log_query(query_id, text, plan, answer)
        yield _event("done", {"query_id": query_id})

    def _events_for(self, track_ids: list[str]) -> list[EventRec]:
        out: list[EventRec] = []
        with self._db.read() as conn:
            for i in range(0, len(track_ids), EVENT_CHUNK):
                chunk = track_ids[i:i + EVENT_CHUNK]
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(
                        f"SELECT id, camera_id, track_id, kind, zone_id, t, payload FROM events "
                        f"WHERE track_id IN ({marks})", chunk):
                    out.append(EventRec.from_row(dict(row)))
        return out

    def _matches(self, plan: QueryPlan, cands: list[Candidate], bound: _Bound, tz: tzinfo) -> tuple[list[Match], int]:
        real = [c for c in cands if c.track.cls != "scene"]
        scenes = [c for c in cands if c.track.cls == "scene"]
        dropped = 0
        if plan.action == "any":
            pool = cands
        else:
            pool, dropped = real, len(scenes)  # a coarse tile cannot show a crossing or a dwell
        events = self._events_for([c.track.id for c in pool if c.track.cls != "scene"]) if plan.action != "any" else []
        matches = apply_action(pool, events, plan.action, bound.zones, plan.time, tz, zone_required=bool(bound.zones))
        return matches, dropped

    def _select(self, plan: QueryPlan, accepted: list[Match]) -> list[Match]:
        if plan.intent in ("first", "last"):
            return order_matches(accepted, plan.intent)[:1]
        per_track = best_per_track(accepted)
        return order_matches(per_track, "list")[: max(1, plan.limit)]

    def _evidence(self, m: Match, cameras: dict[str, _Camera], evidence_id: str) -> Evidence:
        cam = cameras[m.camera_id]
        if m.event_id is not None:
            lo, hi = m.t_peak - EVENT_PAD_S, m.t_peak + EVENT_PAD_S
        else:
            lo, hi = m.t_peak - TRACK_PAD_S, m.t_peak + TRACK_PAD_S
        lo, hi = min(max(lo, m.t_start), m.t_peak), max(min(hi, m.t_end), m.t_peak)
        is_scene = m.track_id.startswith("scene:")
        return Evidence(
            id=evidence_id, camera_id=m.camera_id, camera_name=cam.name,
            t_start=lo, t_end=hi, t_peak=m.t_peak, offset_s=self._offset_in_file(cam, m.t_peak),
            track_id=None if is_scene else m.track_id, global_id=m.global_id,
            bbox=None if is_scene else self._bbox(m.track_id, m.t_peak),
            thumb_url=self.cfg.thumb_fmt.format(id=evidence_id), clip_url=self.cfg.clip_fmt.format(id=evidence_id),
            score=round(m.score, 4), why=list(m.why),
        )

    def _offset_in_file(self, cam: _Camera, t: float) -> float:
        if self._file_offset is not None:
            try:
                looped = self._file_offset(cam.id, cam.duration_s, t)
            except Exception as exc:  # noqa: BLE001 - a bad offset hook must never break an answer
                log.warning("file offset hook failed for %s: %s", cam.id, exc)
                looped = None
            if looped is not None:
                return looped
        return t - cam.t0

    def _bbox(self, track_id: str, t: float) -> tuple[float, float, float, float] | None:
        with self._db.read() as conn:
            row = conn.execute("SELECT x1, y1, x2, y2 FROM track_points WHERE track_id=? "
                               "ORDER BY ABS(t - ?) LIMIT 1", (track_id, t)).fetchone()
        if row is None or None in tuple(row):
            return None
        return (row["x1"], row["y1"], row["x2"], row["y2"])

    def _register(self, ev: Evidence) -> None:
        try:
            register(self._db, ev)
        except EvidenceError as exc:
            log.warning("could not register evidence %s: %s", ev.id, exc)

    def _log_query(self, query_id: str, text: str, plan: QueryPlan, answer: Answer) -> None:
        try:
            with self._db.write() as conn:
                conn.execute(
                    "INSERT OR REPLACE INTO query_log(id, text, plan, answer, timings, created_at) VALUES(?,?,?,?,?,?)",
                    (query_id, text, plan.model_dump_json(), answer.model_dump_json(),
                     json.dumps(answer.timings_ms), time.time()))
        except Exception as exc:  # noqa: BLE001 - the log must never break an answer
            log.warning("could not write query_log: %s", exc)


# ---------------------------------------------------------------- module level
def _ms(since: float) -> float:
    return round((time.monotonic() - since) * 1000, 2)


def _confidence(evidence: Sequence[Evidence], miss: Evidence | None) -> float:
    if evidence:
        return round(max(e.score for e in evidence), 4)
    return round(1.0 - miss.score, 4) if miss else 0.5  # a confident "no" is one with no close miss


def _is_partial(cameras: list[_Camera], plan: QueryPlan, evidence: Sequence[Evidence]) -> bool:
    relevant = set(plan.camera_ids) | {e.camera_id for e in evidence}
    return any("L1" not in c.layers for c in cameras if not relevant or c.id in relevant)


@dataclass
class _ComposeCam:
    id: str
    name: str
    layers: list[str]
    ir_fraction: float | None


def _as_compose_cameras(cameras: list[_Camera]) -> list[_ComposeCam]:
    return [_ComposeCam(c.id, c.name, c.layers, c.ir_fraction) for c in cameras]
