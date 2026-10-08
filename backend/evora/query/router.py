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
from dataclasses import dataclass, field, replace
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
from evora.query import fastpath
from evora.query.action_cues import find_cues
from evora.query.actions import classify_action, unsupported_action
from evora.query.compose import (
    ESTIMATE_PREFIX,
    UNCONFIRMED_NOTE,
    Composed,
    Sentence,
    compose_checked,
    compose_objects,
    make_notes,
    pluralize,
    stamp,
    validate,
)
from evora.query.describe import deterministic_sentences, gather_facts, narrate
from evora.query.logic import (
    Candidate,
    EventRec,
    Match,
    apply_action,
    best_per_track,
    concurrency,
    count_distinct,
    instant_in_window,
    order_matches,
)
from evora.query.look import LookAnswerer, cannot_tell, pick_frames, wants_look, yes_no
from evora.query.objects import FrameRef, OpenObjectSurveyor, Survey, labels_for
from evora.query.planner import Planner, PlanningError, reference_now, workspace_tz
from evora.query.retrieve import Retriever, SearchScope

log = logging.getLogger("evora.query.router")

TRACK_PAD_S = 5.0     # evidence window around the best frame of a track
EVENT_PAD_S = 1.5     # evidence window around a crossing / entry / dwell
MAX_DESCRIBE_EVIDENCE = 8
MAX_LOOK_CAMERAS = 3       # cameras shown to the vision model for one question
MAX_OBJECT_EVIDENCE = 6   # boxes shown from the clearest frame of a camera
SAMPLE_FRAMES = 12        # stored frames looked at per camera for an object question
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
    honest_actions: bool = True   # a question about an action we cannot recognise is answered with who was there
    action_cues: bool = True      # ... with the most likely person and moment first, estimated from how things moved
    thumb_fmt: str = "/api/media/thumb/{id}.jpg"
    clip_fmt: str = "/api/media/clip/{id}.mp4"
    same_place: float = 0.95      # cameras this alike in view are treated as one place (1.0 + switches it off)


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
        objects: OpenObjectSurveyor | None = None,
        look: LookAnswerer | None = None,
    ) -> None:
        # shows frames to the local vision model for questions only the picture can answer (sitting, a phone, a colour)
        self._look = look
        # looks for things the tracker does not follow (chairs, carpets, "red objects") in the stored frames
        self._objects = objects
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
        started = time.perf_counter()
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
        started = time.perf_counter()
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
        self._widen_to_same_place(plan, bound, cameras, notes)

        if plan.intent == "standing":
            yield _event("note", {"text": "Watch requests are created from the Watch panel, not the question bar."})
            yield _event("done", {"query_id": query_id})
            return

        plan = self._with_bound_time(plan, bound)

        if self._look is not None and wants_look(plan, text) and not any(
                z.kind != "frame" for z in bound.zones.values()):
            looked = await self._look_answer(query_id, text, plan, notes, timings, started, cameras, tz, ref_now, bound)
            if looked is not None:
                for ev in looked:
                    yield ev
                return

        if plan.intent == "describe":
            async for ev in self._describe(query_id, text, plan, notes, timings, started, cameras, tz, ref_now, bound):
                yield ev
            return

        if self._objects is not None and plan.intent in ("count", "exists", "list", "first", "last") and plan.targets:
            labels = labels_for(plan.targets[0])
            if labels:
                async for ev in self._objects_answer(query_id, text, plan, labels, notes, timings, started, cameras, tz,
                                                     ref_now, bound):
                    yield ev
                return

        # 2. retrieval
        t = time.perf_counter()
        scope_cams = frozenset(set(plan.camera_ids) | bound.camera_ids)
        found = await self._retriever.search(plan, SearchScope(scope_cams, plan.time, tz))
        timings["retrieve"] = _ms(t)
        notes += [n for n in found.notes if n not in notes]

        # 3. logic: does each candidate actually satisfy the action, where and when
        t = time.perf_counter()
        cands = found.candidates
        if bound.global_ids:
            cands = [c for c in cands if c.track.global_id in bound.global_ids]
        matches, dropped = self._matches(plan, cands, bound, tz)
        if dropped:
            notes.append("Coarse scene matches were left out because they cannot show the requested action.")
        accepted = [m for m in matches if m.score >= self.cfg.accept]
        before = accepted
        accepted = _backed_by_attributes(plan, before, notes)
        checked_by_estimate = sum(1 for m in before if m not in accepted and any(w.startswith("estimated colour") for w in m.why))
        unreadable = (len(before) - len(accepted) - checked_by_estimate) if plan.intent == "count" else 0
        near = [m for m in matches if m.score < self.cfg.accept]
        timings["logic"] = _ms(t)

        # 4. evidence and answer
        t = time.perf_counter()
        camera_by_id = {c.id: c for c in cameras}
        hops: list[PathHop] = []
        if plan.intent == "path" and accepted:
            evidence, hops = self._path_evidence(query_id, max(accepted, key=lambda m: m.score), camera_by_id, notes)
        else:
            shown = self._select_with_cues(plan, accepted, text, tz)
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
        action = unsupported_action(text) if self.cfg.honest_actions else None
        concurrent = None
        if plan.intent == "count" and plan.action == "any" and accepted:
            # "how many are there" asks how many are in view, not how many track fragments exist
            concurrent = self._concurrency(accepted, plan.time, tz)
        composed = compose_checked(
            plan, evidence, count=count, nearest_miss=miss, path=hops, cameras=_as_compose_cameras(cameras),
            source_names={c.id: c.source_name for c in cameras if c.source_name}, tz=tz, reference_now=ref_now,
            partial=_is_partial(cameras, plan, evidence), unconfirmed=_unconfirmed(plan, evidence),
            concurrent=concurrent, appearances=count, unreadable=unreadable, action=action,
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
            t = time.perf_counter()
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
                                       all_notes, timings, action)
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
                notes: list[str], timings: dict[str, float], action: str | None = None) -> Answer | None:
        """The answer after the visual check, or None when the check changed nothing.

        Candidates the check said no to are set aside; confirmed ones rank first, undecided ones keep their
        order after them. If nothing is left, the answer is a grounded not-found with the best rejected
        candidate as the nearest miss.
        """
        failed = [e for e in evidence if checked.get(e.id) is False]
        kept = [e for e in evidence if checked.get(e.id) is not False]
        kept.sort(key=lambda e: 0 if checked.get(e.id) else 1)  # stable: confirmed before undecided
        kept = [e.model_copy(update={"verified": checked.get(e.id)}) for e in kept]
        if not failed and _unconfirmed(plan, evidence) == _unconfirmed(plan, kept):
            return None  # nothing set aside and the first answer's claim about the attributes still stands
        miss = None
        if not kept:
            best = max(failed, key=lambda e: e.score)
            miss = best.model_copy(update={"verified": False, "why": [*best.why, "the visual check did not show it"]})
        what = plan.targets[0].embed_text.removeprefix("a photo of ").strip() or "the object"
        composed = compose_checked(
            plan, kept, nearest_miss=miss, cameras=_as_compose_cameras(cameras),
            source_names={c.id: c.source_name for c in cameras if c.source_name}, tz=tz, reference_now=ref_now,
            partial=_is_partial(cameras, plan, kept), unconfirmed=_unconfirmed(plan, kept), action=action,
        )
        set_aside = ([f"The visual check set aside {len(failed)} of {len(evidence)} "
                      f"candidate(s) that did not clearly show {what}."] if failed else [])
        revised_notes = list(dict.fromkeys([*(n for n in notes if not n.startswith(UNCONFIRMED_NOTE)), *composed.notes,
                                            *set_aside]))
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

    def _widen_to_same_place(self, plan: QueryPlan, bound: _Bound, cameras: list[_Camera], notes: list[str]) -> None:
        """A remembered place that is a whole camera view also covers other cameras that show the same room."""
        finder = getattr(self._retriever, "same_place_cameras", None)
        whole = {c for c in bound.camera_ids if bound.zones.get(c) is None or bound.zones[c].kind == "frame"}
        if finder is None or plan.camera_ids or not whole or self.cfg.same_place > 1.0:
            return
        names = {c.id: c.name for c in cameras}
        extra = {c: s for c, s in finder(sorted(whole), self.cfg.same_place).items()
                 if c in names and c not in bound.camera_ids}
        if not extra:
            return
        for cam in extra:
            bound.camera_ids.add(cam)
            bound.zones[cam] = self._zone(None, cam)
        anchors = " and ".join(names[c] for c in sorted(whole))
        added = ", ".join(f"{names[c]} ({round(s * 100)}% alike)" for c, s in sorted(extra.items()))
        place = f"the {plan.place.text}" if plan.place else "this place"
        notes.append(f"{added} shows the same place as {anchors}, so it is included in {place}.")

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
        t = time.perf_counter()
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

    async def _look_answer(self, query_id: str, text: str, plan: QueryPlan, notes: list[str], timings: dict[str, float],
                           started: float, cameras: list[_Camera], tz: tzinfo, ref_now: float,
                           bound: _Bound) -> list[StreamEvent] | None:
        """Show frames of the camera(s) to the local vision model; None when it cannot answer (the usual path follows)."""
        t = time.perf_counter()
        camera_by_id = {c.id: c for c in cameras}
        scope = [c for c in sorted(set(plan.camera_ids) | bound.camera_ids) if c in camera_by_id] or sorted(camera_by_id)
        frames_for = getattr(self._retriever, "scene_frames", None)
        rows = frames_for(scope, plan.time, tz, 12) if frames_for is not None else {}
        if not rows:
            return None
        # two angles of one room need one look, not two
        chosen: list[str] = []
        for camera_id in [c for c in scope if c in rows]:
            alike = getattr(self._retriever, "same_place_cameras", None)
            if chosen and alike is not None and camera_id in alike(chosen, self.cfg.same_place):
                continue
            chosen.append(camera_id)
        chosen = chosen[:MAX_LOOK_CAMERAS]
        answers: list[tuple[str, str, list[Any]]] = []
        for camera_id in chosen:
            frames = pick_frames(rows[camera_id], camera_id)
            answer = await self._look.ask(text, camera_by_id[camera_id].name, frames, tz)
            if answer:
                answers.append((camera_id, answer, frames))
        if not answers:
            notes.append("The local vision model could not be reached, so the picture was not examined.")
            return None
        evidence: list[Evidence] = []
        sentences: list[Sentence] = []
        for camera_id, answer, frames in answers:
            cam = camera_by_id[camera_id]
            ids: list[str] = []
            for frame in frames:
                eid = f"{query_id}_{len(evidence) + 1}"
                ids.append(eid)
                evidence.append(Evidence(
                    id=eid, camera_id=cam.id, camera_name=cam.name, t_start=frame.t - 2.5, t_end=frame.t + 2.5,
                    t_peak=frame.t, offset_s=self._offset_in_file(cam, frame.t),
                    thumb_url=self.cfg.thumb_fmt.format(id=eid), clip_url=self.cfg.clip_fmt.format(id=eid),
                    score=1.0, why=["frame shown to the vision model"]))
            prefix = f"{cam.name}: " if len(answers) > 1 else ""
            sentences.append(Sentence(prefix + answer.rstrip(), tuple(ids), "fact"))
        validate(sentences, {e.id for e in evidence})
        for ev in evidence:
            self._register(ev)
        first = answers[0][1]
        unsure = all(cannot_tell(a) for _, a, _ in answers)
        verdict = "partial" if unsure else "found"
        if plan.intent == "exists" and not unsure:
            decided = yes_no(first)
            verdict = "yes" if decided is True else "no" if decided is False else "found"
        names = " and ".join(camera_by_id[c].name for c, _, _ in answers)
        all_notes = list(dict.fromkeys([
            *notes,
            f"Answered by the local vision model looking at {sum(len(f) for _, _, f in answers)} frames of {names}, shown "
            "below. It can miss or misread things, so check the frames.",
            *make_notes(plan, _as_compose_cameras(cameras), evidence)]))
        timings["look"] = _ms(t)
        timings["ttfa"] = _ms(started)
        answer_obj = Answer(query_id=query_id, text=" ".join(s.text for s in sentences), verdict=verdict, evidence=evidence,
                            confidence=0.5 if unsure else 0.7, plan=plan, timings_ms=timings, notes=all_notes)
        events = [_event("evidence", ev.model_dump(mode="json")) for ev in evidence]
        events.append(_event("answer", answer_obj.model_dump(mode="json")))
        timings["ttva"] = _ms(started)
        self._log_query(query_id, text, plan, answer_obj)
        events.append(_event("done", {"query_id": query_id}))
        return events

    async def _objects_answer(self, query_id: str, text: str, plan: QueryPlan, labels: list[str], notes: list[str],
                              timings: dict[str, float], started: float, cameras: list[_Camera], tz: tzinfo,
                              ref_now: float, bound: _Bound) -> AsyncIterator[StreamEvent]:
        """Something the tracker does not follow: look for it in the stored frames and answer from what is in view."""
        t = time.perf_counter()
        target = plan.targets[0]
        noun = target.noun.strip().lower()
        colour = next((a for a in target.attributes if a in fastpath.COLOURS.values()), None)
        camera_by_id = {c.id: c for c in cameras}
        scope = [c for c in sorted(set(plan.camera_ids) | bound.camera_ids) if c in camera_by_id] or sorted(camera_by_id)
        frames_for = getattr(self._retriever, "scene_frames", None)
        frames = frames_for(scope, plan.time, tz, SAMPLE_FRAMES) if frames_for is not None else {}
        surveys: list[Survey] = []
        if frames and self._objects is not None and self._objects.ready():
            for camera_id, rows in frames.items():
                refs = [FrameRef(camera_id, ft, path) for ft, path in rows]
                surveys.append(await self._objects.survey(camera_id, refs, labels, colour))
        evidence: list[Evidence] = []
        summaries: list[dict[str, Any]] = []
        for sv in surveys:
            best = sv.best_frame()
            cam = camera_by_id[sv.camera_id]
            summaries.append({"camera_id": cam.id, "camera_name": cam.name, "typical": sv.typical, "peak": sv.peak,
                              "least": sv.least, "frames": len(sv.frames), "seen_in": sv.seen_in,
                              "breakdown": dict(sv.breakdown(best)) if best else {}})
            if best is None:
                continue
            for det in sorted(sv.matching(best), key=lambda d: -d.conf)[:MAX_OBJECT_EVIDENCE]:
                eid = f"{query_id}_{len(evidence) + 1}"
                why = [f"open vocabulary: {det.label} {det.conf:.2f}"]
                if colour:
                    why.append(f"colour {colour} {det.colours.get(colour, 0.0):.2f}")
                evidence.append(Evidence(
                    id=eid, camera_id=cam.id, camera_name=cam.name, t_start=best.ref.t - 2.5, t_end=best.ref.t + 2.5,
                    t_peak=best.ref.t, offset_s=self._offset_in_file(cam, best.ref.t), bbox=det.box,
                    thumb_url=self.cfg.thumb_fmt.format(id=eid), clip_url=self.cfg.clip_fmt.format(id=eid),
                    score=round(det.conf, 4), why=why))
        for ev in evidence:
            self._register(ev)
            yield _event("evidence", ev.model_dump(mode="json"))
        if not surveys:
            why_not = (self._objects.unavailable if self._objects is not None and self._objects.unavailable else
                       "no stored frames for those cameras and that time")
            composed = Composed("partial", [Sentence(f"I can't look for {pluralize(noun)} here: {why_not}.", (), "negative")], [])
        else:
            composed = compose_objects(plan, summaries, evidence, noun=noun, colour=colour, tz=tz,
                                       source_names={c.id: c.source_name for c in cameras if c.source_name},
                                       reference_now=ref_now)
        validate(composed.sentences, {e.id for e in evidence})
        timings["objects"] = _ms(t)
        timings["ttfa"] = _ms(started)
        all_notes = list(dict.fromkeys([*notes, *composed.notes,
                                        *make_notes(plan, _as_compose_cameras(cameras), evidence)]))
        answer = Answer(query_id=query_id, text=composed.text, verdict=composed.verdict, count=composed.count,
                        evidence=evidence, confidence=_confidence(evidence, None), plan=plan, timings_ms=timings,
                        notes=all_notes)
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

    def _concurrency(self, accepted: list[Match], window: TimeWindow | None, tz: tzinfo) -> dict[str, dict[str, int]] | None:
        """Per camera: how many of the accepted tracks were in view at once, from their stored points."""
        by_camera: dict[str, list[str]] = {}
        for m in accepted:
            if not m.track_id.startswith("scene:"):
                by_camera.setdefault(m.camera_id, []).append(m.track_id)
        out: dict[str, dict[str, int]] = {}
        with self._db.read() as conn:
            for camera, ids in by_camera.items():
                times: dict[str, list[float]] = {}
                for i in range(0, len(ids), 400):
                    chunk = ids[i:i + 400]
                    marks = ",".join("?" * len(chunk))
                    for row in conn.execute(f"SELECT track_id, t FROM track_points WHERE track_id IN ({marks}) "
                                            "ORDER BY track_id, t", chunk):
                        if instant_in_window(row["t"], window, tz):
                            times.setdefault(row["track_id"], []).append(row["t"])
                result = concurrency(times)
                if result is not None:
                    out[camera] = result
        return out or None

    def _select_with_cues(self, plan: QueryPlan, accepted: list[Match], text: str, tz: tzinfo) -> list[Match]:
        """For an action we cannot recognise, matches whose movement fits it come first, shown at that moment."""
        found = classify_action(text) if self.cfg.honest_actions and self.cfg.action_cues else None
        if found is None or plan.intent in ("count", "path"):
            return self._select(plan, accepted)
        real = [m for m in accepted if not m.track_id.startswith("scene:")]
        cues = find_cues(self._db, found[0], {(m.track_id, m.camera_id) for m in real})
        cues = {tid: c for tid, c in cues.items() if instant_in_window(c.t, plan.time, tz)}
        if not cues:
            return self._select(plan, accepted)
        cued: dict[str, Match] = {}
        for m in real:
            cue = cues.get(m.track_id)
            if cue is not None and (m.track_id not in cued or m.score > cued[m.track_id].score):
                cued[m.track_id] = replace(m, t_peak=cue.t, why=(*m.why, f"{ESTIMATE_PREFIX}{cue.why}"))
        first = sorted(cued.values(), key=lambda m: (-cues[m.track_id].score, -m.score))
        if plan.intent in ("first", "last"):
            return first[:1]
        rest = self._select(plan, [m for m in accepted if m.track_id not in cued])
        return (first + rest)[: max(1, plan.limit)]

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
        # nearest stored point: one index seek on each side of t instead of sorting all of the track's points
        with self._db.read() as conn:
            before = conn.execute("SELECT t, x1, y1, x2, y2 FROM track_points WHERE track_id=? AND t <= ? "
                                  "ORDER BY t DESC LIMIT 1", (track_id, t)).fetchone()
            after = conn.execute("SELECT t, x1, y1, x2, y2 FROM track_points WHERE track_id=? AND t > ? "
                                 "ORDER BY t LIMIT 1", (track_id, t)).fetchone()
        candidates = [r for r in (before, after) if r is not None]
        row = min(candidates, key=lambda r: abs(r["t"] - t)) if candidates else None
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
    return round((time.perf_counter() - since) * 1000, 2)


def _confidence(evidence: Sequence[Evidence], miss: Evidence | None) -> float:
    if evidence:
        return round(max(e.score for e in evidence), 4)
    return round(1.0 - miss.score, 4) if miss else 0.5  # a confident "no" is one with no close miss


_SUPPORT_WHY = ("colour ", "carrying ", "vehicle type ", "caption match")


def _backed_by_attributes(plan: QueryPlan, accepted: list[Match], notes: list[str]) -> list[Match]:
    """When attributes are asked for, tracks with something behind them come first and alone.

    A person who merely looks like a person is not "a person in a brown shirt": the unread ones are used only when
    nothing is backed (the answer is then "partial"), and a count never includes them.
    """
    if plan.intent not in ("exists", "list", "first", "last", "count") or not plan.targets or not plan.targets[0].attributes:
        return accepted
    backed = [m for m in accepted if any(w.startswith(_SUPPORT_WHY) for w in m.why)]
    estimated = sum(1 for m in backed if any("(estimated" in w for w in m.why))
    if estimated:
        notes.append(f"{estimated} of these colours were estimated from how the person looks, not read from the clothes, "
                     "so they are less certain.")
    if len(backed) == len(accepted):
        return accepted
    if plan.intent == "count":
        looked_at = sum(1 for m in accepted if m not in backed and any(w.startswith("estimated colour") for w in m.why))
        if looked_at:
            notes.append(f"{looked_at} more were checked from how they look and did not look "
                         f"{' '.join(plan.targets[0].attributes)}; that estimate can be wrong.")
        unread = len(accepted) - len(backed) - looked_at
        if unread:
            notes.append(f"{unread} more could not be checked for {' '.join(plan.targets[0].attributes)}.")
        return backed
    return backed or accepted


def _unconfirmed(plan: QueryPlan, evidence: Sequence[Evidence]) -> bool:
    """Attributes were asked for and no candidate has anything behind them: a stored attribute, a caption that
    ties the colour to the garment, or a visual check that said yes."""
    if plan.intent not in ("exists", "list", "first", "last") or not plan.targets or not plan.targets[0].attributes:
        return False
    if not evidence:
        return False
    return not any(e.verified is True or any(w.startswith(_SUPPORT_WHY) for w in e.why) for e in evidence)


def _is_partial(cameras: list[_Camera], plan: QueryPlan, evidence: Sequence[Evidence]) -> bool:
    relevant = set(plan.camera_ids) | {e.camera_id for e in evidence}
    if any(w.startswith("scene fallback") for e in evidence for w in e.why):
        return True  # an answer resting on whole-scene similarity is never a full yes
    return any("L1" not in c.layers for c in cameras if not relevant or c.id in relevant)


@dataclass
class _ComposeCam:
    id: str
    name: str
    layers: list[str]
    ir_fraction: float | None


def _as_compose_cameras(cameras: list[_Camera]) -> list[_ComposeCam]:
    return [_ComposeCam(c.id, c.name, c.layers, c.ir_fraction) for c in cameras]
