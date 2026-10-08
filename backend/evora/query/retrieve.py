"""Hybrid retrieval over a workspace: hard filters, crop and scene search, attributes, captions.

Track-centric by default (contribution C1): candidates are tracks, scored from their
best crops, and cameras whose track layer is not ready yet fall back to coarse scene
tiles so queries still answer while indexing continues. The `unit`, `attributes` and
`expansion` switches are the ablation knobs from PLAN section 4.
"""
from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, tzinfo
from typing import Any, Literal, Protocol

import lancedb
import numpy as np
from contracts.models import QueryPlan, Target, TimeWindow

from evora.baseline.b0_frames import merge_hits
from evora.core.db import Database
from evora.query.expand import Gateway as ExpandGateway
from evora.query.expand import expand
from evora.query.fuse import (
    BM25,
    COLOUR_TERMS,
    DEFAULT_WEIGHTS,
    UPPER_BODY,
    Calibration,
    TrackSignals,
    aggregate_crops,
    asked_garments,
    attribute_score,
    blend,
    caption_supports_colour,
    explain_attributes,
    scene_support,
    squash_bm25,
)
from evora.query.logic import Candidate, TrackRec, instant_in_window, span_in_window
from evora.query.objects import sample_evenly

OTHER_COLOUR_CAP = 0.35     # below the router's accept line, so a person seen in another colour is never an answer
# colour read from a person's crops when the stored colour is unknown: image-text similarity over the 11 colour words,
# compared with each other (absolute cosines are tiny). Checked once against human labels of the upper body (about 70%
# right against 50% for always guessing the commonest colour); the trousers were no better than guessing, so only the
# upper-body wording is used. Constants were fixed before that check and not tuned on it.
ESTIMATE_PROMPT = "a photo of a person wearing {}"
ESTIMATE_TEMPERATURE = 1.5
ESTIMATE_MIN_P = 0.35
SAME_PLACE_MIN = 0.95    # similarity of two cameras' mean whole-frame embeddings above which they are treated as one place
VECTOR_CACHE = 128        # distinct query texts whose embeddings are kept
MIN_POINTS_IN_WINDOW = 2   # track points (about 4 per second) a track needs inside the window to count as present
MIN_PRESENCE_S = 0.5        # and they must span at least this long


class TextEmbedder(Protocol):
    def embed_text(self, text: str) -> np.ndarray: ...


@dataclass(frozen=True)
class RetrievalConfig:
    unit: Literal["track", "frame"] = "track"          # ablation C1
    attributes: bool = True                            # ablation C2
    expansion: Literal["off", "lexicon", "llm"] = "off"
    weights: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_WEIGHTS))
    calibration: Calibration = Calibration()
    ann_k: int = 400
    scene_k: int = 300
    pool_limit: int = 100
    # a camera where the detector found nothing in the window may still answer from whole-scene similarity, but only
    # when the window's best tile stands far above that camera's usual similarity (off until measured on real footage)
    scene_fallback: bool = False
    scene_fallback_z: float = 3.5      # robust z of the window's peak tile against all of the camera's tiles
    scene_fallback_min_tiles: int = 30  # too few tiles to know the camera's usual level: no fallback
    scene_fallback_score: float = 0.42  # just above the accept threshold: below any real track match

    @classmethod
    def from_cfg(cls, cfg: dict[str, Any] | None) -> RetrievalConfig:
        """Read the `retrieval:` section of config/default.yaml; every key is optional."""
        section = (cfg or {}).get("retrieval") or {}
        cal = section.get("calibration") or {}
        return cls(
            unit=section.get("unit", "track"),
            attributes=bool(section.get("attributes", True)),
            expansion=section.get("expansion", "off"),
            weights={**DEFAULT_WEIGHTS, **(section.get("weights") or {})},
            calibration=Calibration(cal.get("midpoint", Calibration.midpoint), cal.get("scale", Calibration.scale)),
            ann_k=int(section.get("ann_k", 400)),
            scene_k=int(section.get("scene_k", 300)),
            pool_limit=int(section.get("pool_limit", 100)),
            scene_fallback=bool(section.get("scene_fallback", False)),
            scene_fallback_z=float(section.get("scene_fallback_z", 3.5)),
            scene_fallback_min_tiles=int(section.get("scene_fallback_min_tiles", 30)),
            scene_fallback_score=float(section.get("scene_fallback_score", 0.42)),
        )


@dataclass(frozen=True)
class SearchScope:
    camera_ids: frozenset[str] = frozenset()   # empty = every camera
    window: TimeWindow | None = None           # anchored: start/end resolved, plus any time-of-day bounds
    tz: tzinfo = UTC


@dataclass
class RetrievalResult:
    candidates: list[Candidate] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    layers: list[str] = field(default_factory=list)  # which kinds of evidence contributed
    timings_ms: dict[str, float] = field(default_factory=dict)


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _in_clause(column: str, values: list[str]) -> str | None:
    return f"{column} IN ({', '.join(_quote(v) for v in values)})" if values else None


def _where(*clauses: str | None) -> str | None:
    parts = [c for c in clauses if c]
    return " AND ".join(parts) if parts else None


def _unit(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    return arr / max(float(np.linalg.norm(arr)), 1e-12)


class Retriever:
    def __init__(
        self,
        db: Database,
        store: lancedb.DBConnection,
        embedder: TextEmbedder,
        cfg: RetrievalConfig | None = None,
        gateway: ExpandGateway | None = None,
    ) -> None:
        self._db = db
        self._store = store
        self._embedder = embedder
        self.cfg = cfg or RetrievalConfig()
        self._gateway = gateway
        self._bm25_cache: tuple[int, BM25] | None = None
        self._vectors: dict[str, np.ndarray] = {}
        self._scene_means: tuple[int, dict[str, np.ndarray]] | None = None

    # ------------------------------------------------------------------ public
    async def search(self, plan: QueryPlan, scope: SearchScope | None = None) -> RetrievalResult:
        scope = scope or SearchScope()
        result = RetrievalResult()
        if not plan.targets:
            result.notes.append("The question names no object to search for.")
            return result
        if len(plan.targets) > 1:
            result.notes.append("Only the first object in the question was searched.")
        target = plan.targets[0]
        started = time.perf_counter()
        variants = await expand(target, self.cfg.expansion, self._gateway)
        result.timings_ms["expand"] = round((time.perf_counter() - started) * 1000, 2)
        started = time.perf_counter()
        await asyncio.to_thread(self._search_sync, plan, scope, variants, result)
        result.timings_ms["retrieve"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    # ----------------------------------------------------------------- internals
    def _tables(self) -> set[str]:
        return set(self._store.list_tables().tables)

    def scene_frames(self, camera_ids: Sequence[str], window: TimeWindow | None, tz: tzinfo = UTC,
                     per_camera: int = 12) -> dict[str, list[tuple[float, str]]]:
        """Whole-frame images stored for each camera inside the window, spread evenly: camera -> [(t, frame_path)]."""
        if not camera_ids or "scenes" not in self._tables():
            return {}
        table = self._store.open_table("scenes")
        where = _where(_in_clause("camera_id", list(camera_ids)), "tile = 'full'")
        rows = table.search().where(where, prefilter=True).select(["camera_id", "t", "frame_path"]).limit(200000).to_list()
        by_camera: dict[str, list[tuple[float, str]]] = {}
        for row in sorted(rows, key=lambda r: (r["camera_id"], r["t"])):
            if row.get("frame_path") and instant_in_window(float(row["t"]), window, tz):
                by_camera.setdefault(row["camera_id"], []).append((float(row["t"]), row["frame_path"]))
        return {camera: sample_evenly(frames, per_camera) for camera, frames in by_camera.items()}

    def _estimated_colours(self, track_ids: Sequence[str], target: Target) -> dict[str, tuple[str, float]]:
        """track id -> (colour, probability) estimated from its crops; only for people when a colour was asked for."""
        if "person" not in target.cls or not any(a in COLOUR_TERMS for a in target.attributes) or not track_ids:
            return {}
        terms = sorted(COLOUR_TERMS)
        text = np.stack([self._text_vector(ESTIMATE_PROMPT.format(c)) for c in terms])
        table = self._store.open_table("crops")
        sums: dict[str, np.ndarray] = {}
        for i in range(0, len(track_ids), 300):
            ids = list(track_ids[i:i + 300])
            where = _in_clause("track_id", ids)
            for row in table.search().where(where, prefilter=True).select(["track_id", "vector"]).limit(100000).to_list():
                vec = _unit(np.asarray(row["vector"], dtype=np.float32))
                sums[row["track_id"]] = sums.get(row["track_id"], 0.0) + vec
        out: dict[str, tuple[str, float]] = {}
        for tid, total in sums.items():
            cos = text @ _unit(total)
            z = (cos - cos.mean()) / (cos.std() + 1e-9)
            p = np.exp(z * ESTIMATE_TEMPERATURE)
            p /= p.sum()
            best = int(np.argmax(p))
            out[tid] = (terms[best], float(p[best]))
        return out

    def same_place_cameras(self, camera_ids: Sequence[str], threshold: float = SAME_PLACE_MIN) -> dict[str, float]:
        """Other cameras whose view looks like the same place as one of `camera_ids`: camera -> similarity.

        Two angles of one room share carpet, chairs, walls and light, so their mean whole-frame embeddings are close
        (0.98 in a two-camera room, against 0.63 to 0.86 between different cameras of a school). The threshold is a
        guard against joining places that merely look alike; it is reported with every use.
        """
        if not camera_ids or threshold > 1.0 or "scenes" not in self._tables():
            return {}
        table = self._store.open_table("scenes")
        count = table.count_rows()
        if self._scene_means is None or self._scene_means[0] != count:
            sums: dict[str, np.ndarray] = {}
            scan = table.search().where("tile = 'full'", prefilter=True).select(["camera_id", "vector"]).limit(500000)
            for row in scan.to_list():
                sums[row["camera_id"]] = sums.get(row["camera_id"], 0.0) + _unit(np.asarray(row["vector"], dtype=np.float32))
            self._scene_means = (count, {c: _unit(v) for c, v in sums.items()})
        means = self._scene_means[1]
        found: dict[str, float] = {}
        for anchor in camera_ids:
            if anchor not in means:
                continue
            for other, vec in means.items():
                similarity = float(means[anchor] @ vec)
                if other not in camera_ids and similarity >= threshold:
                    found[other] = max(found.get(other, 0.0), round(similarity, 3))
        return found

    def _text_vector(self, text: str) -> np.ndarray:
        """The unit text embedding, computed once per distinct text: crops and scenes share the same query."""
        vec = self._vectors.get(text)
        if vec is None:
            if len(self._vectors) >= VECTOR_CACHE:
                self._vectors.pop(next(iter(self._vectors)))  # oldest first
            vec = self._vectors[text] = _unit(self._embedder.embed_text(text))
        return vec

    def _all_cameras(self) -> list[tuple[str, str]]:
        with self._db.read() as conn:
            return [(r["id"], r["name"]) for r in conn.execute("SELECT id, name FROM cameras ORDER BY id")]

    def _camera_layers(self, cameras: list[str]) -> dict[str, set[str]]:
        with self._db.read() as conn:
            marks = ",".join("?" * len(cameras))
            rows = conn.execute(f"SELECT id, layers FROM cameras WHERE id IN ({marks})", cameras)
            return {r["id"]: set(json.loads(r["layers"] or "[]")) for r in rows}

    def _cameras_with_tracks(self, cameras: list[str]) -> set[str]:
        with self._db.read() as conn:
            marks = ",".join("?" * len(cameras))
            rows = conn.execute(f"SELECT DISTINCT camera_id FROM tracks WHERE camera_id IN ({marks})", cameras)
            return {r["camera_id"] for r in rows}

    def _scene_rows(self, cams: list[str], tables: set[str]) -> int:
        if "scenes" not in tables or not cams:
            return 0
        return int(self._store.open_table("scenes").count_rows(_in_clause("camera_id", cams)))

    def _search_sync(self, plan: QueryPlan, scope: SearchScope, variants: list[str], out: RetrievalResult) -> None:
        names = dict(self._all_cameras())
        cams = list(names)
        if scope.camera_ids:
            cams = [c for c in cams if c in scope.camera_ids]
        if not cams:
            out.notes.append("No matching camera has been added yet.")
            return
        tables = self._tables()
        target = plan.targets[0]

        track_cams = set() if self.cfg.unit == "frame" else self._cameras_with_tracks(cams)
        # a camera whose detection layer finished and found nothing is empty, not unfinished
        layers = self._camera_layers(cams) if self.cfg.unit != "frame" else {}
        empty = [c for c in cams if c not in track_cams and "L1" in layers.get(c, set())]
        scene_only = [c for c in cams if c not in track_cams and c not in empty]
        if empty:
            out.notes.append("Nothing was detected on " + ", ".join(names.get(c, c) for c in empty) + ".")

        if track_cams and "crops" in tables:
            self._track_candidates(plan, scope, sorted(track_cams), variants, tables, out)
            if self.cfg.scene_fallback and "scenes" in tables and _fallback_allowed(plan, scope):
                found = {c.track.camera_id for c in out.candidates}
                missed = [c for c in sorted(track_cams) if c not in found]
                if missed:
                    self._scene_fallback(plan, scope, missed, variants, out, names)
        elif track_cams:
            scene_only = cams  # tracks exist but no crop vectors yet: scenes are all we can search
            out.notes.append("Track embeddings are not ready yet; searched coarse scene tiles instead.")

        if scene_only and "scenes" in tables:
            self._scene_candidates(scope, scene_only, variants, out, full_frames_only=self.cfg.unit == "frame")
            if self.cfg.unit != "frame":
                out.notes.append(
                    "Still indexing " + ", ".join(names.get(c, c) for c in scene_only)
                    + ": only coarse scene search is available there."
                )

        if not out.candidates and not track_cams and scene_only and self._scene_rows(scene_only, tables) == 0:
            out.notes.append("Nothing has been indexed for the selected cameras yet.")

        out.candidates.sort(key=lambda c: (-c.score, c.track.t_start))
        del out.candidates[self.cfg.pool_limit:]
        if target.attributes and not self.cfg.attributes:
            out.notes.append("Attribute matching is switched off.")

    # --- tracks
    def _pool(self, plan: QueryPlan, scope: SearchScope, cams: list[str]) -> dict[str, dict[str, Any]]:
        target = plan.targets[0]
        sql = ("SELECT id, camera_id, cls, t_start, t_end, best_t, attrs, global_id FROM tracks "
               f"WHERE camera_id IN ({','.join('?' * len(cams))})")
        args: list[Any] = list(cams)
        if target.cls:
            sql += f" AND cls IN ({','.join('?' * len(target.cls))})"
            args += target.cls
        window = scope.window
        if window is not None and window.start is not None:
            sql += " AND t_end >= ?"
            args.append(window.start)
        if window is not None and window.end is not None:
            sql += " AND t_start <= ?"
            args.append(window.end)
        with self._db.read() as conn:
            rows = [dict(r) for r in conn.execute(sql, args)]
        return {r["id"]: r for r in rows if span_in_window(r["t_start"], r["t_end"], window, scope.tz)}

    def _crop_sims(self, cams: list[str], classes: list[str], variants: list[str]) -> dict[str, list[tuple[float, float]]]:
        """Per variant, per track: (cosine, crop time) pairs. Returns the best variant per track."""
        table = self._store.open_table("crops")
        where = _where(_in_clause("camera_id", cams), _in_clause("cls", classes))
        best: dict[str, tuple[float, list[tuple[float, float]]]] = {}
        for text in variants:
            query = table.search(self._text_vector(text)).metric("cosine")
            if where:
                query = query.where(where, prefilter=True)
            per_track: dict[str, list[tuple[float, float]]] = {}
            for row in query.limit(self.cfg.ann_k).to_list():
                per_track.setdefault(row["track_id"], []).append((1.0 - float(row["_distance"]), float(row["t"])))
            for tid, pairs in per_track.items():
                score = aggregate_crops([s for s, _ in pairs], self.cfg.unit)
                if tid not in best or score > best[tid][0]:
                    best[tid] = (score, pairs)
        return {tid: pairs for tid, (_, pairs) in best.items()}

    def _scene_hits(self, cams: list[str], variants: list[str], scope: SearchScope,
                    full_only: bool = False) -> list[tuple[str, float, float]]:
        """(camera, t, cosine) for scene tiles passing the camera filter and the time window."""
        table = self._store.open_table("scenes")
        where = _where(_in_clause("camera_id", cams), "tile = 'full'" if full_only else None)
        best: dict[tuple[str, float], float] = {}
        for text in variants:
            query = table.search(self._text_vector(text)).metric("cosine")
            if where:
                query = query.where(where, prefilter=True)
            for row in query.limit(self.cfg.scene_k).to_list():
                key = (row["camera_id"], float(row["t"]))
                cos = 1.0 - float(row["_distance"])
                if instant_in_window(key[1], scope.window, scope.tz) and cos > best.get(key, -1.0):
                    best[key] = cos
        return [(cam, t, cos) for (cam, t), cos in best.items()]

    @staticmethod
    def _caption_agrees(bm25: BM25 | None, tid: str, target: Target) -> bool:
        """Words matching is not agreement: when a colour is asked for, the caption must attach it to the garment."""
        colours = [a for a in target.attributes if a in COLOUR_TERMS]
        if not colours or bm25 is None:
            return True
        return caption_supports_colour(bm25.docs.get(tid, ""), colours, asked_garments(target.embed_text))

    @staticmethod
    def _colour_verdict(bm25: BM25 | None, tid: str, target: Target, attrs: dict[str, Any]) -> bool | None:
        """True when a caption or a confident stored colour backs the colour asked for, False when it shows another
        colour on that garment, None when nothing is known (an unknown colour is not a different colour)."""
        colours = [a for a in target.attributes if a in COLOUR_TERMS]
        if not colours or attrs.get("is_ir"):
            return None
        garments = asked_garments(target.embed_text)
        caption = bm25.docs.get(tid) if bm25 is not None else None
        if caption:
            if caption_supports_colour(caption, colours, garments):
                return True
            if caption_supports_colour(caption, sorted(COLOUR_TERMS - set(colours)), garments):
                return False
        slot = "upper_color" if garments & UPPER_BODY else "lower_color" if garments else "color"
        stored = attrs.get(slot)
        if isinstance(stored, str) and stored:
            return stored in colours
        return None

    def _captions(self, cams: list[str], tables: set[str]) -> BM25 | None:
        if "captions" not in tables:
            return None
        table = self._store.open_table("captions")
        count = table.count_rows()
        if count == 0:
            return None
        if self._bm25_cache is not None and self._bm25_cache[0] == count:
            return self._bm25_cache[1]
        docs: dict[str, str] = {}
        for row in table.to_arrow().to_pylist():
            if row.get("track_id"):
                docs[row["track_id"]] = (docs.get(row["track_id"], "") + " " + row["text"]).strip()
        bm25 = BM25(docs)
        self._bm25_cache = (count, bm25)
        return bm25

    def _presence(self, pool: dict[str, dict[str, Any]], scope: SearchScope) -> dict[str, list[float]]:
        """Times each pooled track was actually on screen inside the query window (from track_points).

        A track whose span merely overlaps the window (a parked car, a long visit) is not present in it.
        Tracks with no stored points are left out of the result, so callers keep the span-based decision.
        """
        window = scope.window
        if window is None or not (window.start is not None or window.end is not None or window.tod_after
                                  or window.tod_before):
            return {}
        out: dict[str, list[float]] = {}
        ids = list(pool)
        bounds, args = "", []
        if window.start is not None:  # the same inclusive bounds instant_in_window applies, now in SQL
            bounds, args = bounds + " AND t >= ?", [*args, window.start]
        if window.end is not None:
            bounds, args = bounds + " AND t <= ?", [*args, window.end]
        with self._db.read() as conn:
            for i in range(0, len(ids), 400):
                chunk = ids[i:i + 400]
                marks = ",".join("?" * len(chunk))
                for row in conn.execute(f"SELECT track_id, t FROM track_points WHERE track_id IN ({marks}){bounds} "
                                        "ORDER BY track_id, t", [*chunk, *args]):
                    out.setdefault(row["track_id"], []).append(row["t"])
            if bounds:
                # a track with points but none inside the window is "never on screen in it" (empty list), which
                # differs from a track with no stored points at all (absent): one index seek per such track
                for tid in ids:
                    if tid not in out and conn.execute("SELECT 1 FROM track_points WHERE track_id=? LIMIT 1",
                                                       (tid,)).fetchone():
                        out[tid] = []
        return {tid: [t for t in times if instant_in_window(t, window, scope.tz)] for tid, times in out.items()}

    def _track_candidates(self, plan: QueryPlan, scope: SearchScope, cams: list[str], variants: list[str],
                          tables: set[str], out: RetrievalResult) -> None:
        target = plan.targets[0]
        pool = self._pool(plan, scope, cams)
        if not pool:
            return
        cal = self.cfg.calibration
        sims = self._crop_sims(cams, target.cls, variants)
        scenes = []
        if "scenes" in tables and self.cfg.weights.get("scene", 0) > 0:
            scenes = [(c, t, cal(s)) for c, t, s in self._scene_hits(cams, variants, scope)]
        bm25 = self._captions(cams, tables)
        caption_scores = bm25.scores(" ".join([target.noun, *target.attributes, target.embed_text])) if bm25 else {}

        present = self._presence(pool, scope)
        estimates = self._estimated_colours(list(pool), target) if self.cfg.attributes and "crops" in tables else {}
        used: set[str] = set()
        for tid, row in pool.items():
            inside = present.get(tid)
            if inside is not None and (len(inside) < MIN_POINTS_IN_WINDOW or inside[-1] - inside[0] < MIN_PRESENCE_S):
                continue  # overlaps the window by span only: not on screen in it
            sig = TrackSignals(tid)
            pairs = sims.get(tid)
            if pairs:
                cos = aggregate_crops([s for s, _ in pairs], self.cfg.unit)
                sig.image, sig.raw_cosine = cal(cos), cos
                sig.peak_t = max(pairs, key=lambda p: p[0])[1]
                sig.why.append(f"siglip {cos:.2f}")
                used.add("crops")
            attrs = json.loads(row["attrs"] or "{}")
            other_colour = self.cfg.attributes and self._colour_verdict(bm25, tid, target, attrs) is False
            if self.cfg.attributes:
                sig.attributes = attribute_score(target.attributes, attrs)
                if sig.attributes is not None:
                    sig.why += explain_attributes(target.attributes, attrs)
                    used.add("attributes")
                elif tid in estimates:  # no stored colour: an estimate from the crops, labelled as one
                    term, p = estimates[tid]
                    if term in target.attributes and p >= ESTIMATE_MIN_P:
                        # support for answers and counts only: the ranking is unchanged (image similarity already
                        # carries colour, and an estimate did not improve it on labelled data)
                        sig.why.append(f"colour {term} (estimated {p:.2f})")
                    else:  # looked at, and it looks like something else: checked, not matching
                        sig.why.append(f"estimated colour {term} {p:.2f}")
            if tid in caption_scores and self._caption_agrees(bm25, tid, target):
                sig.caption = squash_bm25(caption_scores[tid])
                sig.why.append("caption match")
                used.add("captions")
            support = scene_support(row["camera_id"], row["t_start"], row["t_end"], scenes)
            if support is not None:
                sig.scene = support
                sig.why.append(f"scene {support:.2f}")
                used.add("scenes")
            if sig.image is None and sig.attributes is None and sig.caption is None:
                continue  # nothing says this track matches: not a candidate
            peak = sig.peak_t if sig.peak_t is not None else row["best_t"]
            if inside:  # the best crop may be from another part of a long track: use the nearest moment in the window
                if peak is None or not instant_in_window(peak, scope.window, scope.tz):
                    peak = min(inside, key=lambda t: abs(t - peak)) if peak is not None else inside[len(inside) // 2]
            track = TrackRec(tid, row["camera_id"], row["cls"], row["t_start"], row["t_end"], peak, row["global_id"],
                             attrs.get("person_group"))
            score, why = blend(sig, self.cfg.weights), tuple(sig.why)
            if other_colour:  # a caption or a stored colour shows another colour on that garment: a near miss at best
                score, why = min(score, OTHER_COLOUR_CAP), (*why, "another colour")
            out.candidates.append(Candidate(track, score, why))
        out.layers += sorted(used - set(out.layers))

    # --- scene fallback for cameras where the detector found nothing
    def _scene_fallback(self, plan: QueryPlan, scope: SearchScope, cams: list[str], variants: list[str],
                        out: RetrievalResult, names: dict[str, str]) -> None:
        """One weak candidate per camera whose best tile in the window stands out from that camera's usual tiles.

        Whole-frame similarity barely separates a small person from an empty street, so a candidate needs the window's
        peak to be an outlier for this camera (a robust z-score over all its tiles), not merely a high cosine.
        """
        table = self._store.open_table("scenes")
        noun = plan.targets[0].noun
        for cam in cams:
            where = _in_clause("camera_id", [cam])
            total = int(table.count_rows(where))
            if total < self.cfg.scene_fallback_min_tiles:
                continue
            best: dict[tuple[float, str], float] = {}
            for text in variants:
                query = table.search(self._text_vector(text)).metric("cosine").where(where, prefilter=True).limit(total)
                for row in query.to_list():
                    key = (float(row["t"]), str(row["tile"]))
                    best[key] = max(best.get(key, -1.0), 1.0 - float(row["_distance"]))
            inside = [(t, cos) for (t, _), cos in best.items() if instant_in_window(t, scope.window, scope.tz)]
            if not inside:
                continue
            values = np.fromiter(best.values(), dtype=np.float64)
            med = float(np.median(values))
            spread = 1.4826 * float(np.median(np.abs(values - med)))
            if spread <= 1e-9:
                continue
            t_peak, peak = max(inside, key=lambda p: p[1])
            z = (peak - med) / spread
            if z < self.cfg.scene_fallback_z:
                continue
            track = TrackRec(f"scene:{cam}:{t_peak:.0f}", cam, "scene", t_peak - 1.0, t_peak + 1.0, t_peak, None)
            out.candidates.append(Candidate(track, self.cfg.scene_fallback_score,
                                            (f"scene fallback z {z:.1f}", f"scene siglip {peak:.2f}")))
            out.notes.append(f"No {noun} was detected on {names.get(cam, cam)} in that window; the moment shown there "
                             "is where the whole picture looks most like it, which is weaker evidence.")
        if "scenes" not in out.layers and any(c.track.cls == "scene" for c in out.candidates):
            out.layers.append("scenes")

    # --- scenes only
    def _scene_candidates(self, scope: SearchScope, cams: list[str], variants: list[str], out: RetrievalResult,
                          full_frames_only: bool) -> None:
        hits = self._scene_hits(cams, variants, scope, full_only=full_frames_only)
        cal = self.cfg.calibration
        for w in merge_hits(hits):
            tid = f"scene:{w.camera_id}:{w.t_start:.0f}"
            track = TrackRec(tid, w.camera_id, "scene", w.t_start, w.t_end, w.t_peak, None)
            out.candidates.append(Candidate(track, cal(w.score), (f"scene siglip {w.score:.2f}",)))
        if "scenes" not in out.layers:
            out.layers.append("scenes")


def _fallback_allowed(plan: QueryPlan, scope: SearchScope) -> bool:
    """Only presence questions with a time window: counts and paths need real tracks, an action needs a track to show it."""
    window = scope.window
    has_window = window is not None and any(
        v is not None for v in (window.start, window.end, window.tod_after, window.tod_before))
    return has_window and plan.intent in ("exists", "list", "first", "last") and plan.action in ("any", "appear")
