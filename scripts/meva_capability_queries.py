"""Capability-specific ground truth from MEVA's object annotations (boxes and types), not activities.

The activity queries from meva_to_queries.py ask for things we do not recognise ("put something down").
This script instead asks only what the system claims to do: find an object of a class on a camera in a
time window, and say honestly when there is none. Ground truth comes from the annotated per-frame boxes
(`*.geom.yml`) and actor classes (`*.types.yml`), never from our own detector.

Caveat that shapes what is generated: MEVA labels only actors that take part in its annotated activities.
A positive query is valid (the labelled actor really is there) but conservative (a correct unlabelled
object scores as a miss). A negative query is only built for a class that has no annotated actor anywhere in
the camera's clip, because a window that is merely empty of annotated actors may still hold an unlabelled
parked car or bystander. Counts cannot be scored for the same reason (annotated counts are lower bounds).

    python scripts/meva_capability_queries.py --date 2018-03-09 --start 10-10-00 \
        --camera-map eval/meva_school_cameras.json --splits dev:0.5,test:0.3,judge_sim:0.2 \
        --out eval/queries/meva_capabilities.yaml

Only the exhaustively annotated `kitware` folder is used: "nothing is there" is only trustworthy where every
instance was labelled. Splits are by camera, so tuning on dev never sees a test scene. Capabilities that the
data cannot ground (colour, carrying, counts, cross-camera paths) are not generated; see UNSUPPORTED.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from meva_to_queries import (  # noqa: E402 - sibling script, run as `python scripts/...`
    FPS,
    activity_files,
    clip_end,
    clip_start,
    parse_tz,
    window_clips,
)

WINDOW_S = 60
MAX_ACTORS = 8          # a window with more is too easy to hit and too hard to count
MIN_PRESENCE_S = 1.0    # an actor must be on screen this long in the window to count
SEGMENT_GAP_FRAMES = 30  # a gap of a second or less is the same appearance
CLASSES = {"person": ("person", "people"), "vehicle": ("vehicle", "vehicles")}
GEOM = re.compile(r"'g0': '(\d+) (\d+) (\d+) (\d+)'.*?'id1': (\d+).*?'ts0': (\d+)")
ANNOTATION_SIZE = (1920.0, 1080.0)  # annotation pixels; the ingested videos are the same 16:9 at 1280x720

UNSUPPORTED = {
    "count": "MEVA annotates only actors that take part in its annotated activities. Parked cars and bystanders are "
             "unlabelled (on G328, 10 of the 15 vehicle tracks are stationary cars present for the whole clip), so "
             "annotated counts are lower bounds and an exact count cannot be scored.",
    "carrying": "Only one camera (G299) has carrying annotations and the carried things are class 'other' "
                "(boxes and similar), not bags; across the six cameras there is one bag actor. Nothing reliable "
                "to score a bag-carrying attribute against.",
    "colour": "MEVA has no clothing or vehicle colour labels. Use scripts/colour_label_tool.py (blind human labels) "
              "and generate queries from labels.json.",
    "path": "Actor ids are per clip and camera, so MEVA has no cross-camera identity ground truth.",
}


@dataclass
class Actor:
    id: int
    cls: str
    frames: list[int] = field(default_factory=list)

    def segments(self, gap: int = SEGMENT_GAP_FRAMES) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        for f in sorted(set(self.frames)):
            if out and f - out[-1][1] <= gap:
                out[-1] = (out[-1][0], f)
            else:
                out.append((f, f))
        return out


def read_types(path: Path) -> dict[int, str]:
    classes: dict[int, str] = {}
    for entry in yaml.safe_load(path.read_text(encoding="utf-8")) or []:
        t = entry.get("types") if isinstance(entry, dict) else None
        if t and "id1" in t and t.get("cset3"):
            classes[int(t["id1"])] = max(t["cset3"], key=t["cset3"].get)
    return classes


def read_actors(types_path: Path, geom_path: Path) -> dict[int, Actor]:
    classes = read_types(types_path)
    actors = {i: Actor(i, c) for i, c in classes.items()}
    for line in geom_path.read_text(encoding="utf-8").splitlines():
        m = GEOM.search(line)
        if m and int(m.group(5)) in actors:
            actors[int(m.group(5))].frames.append(int(m.group(6)))
    return {i: a for i, a in actors.items() if a.frames}


def windows_of(clip_len_s: float, size: int = WINDOW_S) -> list[tuple[int, int]]:
    """Whole windows only: a 301 s clip gives five 60 s windows, not a sixth one-second sliver."""
    return [(s, s + size) for s in range(0, int(clip_len_s // size) * size, size)]


def hits_in_window(actors: list[Actor], w0: int, w1: int) -> list[tuple[int, int]]:
    """(start_s, end_s) of every appearance overlapping the window by at least MIN_PRESENCE_S, clipped to it."""
    out = []
    for actor in actors:
        for a, b in actor.segments():
            s, e = max(a / FPS, w0), min(b / FPS, w1)
            if e - s >= MIN_PRESENCE_S:
                out.append((round(s, 2), round(e, 2)))
    return sorted(out)


def coverage(hits: list[tuple[int, int]], window_s: int = WINDOW_S) -> float:
    """Share of the window that annotated actors occupy (union of their appearances). Low coverage means a random
    moment in the window is unlikely to land on an actor, which is what makes chance a weak baseline."""
    total, current = 0.0, None
    for a, b in sorted(hits):
        if current is None or a > current[1]:
            total += (current[1] - current[0]) if current else 0.0
            current = [a, b]
        else:
            current[1] = max(current[1], b)
    total += (current[1] - current[0]) if current else 0.0
    return total / window_s


def distinct_in_window(actors: list[Actor], w0: int, w1: int) -> int:
    return sum(1 for a in actors if any(min(b / FPS, w1) - max(s / FPS, w0) >= MIN_PRESENCE_S for s, b in a.segments()))


def _hhmm(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).strftime("%H:%M")


def _iso(base: datetime, seconds: float) -> str:
    return (base + timedelta(seconds=seconds)).isoformat(timespec="milliseconds")


def build_items(root: Path, date: str, start: str, tz: timezone, camera_map: dict[str, str], workspace: str,
                per_kind: int = 4, max_actors: int = MAX_ACTORS, max_coverage: float = 1.0,
                with_negatives: bool = True) -> list[dict[str, Any]]:
    """Object, negative and count queries for every mapped camera of one annotated window."""
    items: list[dict[str, Any]] = []
    for clip, activity_path in window_clips(activity_files(root), date, start):
        if clip.camera not in camera_map:
            continue
        types_path = activity_path.with_name(activity_path.name.replace(".activities.yml", ".types.yml"))
        geom_path = activity_path.with_name(activity_path.name.replace(".activities.yml", ".geom.yml"))
        if not (types_path.is_file() and geom_path.is_file()):
            continue
        base = clip_start(clip, tz)
        cam = camera_map[clip.camera]
        actors = read_actors(types_path, geom_path)
        clip_len = (clip_end(clip, tz) - base).total_seconds()  # from the clip name; annotated frames may stop earlier
        wins = windows_of(clip_len)

        def item(kind: str, text: str, expected: dict[str, Any], intent: str, w0: int, cap: str, extra: str,
                 clip=clip, cam=cam, base=base) -> dict[str, Any]:
            return {"id": f"meva_cap_{clip.camera}_{base.strftime('%H%M')}_{w0}_{kind}{extra}", "text": text,
                    "workspace": workspace, "intent": intent, "expected": expected,
                    "tags": ["meva", f"cap:{cap}", clip.camera], "split": "dev"}

        for cls, (singular, _plural) in CLASSES.items():
            pool = [a for a in actors.values() if a.cls == cls]
            rows = [(w0, w1, hits_in_window(pool, w0, w1), distinct_in_window(pool, w0, w1)) for w0, w1 in wins]
            positives = sorted((r for r in rows if 1 <= len(r[2]) and 1 <= r[3] <= max_actors
                               and coverage(r[2]) <= max_coverage), key=lambda r: (coverage(r[2]), r[3]))
            # "nothing there" only for a class with no annotated actor anywhere in the clip: a window that is merely
            # empty of annotated actors may still hold an unlabelled one (a parked car, a bystander)
            empties = [] if pool or not with_negatives else rows
            def when(w0: int, w1: int, clip=clip, base=base) -> str:
                return f"on {clip.camera} between {_hhmm(base, w0)} and {_hhmm(base, w1)}"

            for w0, w1, hits, _n in positives[:per_kind]:
                exp = {"verdict": "yes", "hits": [{"camera_id": cam, "start": _iso(base, s), "end": _iso(base, e)}
                                                 for s, e in hits]}
                items.append(item("object", f"Was there a {singular} {when(w0, w1)}?", exp, "exists", w0, "object",
                                  f"_{cls}"))
            for w0, w1, _, _ in empties[:per_kind]:
                items.append(item("negative", f"Was there a {singular} {when(w0, w1)}?",
                                  {"verdict": "no", "hits": []}, "exists", w0, "negative", f"_{cls}"))
    return items


def assign_by_camera(items: list[dict[str, Any]], ratios: dict[str, float],
                     pinned: dict[str, str] | None = None) -> dict[str, int]:
    """Whole cameras go to one split, richest first, each to the split furthest under its target share.

    `pinned` fixes cameras to a split first (for example the ones already looked at while debugging go to dev),
    and the rest are balanced over the splits around them, so held-out splits stay genuinely unseen.
    """
    pinned = pinned or {}
    total = sum(ratios.values())
    share = {k: v / total for k, v in ratios.items()}
    by_cam: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for it in items:
        by_cam[next(t for t in it["tags"] if re.fullmatch(r"G\d+", t))].append(it)
    counts = dict.fromkeys(ratios, 0)
    seen = 0
    for cam, split in pinned.items():
        for it in by_cam.pop(cam, []):
            it["split"] = split
            counts[split] += 1
            seen += 1
    for _, group in sorted(by_cam.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        taken = set(pinned.values())  # splits that already hold pinned cameras only get unpinned ones if none is left
        free = [n for n in ratios if n not in taken] or list(ratios)
        split = max(free, key=lambda n: (share[n] * (seen + len(group)) - counts[n], share[n]))
        for it in group:
            it["split"] = split
        counts[split] += len(group)
        seen += len(group)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--annotations", type=Path, default=Path("data/raw/meva-annotations"))
    parser.add_argument("--date", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--tz", default="+05:30")
    parser.add_argument("--camera-map", type=Path, required=True)
    parser.add_argument("--workspace", default="meva-school")
    parser.add_argument("--splits", default="dev:0.5,test:0.3,judge_sim:0.2")
    parser.add_argument("--dev-cameras", default="", help="MEVA camera codes pinned to dev, e.g. G328,G419 (the ones "
                        "already examined while debugging; everything else is balanced over test and judge_sim)")
    parser.add_argument("--per-kind", type=int, default=4, help="windows per camera, class and capability")
    parser.add_argument("--max-actors", type=int, default=MAX_ACTORS)
    parser.add_argument("--max-coverage", type=float, default=1.0,
                        help="keep only windows where annotated actors fill at most this share of the window")
    parser.add_argument("--no-negatives", action="store_true")
    parser.add_argument("--splits-from", type=Path, default=None,
                        help="a frozen.json: use its splits_by_camera instead of balancing (cameras keep their split)")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    cmap = json.loads(args.camera_map.read_text(encoding="utf-8"))
    items = build_items(args.annotations, args.date, args.start, parse_tz(args.tz), cmap, args.workspace, args.per_kind,
                        args.max_actors, args.max_coverage, not args.no_negatives)
    if not items:
        print("No ground truth could be built (no annotated clip for those cameras).", file=sys.stderr)
        return 2
    ratios = {}
    for part in args.splits.split(","):
        name, _, weight = part.partition(":")
        ratios[name] = float(weight)
    if args.splits_from is not None:
        fixed = json.loads(args.splits_from.read_text(encoding="utf-8"))["splits_by_camera"]
        owner = {cam: split for split, cams in fixed.items() for cam in cams}
        counts = dict.fromkeys(fixed, 0)
        for it in items:
            cam = next(t for t in it["tags"] if re.fullmatch(r"G\d+", t))
            it["split"] = owner.get(cam, "dev")  # an unlisted camera was never held out: it can only be dev
            counts[it["split"]] = counts.get(it["split"], 0) + 1
    else:
        pinned = {c.strip(): "dev" for c in args.dev_cameras.split(",") if c.strip()}
        counts = assign_by_camera(items, ratios, pinned)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("# generated by scripts/meva_capability_queries.py; ground truth from MEVA object annotations\n"
                        + yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=120), encoding="utf-8")
    caps: dict[str, int] = defaultdict(int)
    for it in items:
        caps[next(t for t in it["tags"] if t.startswith("cap:"))] += 1
    print(f"{len(items)} queries -> {args.out}")
    print("by capability:", ", ".join(f"{k[4:]} {v}" for k, v in sorted(caps.items())))
    print("by split:", ", ".join(f"{k} {v}" for k, v in counts.items()))
    for cap, why in UNSUPPORTED.items():
        print(f"unsupported - {cap}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
