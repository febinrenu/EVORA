"""Turn MEVA activity annotations into ground-truth queries for the eval harness (PLAN P3.15).

    # which five-minute windows have the most annotated cameras? (pick the footage to download from this)
    python scripts/meva_to_queries.py find --annotations data/raw/meva-annotations --min-cams 4

    # queries for one window; hits are the annotated instances in every camera of that window
    python scripts/meva_to_queries.py generate --annotations data/raw/meva-annotations \
        --date 2018-03-09 --start 10-10-00 --split dev --out eval/queries/meva_dev.yaml

Annotation files are named <date>.<start>.<end>.<site>.<camera>.activities.yml and hold KPF
entries whose time spans are frame numbers in the clip (30 fps). Times in clip names are local
time; --tz must match how the ingest set the camera clock (the team reads them as +05:30).
Camera ids in the output are the MEVA camera codes unless --camera-map maps them to workspace ids.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import yaml

FPS = 30.0
MAX_HITS = 50
BROAD_HITS = 20  # a query with this many matching windows is easy to satisfy; tagged so it can be sliced out
ANNOTATION_SUBPATH = Path("annotation") / "DIVA-phase-2" / "MEVA"
DEFAULT_SOURCES = ("kitware",)  # evaluation-level: every instance annotated, so negatives are trustworthy
CLIP = re.compile(r"^(\d{4}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.(\d{2}-\d{2}-\d{2})\.([A-Za-z0-9]+)\.(G\d+)$")

# activity -> (question stem, who, extra tags). The stem completes "Did/Was ... <window>?".
TEMPLATES: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "person_carries_heavy_object": ("Did a person carry something heavy", "person", ("carrying",)),
    "person_opens_vehicle_door": ("Did a person open a vehicle door", "person", ("vehicle",)),
    "person_closes_vehicle_door": ("Did a person close a vehicle door", "person", ("vehicle",)),
    "person_exits_vehicle": ("Did a person get out of a vehicle", "person", ("vehicle",)),
    "person_enters_vehicle": ("Did a person get into a vehicle", "person", ("vehicle",)),
    "person_opens_trunk": ("Did a person open the trunk of a vehicle", "person", ("vehicle",)),
    "person_closes_trunk": ("Did a person close the trunk of a vehicle", "person", ("vehicle",)),
    "vehicle_starts": ("Did a vehicle start moving", "vehicle", ()),
    "vehicle_stops": ("Did a vehicle stop", "vehicle", ()),
    "vehicle_turns_left": ("Did a vehicle turn left", "vehicle", ()),
    "vehicle_turns_right": ("Did a vehicle turn right", "vehicle", ()),
    "vehicle_makes_u_turn": ("Did a vehicle make a U-turn", "vehicle", ()),
    "vehicle_reverses": ("Did a vehicle reverse", "vehicle", ()),
    "person_enters_scene_through_structure": ("Did a person walk in through a building entrance", "person", ("place",)),
    "person_exits_scene_through_structure": ("Did a person walk out through a building entrance", "person", ("place",)),
    "person_opens_facility_door": ("Did a person open a door", "person", ("place",)),
    "person_sits_down": ("Did a person sit down", "person", ()),
    "person_stands_up": ("Did a person stand up", "person", ()),
    "person_texts_on_phone": ("Was a person texting on a phone", "person", ()),
    "person_talks_to_person": ("Were two people talking to each other", "person", ("count",)),
    "person_picks_up_object": ("Did a person pick something up", "person", ()),
    "person_puts_down_object": ("Did a person put something down", "person", ()),
}


@dataclass(frozen=True)
class Clip:
    date: str
    start: str
    end: str
    site: str
    camera: str

    @property
    def name(self) -> str:
        return f"{self.date}.{self.start}.{self.end}.{self.site}.{self.camera}"


@dataclass(frozen=True)
class Instance:
    activity: str
    clip: Clip
    start_frame: int
    end_frame: int


def parse_tz(value: str) -> timezone:
    m = re.fullmatch(r"([+-])(\d{2}):?(\d{2})", value.strip())
    if not m:
        raise ValueError(f"--tz must look like +05:30, got {value!r}")
    delta = timedelta(hours=int(m.group(2)), minutes=int(m.group(3)))
    return timezone(delta if m.group(1) == "+" else -delta)


def parse_clip_name(stem: str) -> Clip | None:
    m = CLIP.match(stem)
    return Clip(*m.groups()) if m else None


def clip_start(clip: Clip, tz: timezone) -> datetime:
    return datetime.strptime(f"{clip.date} {clip.start}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=tz)


def clip_end(clip: Clip, tz: timezone) -> datetime:
    end = datetime.strptime(f"{clip.date} {clip.end}", "%Y-%m-%d %H-%M-%S").replace(tzinfo=tz)
    return end if end > clip_start(clip, tz) else end + timedelta(days=1)  # a clip may cross midnight


def activity_files(root: Path, sources: tuple[str, ...] = DEFAULT_SOURCES) -> list[tuple[Clip, Path]]:
    base = root / ANNOTATION_SUBPATH if (root / ANNOTATION_SUBPATH).is_dir() else root
    found: list[tuple[Clip, Path]] = []
    for source in sources:
        for path in sorted((base / source).rglob("*.activities.yml")):
            clip = parse_clip_name(path.name.removesuffix(".activities.yml"))
            if clip is not None:
                found.append((clip, path))
    return found


def read_instances(clip: Clip, path: Path) -> list[Instance]:
    entries = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    out: list[Instance] = []
    for entry in entries:
        act = entry.get("act") if isinstance(entry, dict) else None
        if not act:
            continue
        names = list((act.get("act2") or {}).keys())
        spans = [s["tsr0"] for s in act.get("timespan") or [] if "tsr0" in s]
        if not names or not spans:
            continue
        out.append(Instance(names[0], clip, int(min(s[0] for s in spans)), int(max(s[1] for s in spans))))
    return out


def window_clips(files: list[tuple[Clip, Path]], date: str, start: str) -> list[tuple[Clip, Path]]:
    return [(c, p) for c, p in files if c.date == date and c.start == start]


def find_windows(files: list[tuple[Clip, Path]], min_cams: int) -> list[tuple[str, str, int, int]]:
    """(date, start, cameras, activity instances), most cameras first."""
    cams: dict[tuple[str, str], set[str]] = defaultdict(set)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    for clip, path in files:
        key = (clip.date, clip.start)
        cams[key].add(clip.camera)
        counts[key] += len(read_instances(clip, path))
    rows = [(d, s, len(c), counts[(d, s)]) for (d, s), c in cams.items() if len(c) >= min_cams]
    return sorted(rows, key=lambda r: (-r[2], -r[3], r[0], r[1]))


def spoken_window(first: datetime, last: datetime) -> str:
    day = f"{first.day} {first.strftime('%B')}"
    return f"on {day} between {first.strftime('%H:%M')} and {last.strftime('%H:%M')}"


def build_queries(
    clips: list[tuple[Clip, Path]],
    tz: timezone,
    split: str = "dev",
    workspace: str = "meva",
    camera_map: dict[str, str] | None = None,
    max_queries: int = 40,
    max_negatives: int = 6,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Positive queries for activities that occur in the window, negatives for ones that do not."""
    if not clips:
        return [], {}
    camera_map = camera_map or {}
    if camera_map:  # only cameras that were ingested can be searched; ground truth elsewhere would be unfair misses
        clips = [(c, p) for c, p in clips if c.camera in camera_map]
        if not clips:
            return [], {}
    first = min(clip_start(c, tz) for c, _ in clips)
    last = max(clip_end(c, tz) for c, _ in clips)
    when = spoken_window(first, last)
    key = first.strftime("%Y%m%d_%H%M")

    by_activity: dict[str, list[Instance]] = defaultdict(list)
    skipped: dict[str, int] = defaultdict(int)
    for clip, path in clips:
        for inst in read_instances(clip, path):
            if inst.activity in TEMPLATES:
                by_activity[inst.activity].append(inst)
            else:
                skipped[inst.activity] += 1

    def item(activity: str, hits: list[dict[str, Any]]) -> dict[str, Any]:
        stem, who, extra = TEMPLATES[activity]
        positive = bool(hits)
        return {
            "id": f"meva_{key}_{activity}",
            "text": f"{stem} {when}?",
            "workspace": workspace,
            "intent": "exists",
            "expected": {"verdict": "yes" if positive else "no", "hits": hits},
            "tags": ["meva", activity, who, "absolute_time", *extra, *([] if positive else ["negative"]),
                     *(["broad"] if len(hits) >= BROAD_HITS else [])],
            "split": split,
        }

    items: list[dict[str, Any]] = []
    for activity, instances in sorted(by_activity.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        hits = []
        for inst in sorted(instances, key=lambda i: (clip_start(i.clip, tz), i.start_frame))[:MAX_HITS]:
            t0 = clip_start(inst.clip, tz) + timedelta(seconds=inst.start_frame / FPS)
            t1 = clip_start(inst.clip, tz) + timedelta(seconds=max(inst.end_frame, inst.start_frame + FPS) / FPS)
            hits.append({"camera_id": camera_map.get(inst.clip.camera, inst.clip.camera),
                         "start": t0.isoformat(timespec="milliseconds"), "end": t1.isoformat(timespec="milliseconds")})
        items.append(item(activity, hits))
    items = items[:max_queries]
    absent = sorted(a for a in TEMPLATES if a not in by_activity)
    items += [item(a, []) for a in absent[:max_negatives]]
    return items, dict(skipped)


def assign_splits(items: list[dict[str, Any]], ratios: dict[str, float]) -> dict[str, int]:
    """Give each activity type to one split so no split shares a question type.

    With a single window of footage, holding out whole activity types is the fairest generalisation
    test available: tuning on `dev` cannot memorise a `test` question. Types are dealt out
    deterministically, richest first, each to the split furthest below its target share, so every
    split gets some of the discriminating queries instead of whatever a hash happens to give it.
    Returns the number of queries per split.
    """
    total = sum(ratios.values())
    share = {name: weight / total for name, weight in ratios.items()}
    by_activity: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_activity[next((t for t in item["tags"] if t in TEMPLATES), item["id"])].append(item)

    def richness(group: list[dict[str, Any]]) -> int:
        return max((len(i["expected"]["hits"]) for i in group), default=0)

    counts = dict.fromkeys(ratios, 0)
    assigned = 0
    for _, group in sorted(by_activity.items(), key=lambda kv: (-richness(kv[1]), kv[0])):
        # the split whose share of what has been dealt so far is furthest under its target
        split = max(ratios, key=lambda n: (share[n] * (assigned + len(group)) - counts[n], share[n]))
        for item in group:
            item["split"] = split
        counts[split] += len(group)
        assigned += len(group)
    return counts


def parse_ratios(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for part in text.split(","):
        name, _, weight = part.partition(":")
        if name not in ("dev", "test", "judge_sim") or not weight:
            raise ValueError(f"--splits expects names from dev,test,judge_sim with weights, got {part!r}")
        out[name] = float(weight)
    return out


def write_yaml(items: list[dict[str, Any]], path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.safe_dump(items, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(f"# {header}\n{body}", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("find", "generate"):
        p = sub.add_parser(name)
        p.add_argument("--annotations", type=Path, default=Path("data/raw/meva-annotations"))
        p.add_argument("--sources", default=",".join(DEFAULT_SOURCES), help="annotation folders, comma separated")
    sub.choices["find"].add_argument("--min-cams", type=int, default=4)
    sub.choices["find"].add_argument("--top", type=int, default=15)
    g = sub.choices["generate"]
    g.add_argument("--date", required=True, help="2018-03-09")
    g.add_argument("--start", required=True, help="window start as in the clip names, e.g. 10-10-00")
    g.add_argument("--split", choices=["dev", "test", "judge_sim"], default="dev")
    g.add_argument("--splits", help="spread activity types over splits, e.g. dev:0.6,test:0.25,judge_sim:0.15")
    g.add_argument("--workspace", default="meva")
    g.add_argument("--tz", default="+05:30")
    g.add_argument("--camera-map", type=Path, help='JSON like {"G340": "cam_01"}')
    g.add_argument("--max-queries", type=int, default=40)
    g.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    files = activity_files(args.annotations, tuple(s for s in args.sources.split(",") if s))
    if not files:
        print(f"No annotation files under {args.annotations}. Clone gitlab.kitware.com/meva/meva-data-repo there.",
              file=sys.stderr)
        return 2
    if args.cmd == "find":
        print("date        start     cameras  instances")
        for date, start, n_cams, n_inst in find_windows(files, args.min_cams)[: args.top]:
            print(f"{date}  {start}  {n_cams:7d}  {n_inst:9d}")
        return 0

    clips = window_clips(files, args.date, args.start)
    if not clips:
        print(f"No annotated clips for {args.date} {args.start}; run `find` to see windows.", file=sys.stderr)
        return 2
    cmap = json.loads(args.camera_map.read_text(encoding="utf-8")) if args.camera_map else None
    items, skipped = build_queries(clips, parse_tz(args.tz), args.split, args.workspace, cmap, args.max_queries)
    if not items:
        print("None of the window's cameras are in --camera-map.", file=sys.stderr)
        return 2
    split_counts = assign_splits(items, parse_ratios(args.splits)) if args.splits else None
    write_yaml(items, args.out, f"generated by scripts/meva_to_queries.py from MEVA {args.date} {args.start}; "
                                f"{len({c.camera for c, _ in clips if not cmap or c.camera in cmap})} cameras, "
                                f"{len(items)} queries")
    positives = sum(1 for i in items if i["expected"]["hits"])
    print(f"{len(items)} queries ({positives} positive, {len(items) - positives} negative) -> {args.out}")
    if split_counts:
        print("queries per split:", ", ".join(f"{k} {v}" for k, v in split_counts.items()))
    if skipped:
        print("skipped activities without a template:", ", ".join(f"{k} x{v}" for k, v in sorted(skipped.items())))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
