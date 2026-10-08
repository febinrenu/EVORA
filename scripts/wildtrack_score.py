"""Score tracking and identity linking against WILDTRACK ground truth (7 cameras, about 20 people with ids).

Usage (from backend/, after `python -m evora.perception.cli ingest data/norm/wildtrack/c1.mp4 ... --workspace wildtrack
--names C1 ... C7 --layers L0 L1 L2 --t0 2026-01-01T00:00:00+00:00`):

    PYTHONPATH=".;.." python ../scripts/wildtrack_score.py wildtrack
    PYTHONPATH=".;.." python ../scripts/wildtrack_score.py wildtrack --set reid_recluster_q=0.95 --set reid_recluster=false

The frames are 2 per second, annotation file N belongs to video time N/10 s. Each track is given the ground-truth person
it overlaps most (IoU 0.4 or more on at least half of its matched points). Reported per camera and overall:

* recall: ground-truth boxes covered by some person track box;
* tracks per person (fragmentation) before and after identity linking;
* wrong merges: identities that hold tracks of two or more different people;
* cross-camera link precision: pairs of tracks from different cameras that were joined and really are one person.

Linking is recomputed in memory with the settings given by --set; the database is not changed.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ANNOTATIONS = Path(__file__).resolve().parent.parent / "data/raw/wildtrack/extract/Wildtrack_dataset/annotations_positions"
WIDTH, HEIGHT = 1920.0, 1080.0
IOU_MIN = 0.4


def load_ground_truth(root: Path) -> dict[int, dict[int, list[tuple[int, tuple[float, float, float, float]]]]]:
    """{view: {frame_time_s: [(person_id, (x1, y1, x2, y2))]}} with video time = annotation number / 10."""
    gt: dict[int, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for path in sorted(root.glob("*.json")):
        t = int(path.stem) / 10.0
        for person in json.loads(path.read_text(encoding="utf-8")):
            for v in person["views"]:
                if v["xmin"] >= 0 and v["xmax"] > v["xmin"] and v["ymax"] > v["ymin"]:
                    gt[v["viewNum"]][t].append((person["personID"], (v["xmin"], v["ymin"], v["xmax"], v["ymax"])))
    return gt


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def parse_overrides(pairs: list[str]) -> dict:
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        low = raw.lower()
        out[key] = True if low == "true" else False if low == "false" else (float(raw) if "." in raw or "e" in low else int(raw))
    return out


def label_tracks(db, gt) -> tuple[dict[str, int | None], dict, dict, int, int]:
    """Ground-truth person for each track: (person_of, cameras {name: (id, t0)}, spans, hit boxes, total boxes)."""
    with db.read() as c:
        cams = {r["name"]: (r["id"], r["t0"]) for r in c.execute("SELECT id, name, t0 FROM cameras")}
        points = defaultdict(list)
        for r in c.execute("SELECT p.track_id, p.t, p.x1, p.y1, p.x2, p.y2, k.camera_id FROM track_points p "
                           "JOIN tracks k ON k.id = p.track_id WHERE k.cls = 'person' ORDER BY p.t"):
            points[r["track_id"]].append((r["t"], r["x1"] * WIDTH, r["y1"] * HEIGHT, r["x2"] * WIDTH, r["y2"] * HEIGHT,
                                          r["camera_id"]))
        spans = {r["id"]: r["t_end"] - r["t_start"] for r in c.execute("SELECT id, t_start, t_end FROM tracks")}
    person_of: dict[str, int | None] = {}
    total_gt = hit_gt = 0
    for name, (cid, t0) in sorted(cams.items()):
        view = int(name.lstrip("Cc")) - 1
        frames = gt.get(view, {})
        by_time: dict[float, list] = defaultdict(list)
        for tid, pts in points.items():
            if pts[0][5] != cid:
                continue
            for t, x1, y1, x2, y2, _ in pts:
                by_time[round(t - t0, 1)].append((tid, (x1, y1, x2, y2)))
        votes: dict[str, Counter] = defaultdict(Counter)
        for ft, boxes in frames.items():
            near = by_time.get(round(ft, 1), []) + by_time.get(round(ft + 0.1, 1), []) + by_time.get(round(ft - 0.1, 1), [])
            for pid, gbox in boxes:
                total_gt += 1
                best, best_tid = 0.0, None
                for tid, tbox in near:
                    v = iou(gbox, tbox)
                    if v > best:
                        best, best_tid = v, tid
                if best >= IOU_MIN:
                    hit_gt += 1
                    votes[best_tid][pid] += 1
        for tid, cnt in votes.items():
            pid, n = cnt.most_common(1)[0]
            if n >= 0.5 * sum(cnt.values()):
                person_of[tid] = pid
    # tracks without any matched ground truth are unlabelled people or false detections: not counted either way
    return person_of, cams, spans, hit_gt, total_gt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workspace")
    ap.add_argument("--annotations", type=Path, default=ANNOTATIONS)
    ap.add_argument("--set", action="append", default=[], help="IngestSettings override, e.g. reid_recluster_q=0.95")
    ap.add_argument("--min-track-s", type=float, default=0.0, help="ignore tracks shorter than this when scoring")
    args = ap.parse_args(argv)

    from evora.core.db import open_db
    from evora.perception.pipeline import resolve_workspace
    from evora.perception.settings import IngestSettings
    from evora.reid.associate import group_tracks, link_tracks, load_tracks

    ws = resolve_workspace(args.workspace)
    db = open_db(ws.db_path)
    st = IngestSettings(**parse_overrides(args.set))
    gt = load_ground_truth(args.annotations)
    person_of, cams, spans, hit_gt, total_gt = label_tracks(db, gt)

    tracks = [t for t in load_tracks(db, ws, st) if t.cls == "person" and spans.get(t.id, 0) >= args.min_track_s]
    clusters, _ = link_tracks(tracks, st)
    identity_of = {t.id: i for i, members in enumerate(clusters) for t in members}

    print(f"settings: {args.set or 'defaults'}  tracks scored: {len(tracks)}")
    print(f"detection recall (ground-truth boxes with a person track box, IoU>={IOU_MIN}): "
          f"{hit_gt}/{total_gt} = {hit_gt / max(1, total_gt):.0%}")
    print(f"{'camera':8}{'people':>8}{'tracks':>8}{'tr/person':>10}{'identities':>11}{'id/person':>10}{'wrong merges':>13}")
    sum_t = sum_i = sum_p = sum_bad = 0
    for name, (cid, _) in sorted(cams.items()):
        mine = [t for t in tracks if t.camera_id == cid and person_of.get(t.id) is not None]
        people = {person_of[t.id] for t in mine}
        idents = {identity_of[t.id] for t in mine}
        mixed = sum(1 for i in idents if len({person_of[t.id] for t in mine if identity_of[t.id] == i}) > 1)
        ratio_t = len(mine) / max(1, len(people))
        ratio_i = len(idents) / max(1, len(people))
        print(f"{name:8}{len(people):>8}{len(mine):>8}{ratio_t:>10.1f}{len(idents):>11}{ratio_i:>10.1f}{mixed:>13}")
        sum_t, sum_i, sum_p, sum_bad = sum_t + len(mine), sum_i + len(idents), sum_p + len(people), sum_bad + mixed
    print(f"{'all':8}{sum_p:>8}{sum_t:>8}{sum_t / max(1, sum_p):>10.1f}{sum_i:>11}{sum_i / max(1, sum_p):>10.1f}{sum_bad:>13}")

    pairs = good = 0
    for members in clusters:
        labelled = [t for t in members if person_of.get(t.id) is not None]
        for a in range(len(labelled)):
            for b in range(a + 1, len(labelled)):
                if labelled[a].camera_id != labelled[b].camera_id:
                    pairs += 1
                    good += person_of[labelled[a].id] == person_of[labelled[b].id]
    print(f"cross-camera links between labelled tracks: {pairs}, joining the same person: {good / max(1, pairs):.0%}")
    ids = [t.id for t in tracks if person_of.get(t.id) is not None]
    pid = np.array([person_of[i] for i in ids])
    cid_ = np.array([identity_of[i] for i in ids])
    same_person = pid[:, None] == pid[None, :]
    same_identity = cid_[:, None] == cid_[None, :]
    iu = np.triu_indices(len(ids), 1)
    both = int((same_person & same_identity)[iu].sum())
    precision = both / max(1, int(same_identity[iu].sum()))
    recall = both / max(1, int(same_person[iu].sum()))
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    print(f"track pairs: merge precision {precision:.0%}, merge recall {recall:.0%}, F1 {f1:.2f}")
    groups = group_tracks(tracks, clusters, st)
    group_of = {t.id: i for i, members in enumerate(groups) for t in members}
    print(f"{'count groups (what a count uses)':34}{'people':>8}{'groups':>8}{'g/person':>10}{'mixed':>7}")
    tot_p = tot_g = tot_m = 0
    for _, (cid, _t0) in sorted(cams.items()):
        mine = [t for t in tracks if t.camera_id == cid and person_of.get(t.id) is not None]
        people = {person_of[t.id] for t in mine}
        gs = {group_of[t.id] for t in mine}
        mixed = sum(1 for g in gs if len({person_of[t.id] for t in mine if group_of[t.id] == g}) > 1)
        tot_p, tot_g, tot_m = tot_p + len(people), tot_g + len(gs), tot_m + mixed
    print(f"{'all cameras':34}{tot_p:>8}{tot_g:>8}{tot_g / max(1, tot_p):>10.2f}{tot_m:>7}")
    people_all = {p for p in person_of.values() if p is not None}
    all_ident = {identity_of[t] for t in person_of if t in identity_of}
    print(f"whole scene: {len(people_all)} labelled people, {len(all_ident)} identities in all cameras together")
    return 0


if __name__ == "__main__":
    sys.exit(main())
