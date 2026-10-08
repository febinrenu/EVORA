"""Assemble the evaluation results into one `report.json` for the report page and the write-up.

Inputs are the files the harness and the ablation runner already write to `eval/reports/`. The output keeps
every number next to the number of queries behind it, pools the same capability across splits (weighted by n,
never an average of averages), lists what was not evaluated and why, and carries the caveats with the data so
a page cannot show the wins without the limits.

    python -m eval.report                # writes eval/reports/report.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any

from eval.harness import UNSUPPORTED_CAPABILITIES

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
FROZEN_FILE = Path(__file__).resolve().parent / "frozen.json"
SPLITS = ("dev", "test", "judge_sim")
# capability -> the metrics worth pooling across splits
POOLED = {
    "object": ["hit@1", "hit@5", "mrr", "camera_accuracy"],
    "negative": ["negative_precision"],
}
# statements that hold whatever the numbers are; the data decides which capabilities are listed as not evaluated
LIMITS = [
    "Ground truth is MEVA's: it labels only actors that take part in annotated activities, so a correct but "
    "unlabelled object counts as a miss. Object scores are conservative; negatives are built only for classes "
    "absent from a camera's whole annotation.",
    "Sample sizes are small (one 5-minute window, 8 cameras). Read every number together with its n.",
    "The judge_sim split shares the recording day with dev and test: it checks unseen cameras, not unseen footage.",
    "Thresholds were frozen before the test and judge_sim runs; nothing was tuned on them.",
]
NOT_SHOWN = [
    "Object retrieval better than chance: a random moment inside the same camera and window (the null system) scores "
    "as well as or better than our object Hit@1 on every split, and the frame-similarity baseline is the only "
    "system clearly above it.",
    "Better timestamp localisation than chance or the frame baseline: after the time-window boundary fix the "
    "differences are within noise, and the random baseline is not worse.",
    "A benefit of track-centric indexing: frame-level retrieval is better than the full system in the ablation. "
    "The track layer is limited by detector recall on small, distant people.",
    "Any effect of attribute scoring, verification or clarify-once memory: no evaluated query exercises them.",
]
SUPPORTED = [
    "When no object of the requested class is on the camera, the system says so. The frame-similarity baseline and "
    "the random baseline always return something and never answer 'nothing there'.",
    "Parsing the question into class, camera and time constraints is what makes the system work at all: replacing "
    "the parsed plan with raw-text search collapses both object hits and negatives in the ablation.",
]


def _load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def _git_head() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5,
                             cwd=Path(__file__).resolve().parent, check=False)
        return out.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def pool(per_split: dict[str, dict[str, Any]], system: str, capability: str, metric: str) -> dict[str, Any]:
    """Weighted pool of one capability metric over the splits that have it: sum(value * n) / sum(n)."""
    total = n_total = 0.0
    used: list[str] = []
    for split, caps in per_split.items():
        m = (((caps.get(system) or {}).get(capability) or {}).get("metrics") or {}).get(metric)
        if m and m.get("value") is not None and m.get("n"):
            total += m["value"] * m["n"]
            n_total += m["n"]
            used.append(split)
    return {"value": round(total / n_total, 4) if n_total else None, "n": int(n_total), "splits": used}


def build_report(reports_dir: Path = REPORTS_DIR, frozen_file: Path = FROZEN_FILE) -> dict[str, Any]:
    splits: dict[str, Any] = {}
    cap_by_split: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        overall = _load(reports_dir / f"eval_{split}.json")
        caps = _load(reports_dir / f"eval_{split}_capabilities.json")
        if overall is None and caps is None:
            continue
        splits[split] = {"overall": overall or {}, "capabilities": caps or {}}
        cap_by_split[split] = caps or {}
    systems = sorted({s for caps in cap_by_split.values() for s in caps})
    evaluated = sorted({c for caps in cap_by_split.values() for sys_caps in caps.values() for c in sys_caps})

    pooled: dict[str, Any] = {}
    for system in systems:
        pooled[system] = {cap: {metric: pool(cap_by_split, system, cap, metric) for metric in metrics}
                          for cap, metrics in POOLED.items()}

    limits = list(LIMITS)
    chance = pooled.get("null", {}).get("object", {}).get("hit@1", {})
    if chance.get("value") is not None:
        limits.append(
            f"Chance level: a random moment inside the same camera and window already scores object Hit@1 "
            f"{chance['value']:.2f} (n={chance['n']}), because MEVA's annotated actors are dense in time. "
            "Read every object result next to this row."
        )
    frozen_v1 = reports_dir / "frozen_v1"
    if frozen_v1.is_dir():
        v1 = {}
        for split in SPLITS:
            caps = _load(frozen_v1 / f"eval_{split}_capabilities.json")
            if caps:
                v1[split] = caps
        pooled_v1 = {sys_: {cap: {m: pool(v1, sys_, cap, m) for m in metrics} for cap, metrics in POOLED.items()}
                     for sys_ in sorted({x for caps in v1.values() for x in caps})}
    else:
        pooled_v1 = {}

    ablation_raw = _load(reports_dir / "ablation.json") or {}
    ablations = [
        {"label": label, "switch": row.get("switch"), "skipped": row.get("skipped"),
         "metrics": ((row.get("report") or {}).get("metrics")), "n_queries": (row.get("report") or {}).get("n_queries")}
        for label, row in ablation_raw.items()
    ]
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "code_commit": _git_head(),
        "frozen": _load(frozen_file),
        "splits": splits,
        "pooled": pooled,
        "pooled_as_first_frozen": pooled_v1,
        "ablations": ablations,
        "capabilities_evaluated": [c for c in evaluated if not c.startswith("activity")],
        "diagnostics": [c for c in evaluated if c.startswith("activity")],
        "not_evaluated": {c: why for c, why in UNSUPPORTED_CAPABILITIES.items() if c not in evaluated},
        "supported_claims": SUPPORTED,
        "not_shown": NOT_SHOWN,
        "limits": limits,
    }


def load_report(path: Path | None = None) -> dict[str, Any]:
    """What GET /api/report should return: the saved report, or the empty shape before any evaluation exists."""
    saved = _load(path or REPORTS_DIR / "report.json")
    return saved if saved is not None else {"eval": None, "ablations": None}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--reports", type=Path, default=REPORTS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    report = build_report(args.reports)
    if not report["splits"]:
        print(f"No evaluation results in {args.reports}; run `make eval` first.")
        return 2
    out = args.out or args.reports / "report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    for system, caps in report["pooled"].items():
        for cap, metrics in caps.items():
            shown = ", ".join(f"{k} {v['value']} (n={v['n']})" for k, v in metrics.items() if v["value"] is not None)
            if shown:
                print(f"  pooled {system} {cap}: {shown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
