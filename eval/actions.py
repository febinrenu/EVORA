"""Actions: what the system answered before and what it answers now, on the same workspace and questions.

The activity questions of eval/queries/meva_school.yaml (get in or out of a vehicle, stop, start, turn, reverse, talk,
pick up, open a door ...) are run under three router settings:

  plain       the original behaviour: an action question is answered like any other, with whoever matched
  estimates   an action nobody recognises is answered "I can't tell", with the people nearest to fitting it shown
  detected    the actions perception detects are answered from its events; the others get "I can't verify that" and
              no evidence (the current default)

    python -m eval.actions --workspace meva-school --root data/share/ws-v7a

The workspace needs the action events in its `events` table (perception writes them at ingest).
"""
from __future__ import annotations

import argparse
import asyncio
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from eval.harness import REPORTS_DIR, run_system
from eval.metrics import score
from eval.queries import load_queries

QUERIES = Path(__file__).resolve().parent / "queries" / "meva_school.yaml"
SETTINGS: dict[str, dict[str, Any]] = {
    "plain": {"honest_actions": False, "detected_actions": False, "show_unverified_actions": True},
    "estimates": {"honest_actions": True, "detected_actions": False, "show_unverified_actions": True},
    "detected": {"honest_actions": True, "detected_actions": True, "show_unverified_actions": False},
}
COLUMNS = [("hit@1", "Hit@1"), ("hit@1_strict", "Strict Hit@1"), ("hit@5", "Hit@5"), ("negative_precision", "Neg prec"),
           ("existence_accuracy", "Existence acc"), ("abstain_rate", "Abstained"),
           ("existence_accuracy_answered", "Acc when answered")]


async def run_setting(name: str, workspace: str, root: Path | None, items: list) -> dict[str, Any]:
    from eval.systems import build_ours

    system = build_ours(workspace, root)
    system.ctx.router.cfg = replace(system.ctx.router.cfg, **SETTINGS[name])
    try:
        results = await run_system(system, items)
    finally:
        await system.aclose()
    report = score(items, results, split="all")
    per_query = []
    for item, result in zip(items, results, strict=True):
        answer = result.answer
        per_query.append({"id": item.id, "text": item.text, "expected": item.expected.verdict,
                          "verdict": answer.verdict if answer else None, "evidence": len(answer.evidence) if answer else 0,
                          "unsupported": answer.unsupported_action if answer else None,
                          "asked": result.asked_clarify})
    return {"metrics": {k: {"value": m.value, "n": m.n} for k, m in report.metrics.items()}, "queries": per_query}


def event_quality(db_path: Path, kinds: tuple[str, ...], hits: list, slack_s: float = 2.0) -> dict[str, Any]:
    """The detector against the labels: how many detected events fall in a labelled window, and how many labelled windows
    have a detected event (inside it, widened by the slack). Camera ids must match."""
    import sqlite3

    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        marks = ",".join("?" * len(kinds))
        events = conn.execute(f"SELECT camera_id, t FROM events WHERE kind IN ({marks})", kinds).fetchall()
    finally:
        conn.close()
    windows = [(h.camera_id, h.start, h.end) for h in hits]
    inside = sum(1 for cam, t in events if any(cam == c and s - slack_s <= t <= e + slack_s for c, s, e in windows))
    covered = sum(1 for c, s, e in windows if any(cam == c and s - slack_s <= t <= e + slack_s for cam, t in events))
    def gap(t: float, cam: str) -> float | None:
        """Signed seconds from t to the nearest labelled window on that camera (0 inside, negative before it)."""
        gaps = [0.0 if s <= t <= e else (t - s if t < s else t - e) for c, s, e in windows if c == cam]
        return min(gaps, key=abs) if gaps else None

    gaps = [g for cam, t in events if (g := gap(t, cam)) is not None]
    median_gap = sorted(abs(g) for g in gaps)[len(gaps) // 2] if gaps else None
    return {"events": len(events), "labelled_windows": len(windows),
            "precision": round(inside / len(events), 3) if events else None,
            "recall": round(covered / len(windows), 3) if windows else None,
            "median_seconds_from_a_labelled_window": round(median_gap, 1) if median_gap is not None else None,
            "events_on_cameras_with_no_label": sum(1 for cam, _ in events if all(c != cam for c, _, _ in windows))}


def table(results: dict[str, dict[str, Any]]) -> str:
    header = "| Setting | n | " + " | ".join(label for _, label in COLUMNS) + " |"
    lines = [header, "|" + "---|" * (len(COLUMNS) + 2)]
    for name in SETTINGS:
        data = results[name]
        cells = []
        for key, _ in COLUMNS:
            m = data["metrics"].get(key)
            cells.append("n/a" if not m or m["value"] is None else f"{m['value']:.2f}")
        lines.append(f"| {name} | {len(data['queries'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="meva-school")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--queries", type=Path, default=QUERIES)
    parser.add_argument("--out", type=Path, default=REPORTS_DIR / "actions.json")
    args = parser.parse_args(argv)
    items = load_queries([args.queries])
    results = {name: asyncio.run(run_setting(name, args.workspace, args.root, items)) for name in SETTINGS}
    from evora.query.actions import detected_action
    from evora.query.logic import ACTION_EVENT_KINDS

    db_path = (args.root or Path("workspaces")) / args.workspace / "evora.sqlite"
    quality = {}
    for item in items:
        about_vehicles = "vehicle" in item.text.lower() and "person" not in item.text.lower()
        found = detected_action(item.text, ["car"] if about_vehicles else [])
        if found is not None:
            quality[item.id] = {"action": found.key, **event_quality(db_path, ACTION_EVENT_KINDS[found.key], item.expected.hits)}
    results["detector_vs_labels"] = quality
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(table(results))
    print("\nper question (expected -> verdict, evidence items):")
    for i, q in enumerate(results["detected"]["queries"]):
        row = " | ".join(f"{name[:5]} {results[name]['queries'][i]['verdict']}/{results[name]['queries'][i]['evidence']}"
                         for name in SETTINGS)
        print(f"  {q['expected']:>3} -> {row} | {q['text'][:70]}")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
