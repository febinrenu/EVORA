"""Ablations (PLAN section 4 and 9.5): one run per switch, compared with the full system.

The full system runs once; then each contribution is switched off through its config flag
and the same queries run again. A system factory turns {config key: value} overrides into a
system, so this module knows nothing about how the pipeline is built. Switches that change
what is stored (motion gate, topology) need the footage re-indexed, which only the caller
can do: pass `reindex`, otherwise those rows are reported as skipped rather than guessed.

    python -m eval.ablate --split test      (once `ours` is registered in eval.harness)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eval.harness import REPORTS_DIR, System, build_system, run_system
from eval.metrics import Report, score
from eval.queries import QueryItem, load_queries

SystemFactory = Callable[[dict[str, Any]], System]
Reindex = Callable[[dict[str, Any]], Awaitable[None]]


@dataclass(frozen=True)
class Switch:
    id: str
    contribution: str
    key: str                  # config key, dotted
    full: Any
    ablated: Any
    metric: str               # the metric this contribution is meant to move
    needs_reindex: bool = False
    owner: str = "M3"


SWITCHES: list[Switch] = [
    Switch("C1", "Track-centric multi-granular retrieval", "retrieval.unit", "track", "frame", "hit@1"),
    Switch("C2", "Attribute grounding", "retrieval.attributes", True, False, "hit@1"),
    Switch("C3a", "Plan-then-verify: LLM planner", "query.planner", "llm", "raw", "hit@1"),
    Switch("C3b", "Plan-then-verify: verification", "query.verify", True, False, "negative_precision"),
    Switch("C4", "Clarify-once referent memory", "memory.alias_embed", True, False, "reask_count", owner="M1"),
    Switch("C5", "Topology-aware cross-camera linking", "reid.topology", True, False, "path_hop_accuracy",
           needs_reindex=True, owner="M2"),
    Switch("C6", "Motion-gated adaptive sampling", "ingest.motion_gate", True, False, "hit@5",
           needs_reindex=True, owner="M2"),
]

COLUMNS = [("Hit@1", "hit@1", "{:.2f}"), ("Strict Hit@1", "hit@1_strict", "{:.2f}"), ("Hit@5", "hit@5", "{:.2f}"),
           ("MRR", "mrr", "{:.2f}"),
           ("Neg prec", "negative_precision", "{:.2f}"), ("TTFA p50 (ms)", "ttfa_p50_ms", "{:.0f}")]


@dataclass
class Row:
    label: str
    switch: Switch | None
    report: Report | None
    skipped: str | None = None


def _value(report: Report | None, key: str) -> float | None:
    metric = report.metrics.get(key) if report else None
    return None if metric is None else metric.value


async def run_ablation(
    factory: SystemFactory,
    items: list[QueryItem],
    split: str,
    switches: list[Switch] | None = None,
    reindex: Reindex | None = None,
) -> list[Row]:
    rows = [Row("full system", None, score(items, await run_system(factory({}), items), split))]
    for sw in switches if switches is not None else SWITCHES:
        label = f"no {sw.contribution} ({sw.id})"
        if sw.needs_reindex and reindex is None:
            rows.append(Row(label, sw, None, "needs the footage re-indexed with the switch off"))
            continue
        overrides = {sw.key: sw.ablated}
        try:
            if sw.needs_reindex and reindex is not None:
                await reindex(overrides)
            report = score(items, await run_system(factory(overrides), items), split)
            rows.append(Row(label, sw, report))
        except Exception as exc:  # noqa: BLE001 - one broken ablation must not hide the others
            rows.append(Row(label, sw, None, f"{type(exc).__name__}: {exc}"))
        finally:
            if sw.needs_reindex and reindex is not None:
                await reindex({})  # restore the full index for the next row
    return rows


def format_markdown(rows: list[Row]) -> str:
    full = rows[0].report
    header = ["Configuration", "n", *[c[0] for c in COLUMNS], "Contribution metric (change)"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for row in rows:
        if row.report is None:
            lines.append(f"| {row.label} | - | " + " | ".join("-" for _ in COLUMNS) + f" | skipped: {row.skipped} |")
            continue
        cells = []
        for _, key, fmt in COLUMNS:
            v = _value(row.report, key)
            cells.append("n/a" if v is None else fmt.format(v))
        special = "-"
        if row.switch is not None:
            now, base = _value(row.report, row.switch.metric), _value(full, row.switch.metric)
            if now is not None and base is not None:
                special = f"{row.switch.metric} {now:.2f} ({now - base:+.2f})"
            else:
                special = f"{row.switch.metric} n/a"
        lines.append(f"| {row.label} | {row.report.n_queries} | " + " | ".join(cells) + f" | {special} |")
    return "\n".join(lines)


def to_json(rows: list[Row]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for row in rows:
        out[row.label] = {
            "switch": None if row.switch is None else {"id": row.switch.id, "key": row.switch.key,
                                                        "full": row.switch.full, "ablated": row.switch.ablated,
                                                        "metric": row.switch.metric, "owner": row.switch.owner},
            "skipped": row.skipped,
            "report": row.report.to_dict() if row.report else None,
        }
    return out


def write_reports(rows: list[Row], out_dir: Path = REPORTS_DIR) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    jp, mp = out_dir / "ablation.json", out_dir / "ablation.md"
    jp.write_text(json.dumps(to_json(rows), indent=2), encoding="utf-8")
    mp.write_text(format_markdown(rows) + "\n", encoding="utf-8")
    return jp, mp


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", choices=["dev", "test", "judge_sim"], default="test")
    parser.add_argument("--queries", type=Path, action="append")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    parser.add_argument("--workspace", required=False, help="slug of the indexed workspace")
    parser.add_argument("--root", type=Path)
    parser.add_argument("--replay", type=Path, help="shared replay file: ablations that plan alike reuse each other")
    args = parser.parse_args(argv)
    items = load_queries(args.queries, split=args.split)
    if not items:
        print(f"No {args.split} queries found.", file=sys.stderr)
        return 2
    try:
        build_system("ours", args.workspace, args.root, args.replay)
    except Exception as exc:  # noqa: BLE001 - report whatever stops the run, once, clearly
        print(str(exc), file=sys.stderr)
        return 2
    rows = asyncio.run(run_ablation(
        lambda overrides: build_system("ours", args.workspace, args.root, args.replay, overrides), items, args.split))
    jp, mp = write_reports(rows, args.out)
    print(format_markdown(rows))
    print(f"\nwrote {jp} and {mp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
