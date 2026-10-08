"""Evaluation harness: run a system over ground-truth queries and report section 9.3 metrics.

A system is anything with `name` and `async run(item) -> RunResult`: our full pipeline,
baseline B0 or B1, or an ablation of ours. `make eval` runs all of them on the same
queries, same footage and same machine.

    python -m eval.harness --system ours --split dev
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from eval.metrics import Report, RunResult, score
from eval.queries import QueryItem, Split, load_queries

REPORTS_DIR = Path(__file__).resolve().parent / "reports"

# (display name, metric key, format). Order is the order of columns in the scoreboard.
HEADLINE = [
    ("Hit@1", "hit@1", "{:.2f}"), ("Hit@5", "hit@5", "{:.2f}"), ("MRR", "mrr", "{:.2f}"),
    ("Cam acc", "camera_accuracy", "{:.2f}"), ("Ts err (s)", "timestamp_error_s", "{:.1f}"),
    ("Neg prec", "negative_precision", "{:.2f}"), ("Re-asks", "reask_count", "{:.0f}"),
    ("TTFA p50 (ms)", "ttfa_p50_ms", "{:.0f}"), ("No-LLM share", "no_model_share", "{:.2f}"),
]


class System(Protocol):
    name: str

    async def run(self, item: QueryItem) -> RunResult: ...


class SystemNotReady(RuntimeError):
    """A named system exists in the plan but is not wired up yet."""


async def run_system(system: System, items: list[QueryItem],
                     on_result: Callable[[QueryItem, RunResult], None] | None = None) -> list[RunResult]:
    """Run queries one at a time (clarify state and latency numbers must not interleave).

    A crash in one query is recorded as an error result and never stops the run.
    """
    results: list[RunResult] = []
    for item in items:
        started = time.perf_counter()
        try:
            result = await system.run(item)
        except Exception as exc:  # noqa: BLE001 - any failure of the system under test is a result, not a crash
            result = RunResult(item.id, None, error=f"{type(exc).__name__}: {exc}")
        elapsed_ms = (time.perf_counter() - started) * 1000
        if result.ttva_ms is None:
            result.ttva_ms = elapsed_ms
        if result.ttfa_ms is None:
            result.ttfa_ms = result.ttva_ms
        results.append(result)
        if on_result:
            on_result(item, result)
    return results


def _cell(report: Report, key: str, fmt: str) -> str:
    metric = report.metrics.get(key)
    return "n/a" if metric is None or metric.value is None else fmt.format(metric.value)


def format_markdown(reports: dict[str, Report]) -> str:
    """One row per system, with the number of queries behind it."""
    header = ["System", "Split", "n", *[h[0] for h in HEADLINE]]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for name, report in reports.items():
        row = [name, report.split, str(report.n_queries),
               *[_cell(report, key, fmt) for _, key, fmt in HEADLINE]]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def scoreboard_row(report: Report, commit: str, baseline_hit1: float | None = None, notes: str = "") -> str:
    """A line for the PROGRESS.md eval scoreboard (append only)."""
    def val(key: str, fmt: str = "{:.2f}") -> str:
        return _cell(report, key, fmt)

    base = "n/a" if baseline_hit1 is None else f"{baseline_hit1:.2f}"
    stamp = time.strftime("%H:%M")
    return (f"| {stamp} | {commit} | {report.split} | {val('hit@1')} | {val('hit@5')} | {val('mrr')} | "
            f"{val('camera_accuracy')} | {val('timestamp_error_s', '{:.1f}')} | {val('negative_precision')} | "
            f"{val('reask_count', '{:.0f}')} | {val('ttfa_p50_ms', '{:.0f}')} | {base} | {notes} |")


def write_reports(reports: dict[str, Report], out_dir: Path = REPORTS_DIR, stem: str = "eval") -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, md_path = out_dir / f"{stem}.json", out_dir / f"{stem}.md"
    json_path.write_text(json.dumps({k: r.to_dict() for k, r in reports.items()}, indent=2), encoding="utf-8")
    md_path.write_text(format_markdown(reports) + "\n", encoding="utf-8")
    return json_path, md_path


def build_system(name: str, workspace: str | None = None, root: Path | None = None, replay: Path | None = None,
                 overrides: dict | None = None) -> System:
    """Resolve a system by name against an indexed workspace."""
    from eval.systems import SYSTEM_BUILDERS

    builder = SYSTEM_BUILDERS.get(name)
    if builder is None:
        raise SystemNotReady(f"unknown system {name!r}; known: {', '.join(sorted(SYSTEM_BUILDERS))}")
    if workspace is None:
        raise SystemNotReady(f"system {name!r} needs an indexed workspace: pass --workspace <slug>")
    kwargs: dict = {"root": root, "replay_path": replay}
    if name == "ours":
        kwargs["overrides"] = overrides
    return builder(workspace, **kwargs)


async def evaluate(system: System, items: list[QueryItem], split: str) -> Report:
    return score(items, await run_system(system, items), split=split)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", action="append", required=True, help="ours, b0, b1 (repeatable)")
    parser.add_argument("--split", choices=["dev", "test", "judge_sim"], default="dev")
    parser.add_argument("--queries", type=Path, action="append", help="query file(s); default eval/queries/*.yaml")
    parser.add_argument("--out", type=Path, default=REPORTS_DIR)
    parser.add_argument("--workspace", help="slug of the indexed workspace to evaluate on")
    parser.add_argument("--root", type=Path, help="workspaces folder (default: workspaces/)")
    parser.add_argument("--replay", type=Path, help="record and replay model calls here, so reruns cost no Groq calls")
    args = parser.parse_args(argv)

    split: Split = args.split
    items = load_queries(args.queries, split=split)
    if not items:
        print(f"No {split} queries found.", file=sys.stderr)
        return 2
    reports: dict[str, Report] = {}
    for name in args.system:
        try:
            system = build_system(name, args.workspace, args.root, args.replay)
        except SystemNotReady as exc:
            print(str(exc), file=sys.stderr)
            return 2
        reports[name] = asyncio.run(evaluate(system, items, split))
    json_path, md_path = write_reports(reports, args.out, stem=f"eval_{split}")
    print(format_markdown(reports))
    print(f"\nwrote {json_path} and {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
