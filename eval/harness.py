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


# Capabilities the system claims, in table order; anything without a `cap:` tag is the activity diagnostic.
CAPABILITIES = ["object", "negative", "count", "colour", "carrying", "zone", "path"]
ACTIVITY = "activity (diagnostic)"
# Shown as rows so a missing score is visible rather than silently absent. They flip when ground truth exists.
UNSUPPORTED_CAPABILITIES = {
    "count": "MEVA labels only actors in annotated activities (parked cars and bystanders are unlabelled), so "
             "annotated counts are lower bounds and exact counts cannot be scored",
    "colour": "no colour labels in MEVA; needs the blind human labels from scripts/colour_label_tool.py",
    "carrying": "no bag-carrying ground truth: MEVA's carried objects are class 'other', one bag actor in 6 cameras",
    "path": "no cross-camera identity ground truth (MEVA actor ids are per clip and camera)",
}
CAPABILITY_COLUMNS = [
    ("Hit@1", "hit@1", "{:.2f}"), ("Hit@5", "hit@5", "{:.2f}"), ("MRR", "mrr", "{:.2f}"),
    ("Cam acc", "camera_accuracy", "{:.2f}"), ("Ts err (s)", "timestamp_error_s", "{:.1f}"),
    ("Neg prec", "negative_precision", "{:.2f}"), ("Count acc", "count_accuracy", "{:.2f}"),
    ("Count MAE", "count_mae", "{:.2f}"), ("Within 1", "count_within_1", "{:.2f}"),
]


def capability_of(item: QueryItem) -> str:
    return next((t[4:] for t in item.tags if t.startswith("cap:")), ACTIVITY)


def by_capability(items: list[QueryItem], results: list[RunResult], split: str = "all") -> dict[str, Report]:
    """One report per capability present in `items`, so a mixed set never produces one blended headline."""
    groups: dict[str, list[QueryItem]] = {}
    for item in items:
        groups.setdefault(capability_of(item), []).append(item)
    order = [c for c in [*CAPABILITIES, ACTIVITY] if c in groups]
    return {c: score(groups[c], results, split=split) for c in order}


def format_capability_markdown(reports: dict[str, Report], system: str = "ours") -> str:
    """Rows only where valid ground truth exists; unsupported capabilities are listed with the reason."""
    header = ["Capability", "n", *[c[0] for c in CAPABILITY_COLUMNS]]
    lines = [f"System: {system}", "", "| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for cap, report in reports.items():
        cells = [_cell(report, key, fmt) for _, key, fmt in CAPABILITY_COLUMNS]
        lines.append("| " + " | ".join([cap, str(report.n_queries), *cells]) + " |")
    missing = [c for c in UNSUPPORTED_CAPABILITIES if c not in reports]
    if missing:
        lines += ["", "Not evaluated (no reliable ground truth in the available data):"]
        lines += [f"- {c}: {UNSUPPORTED_CAPABILITIES[c]}" for c in missing]
    return "\n".join(lines)


async def evaluate_detailed(system: System, items: list[QueryItem], split: str) -> tuple[Report, list[RunResult]]:
    results = await run_system(system, items)
    return score(items, results, split=split), results


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
    per_capability: dict[str, dict[str, Report]] = {}
    for name in args.system:
        try:
            system = build_system(name, args.workspace, args.root, args.replay)
        except SystemNotReady as exc:
            print(str(exc), file=sys.stderr)
            return 2
        reports[name], results = asyncio.run(evaluate_detailed(system, items, split))
        per_capability[name] = by_capability(items, results, split)
    json_path, md_path = write_reports(reports, args.out, stem=f"eval_{split}")
    cap_text = "\n\n".join(format_capability_markdown(r, name) for name, r in per_capability.items())
    cap_md = args.out / f"eval_{split}_capabilities.md"
    cap_md.write_text(cap_text + "\n", encoding="utf-8")
    (args.out / f"eval_{split}_capabilities.json").write_text(
        json.dumps({n: {c: r.to_dict() for c, r in caps.items()} for n, caps in per_capability.items()}, indent=2),
        encoding="utf-8")
    print(format_markdown(reports))
    print()
    print(cap_text)
    print(f"\nwrote {json_path}, {md_path} and {cap_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
