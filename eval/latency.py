"""Time to first answer by how the question was planned: fast path, plan cache, cloud model, local model.

The evaluation queries are phrased so that the fast path parses them, which is why their latency is the best case.
This runs a fixed set of freer phrasings through the full pipeline in three conditions and reports the median and
95th percentile per planner route, so the cost of needing a language model is a number rather than a guess.

    python -m eval.latency --workspace meva-school --root <workspaces folder> --out eval/reports/latency.json

Conditions: `cloud` plans with the configured cloud model; `cache` repeats the same questions, which now hit the plan
cache; `local` removes the cloud keys so the local model has to plan. Each condition uses a copy of the workspace
database state it is given, so run this on a copy: plan caches and remembered places are written to it.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

FREE_PHRASING = [
    "Has any car turned up on G328 in the first couple of minutes?",
    "Anybody walking about on G299 between ten past and quarter past ten?",
    "Show me when a vehicle shows up on G423",
    "Any people hanging around G300 near 10:12?",
    "Was somebody in dark clothing seen on G419?",
    "Find the earliest time somebody appears on G421",
    "Is there any activity on G299 at 10:13?",
    "Did someone cross the view of G423 around 10:14?",
    "Were there pedestrians on G328 early on?",
    "Tell me about vehicles on G300",
    "What happened on G419 between 10:11 and 10:12?",
    "Anything moving on G421 in the last minute of the clip?",
]


def percentile(values: list[float], q: float) -> float:
    """Nearest-rank percentile; defined for any non-empty list (a 12-point p95 is its largest value)."""
    ordered = sorted(values)
    return ordered[max(0, math.ceil(q * len(ordered)) - 1)]


def summarise(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Group {source, ttfa_ms} rows by planner route: n, median and p95 of time to first answer."""
    by_source: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        by_source[r.get("source") or "unknown"].append(float(r["ttfa_ms"]))
    return {
        src: {"n": len(v), "p50_ms": round(statistics.median(v), 1), "p95_ms": round(percentile(v, 0.95), 1),
              "max_ms": round(max(v), 1)}
        for src, v in sorted(by_source.items())
    }


async def _run(system: Any, questions: list[str]) -> list[dict[str, Any]]:
    rows = []
    for i, text in enumerate(questions):
        item = SimpleNamespace(id=f"lat_{i}", text=text, first_time_requires_clarify=[], clarify_answer=None)
        result = await system.run(item)
        rows.append({"text": text, "source": result.plan_source, "ttfa_ms": result.ttfa_ms, "ttva_ms": result.ttva_ms,
                     "asked": result.asked_clarify, "error": result.error})
    return rows


def _keyless_gateway() -> Any:
    import httpx

    from evora.core.config import load_env_file
    from evora.llm.gateway import Gateway
    from evora.llm.keypool import KeyPool
    from evora.llm.schemas import GatewayConfig

    load_env_file()
    return Gateway(GatewayConfig.from_env(os.environ), KeyPool([]), httpx.AsyncClient())


async def measure(workspace: str, root: Path | None, questions: list[str]) -> dict[str, Any]:
    from eval.systems import build_ours

    out: dict[str, Any] = {"questions": len(questions), "conditions": {}}
    system = build_ours(workspace, root)
    try:
        cold = await _run(system, questions)
        warm = await _run(system, questions)  # the plans are cached now
    finally:
        await system.aclose()
    out["conditions"]["cloud"] = {"rows": cold, "by_route": summarise(cold)}
    out["conditions"]["cache"] = {"rows": warm, "by_route": summarise(warm)}

    # no cloud keys: the same gateway code, an empty key pool, so the local model has to plan
    local_system = build_ours(workspace, root, gateway=_keyless_gateway())
    # a different wording per question so the plan cache cannot answer
    fresh = [q.rstrip("?") + " today?" for q in questions]
    try:
        local = await _run(local_system, fresh)
    finally:
        await local_system.aclose()
    out["conditions"]["local"] = {"rows": local, "by_route": summarise(local)}
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workspace", default="meva-school")
    parser.add_argument("--root", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("eval/reports/latency.json"))
    args = parser.parse_args(argv)
    result = asyncio.run(measure(args.workspace, args.root, FREE_PHRASING))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for cond, data in result["conditions"].items():
        for route, s in data["by_route"].items():
            print(f"{cond:6s} {route:10s} n={s['n']:2d} p50 {s['p50_ms']:8.1f} ms  p95 {s['p95_ms']:8.1f} ms")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
