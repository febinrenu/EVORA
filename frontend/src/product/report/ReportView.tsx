"use client";

// Report page (P4.17): the research story from the latest `make eval` and
// `make ablate`, exactly as measured. Headline tiles, the full table, a paired
// comparison against the baseline, ablations and latency. Numbers are never
// rounded into claims; anything not measured says so.
import { useEffect, useState } from "react";
import { ApiError } from "@/lib/api/client";
import { LABELS, SPLITS, count, evalBySplit, fetchReport, metric, systemName, type AblationFile, type EvalFile, type ReportPayload, type SystemReport } from "./data";

const RATIOS = ["hit@1", "hit@5", "mrr", "camera_accuracy", "negative_precision"] as const;
const TABLE = ["hit@1", "hit@5", "mrr", "camera_accuracy", "timestamp_error_s", "negative_precision", "reask_count", "ttfa_p50_ms"] as const;

const fmt = (v: number | null, key: string): string => {
  if (v === null || Number.isNaN(v)) return "–";
  if (key === "timestamp_error_s") return `${v.toFixed(1)} s`;
  if (key.endsWith("_ms")) return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`;
  if (key === "reask_count") return String(Math.round(v * 100) / 100);
  return v.toFixed(2);
};

export function ReportView() {
  const [data, setData] = useState<ReportPayload | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    fetchReport()
      .then(setData)
      .catch((e: unknown) => setError(e instanceof ApiError ? e.message : "The API did not answer."));
  }, []);

  const bySplit = evalBySplit(data);
  const known: readonly string[] = SPLITS;
  const splits: string[] = [...known.filter((s) => bySplit[s]), ...Object.keys(bySplit).filter((s) => !known.includes(s))];
  const main = bySplit.test ?? bySplit[splits[0]];
  const systems = main ? Object.keys(main) : [];
  const baseline = systems.find((s) => s !== "ours");

  return (
    <div className="rp">
      <header className="rp-top">
        {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
        <a href="/" className="rp-mark">
          EVORA
        </a>
        <a href="/app/">Open the case log</a>
      </header>
      <main className="rp-main">
        <h1 className="rp-title">What the measurements show</h1>
        {error ? <p className="rp-note">The report could not be loaded: {error}</p> : null}
        {!error && !data ? <p className="rp-note">Loading the latest evaluation…</p> : null}
        {data && !main ? (
          <p className="rp-note">
            No evaluation has been published to this machine yet. Run <code>make eval</code> and <code>make ablate</code>; the page reads their reports.
          </p>
        ) : null}
        {main ? (
          <>
            <Summary main={main} baseline={baseline} />
            <Tiles main={main} baseline={baseline} />
            <section className="rp-section" aria-labelledby="rp-results">
              <h2 id="rp-results">Results</h2>
              <p className="rp-lead">Same footage, same parsed time window and camera filter for every system; only retrieval differs. n counts the queries a metric applies to.</p>
              <ResultsTable bySplit={bySplit} splits={splits} />
            </section>
            {baseline ? (
              <section className="rp-section" aria-labelledby="rp-vs">
                <h2 id="rp-vs">Against the baseline, {main[systems[0]]?.split ?? "test"} split</h2>
                <PairedBars a={main.ours} b={main[baseline]} bName={systemName(baseline)} />
              </section>
            ) : null}
            {data?.ablations ? (
              <section className="rp-section" aria-labelledby="rp-abl">
                <h2 id="rp-abl">What each part contributes</h2>
                <p className="rp-lead">Each row turns one contribution off and reruns the frozen test split. A shorter bar than the full system means that part was helping.</p>
                <Ablations rows={data.ablations} />
              </section>
            ) : null}
            <section className="rp-section" aria-labelledby="rp-lat">
              <h2 id="rp-lat">Time to the first answer</h2>
              <p className="rp-lead">Dot is the median, the line runs to the 95th percentile, measured on the same machine in the same session.</p>
              <Latency bySplit={bySplit} splits={splits} />
            </section>
          </>
        ) : null}
      </main>
    </div>
  );
}

/** One plain sentence that states the trade-off the numbers show, both ways. */
function Summary({ main, baseline }: { main: EvalFile; baseline?: string }) {
  const o = main.ours;
  const b = baseline ? main[baseline] : undefined;
  if (!o) return null;
  const parts: string[] = [];
  const neg = [metric(o, "negative_precision"), metric(b, "negative_precision")];
  const ts = [metric(o, "timestamp_error_s"), metric(b, "timestamp_error_s")];
  const hit = [metric(o, "hit@1"), metric(b, "hit@1")];
  if (neg[0] !== null && neg[1] !== null) parts.push(neg[0] > neg[1] ? `says “nothing there” correctly far more often (${fmt(neg[0], "")} vs ${fmt(neg[1], "")})` : `is not better at saying “nothing there” (${fmt(neg[0], "")} vs ${fmt(neg[1], "")})`);
  if (ts[0] !== null && ts[1] !== null) parts.push(ts[0] < ts[1] ? `lands closer to the moment (${fmt(ts[0], "timestamp_error_s")} vs ${fmt(ts[1], "timestamp_error_s")} median error)` : `lands further from the moment (${fmt(ts[0], "timestamp_error_s")} vs ${fmt(ts[1], "timestamp_error_s")})`);
  const hitLine = hit[0] !== null && hit[1] !== null ? (hit[0] >= hit[1] ? `and finds the right clip first at least as often (Hit@1 ${fmt(hit[0], "")} vs ${fmt(hit[1], "")}).` : `but finds the right clip first less often (Hit@1 ${fmt(hit[0], "")} vs ${fmt(hit[1], "")}).`) : ".";
  return (
    <p className="rp-summary">
      On the {o.split} split ({o.n_queries} questions), EVORA {parts.join(" and ")} {hitLine}
    </p>
  );
}

function Tiles({ main, baseline }: { main: EvalFile; baseline?: string }) {
  const o = main.ours;
  const b = baseline ? main[baseline] : undefined;
  const tiles: [string, string][] = [
    ["negative_precision", "Says “nothing there” when nothing is there"],
    ["timestamp_error_s", "Median distance from the true moment"],
    ["hit@1", "Right clip ranked first"],
  ];
  return (
    <ul className="rp-tiles">
      {tiles.map(([k, what]) => (
        <li key={k}>
          <span className="rp-tile-what">{what}</span>
          <span className="rp-tile-value">{fmt(metric(o, k), k)}</span>
          <span className="rp-tile-vs">
            {b ? `${systemName(baseline ?? "")} ${fmt(metric(b, k), k)}` : "no baseline"} · n {count(o, k)}
          </span>
        </li>
      ))}
    </ul>
  );
}

function ResultsTable({ bySplit, splits }: { bySplit: Record<string, EvalFile>; splits: string[] }) {
  return (
    <div className="rp-table-wrap">
      <table className="rp-table">
        <thead>
          <tr>
            <th scope="col">System</th>
            <th scope="col">Split</th>
            <th scope="col">n</th>
            {TABLE.map((k) => (
              <th key={k} scope="col">
                {LABELS[k] ?? k}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {splits.flatMap((s) =>
            Object.entries(bySplit[s]).map(([sys, r]) => (
              <tr key={`${s}-${sys}`} className={sys === "ours" ? "is-ours" : undefined}>
                <th scope="row">{systemName(sys)}</th>
                <td>{s.replace("_", " ")}</td>
                <td>{r.n_queries}</td>
                {TABLE.map((k) => (
                  <td key={k} title={`n = ${count(r, k)}`}>
                    {fmt(metric(r, k), k)}
                  </td>
                ))}
              </tr>
            )),
          )}
        </tbody>
      </table>
    </div>
  );
}

/** Hover tooltip shared by the charts: one element, positioned at the pointer. */
function useTip() {
  const [tip, setTip] = useState<{ x: number; y: number; text: string } | null>(null);
  const bind = (text: string) => ({
    onPointerEnter: (e: React.PointerEvent) => setTip({ x: e.clientX, y: e.clientY, text }),
    onPointerMove: (e: React.PointerEvent) => setTip({ x: e.clientX, y: e.clientY, text }),
    onPointerLeave: () => setTip(null),
    onFocus: (e: React.FocusEvent) => {
      const r = (e.target as HTMLElement).getBoundingClientRect();
      setTip({ x: r.right, y: r.top, text });
    },
    onBlur: () => setTip(null),
    tabIndex: 0,
    "aria-label": text,
  });
  const node = tip ? (
    <div className="rp-tip" style={{ left: tip.x + 12, top: tip.y - 10 }} role="tooltip">
      {tip.text}
    </div>
  ) : null;
  return { bind, node };
}

function PairedBars({ a, b, bName }: { a?: SystemReport; b?: SystemReport; bName: string }) {
  const { bind, node } = useTip();
  return (
    <figure className="rp-figure">
      <div className="rp-legend" aria-hidden="true">
        <span className="rp-key is-a">EVORA</span>
        <span className="rp-key is-b">{bName}</span>
      </div>
      <ol className="rp-pairs">
        {RATIOS.map((k) => {
          const va = metric(a, k);
          const vb = metric(b, k);
          return (
            <li key={k}>
              <span className="rp-row-label">{LABELS[k]}</span>
              <span className="rp-bars">
                <span className="rp-track">
                  <i className="rp-bar is-a" style={{ width: `${(va ?? 0) * 100}%` }} {...bind(`EVORA, ${LABELS[k]}: ${fmt(va, k)} (n ${count(a, k)})`)} />
                  <span className="rp-val">{fmt(va, k)}</span>
                </span>
                <span className="rp-track">
                  <i className="rp-bar is-b" style={{ width: `${(vb ?? 0) * 100}%` }} {...bind(`${bName}, ${LABELS[k]}: ${fmt(vb, k)} (n ${count(b, k)})`)} />
                  <span className="rp-val">{fmt(vb, k)}</span>
                </span>
              </span>
            </li>
          );
        })}
      </ol>
      <figcaption>Scale 0 to 1 for every row. Higher is better.</figcaption>
      {node}
    </figure>
  );
}

function Ablations({ rows }: { rows: AblationFile }) {
  const { bind, node } = useTip();
  const entries = Object.entries(rows);
  const full = rows["full system"]?.report;
  const metricsShown = ["hit@1", "negative_precision"] as const;
  return (
    <figure className="rp-figure">
      <div className="rp-abl">
        {metricsShown.map((k) => {
          const ref = metric(full, k);
          return (
            <div key={k} className="rp-abl-col">
              <h3>{LABELS[k]}</h3>
              <ol>
                {entries.map(([name, row]) => {
                  const v = metric(row.report, k);
                  const label = name.replace(/^no /, "without ");
                  return (
                    <li key={name} className={name === "full system" ? "is-full" : undefined}>
                      <span className="rp-row-label">{name === "full system" ? "Full system" : label}</span>
                      {row.skipped ? (
                        <span className="rp-skip">Not run: {row.skipped.replace(/^skipped: /, "")}</span>
                      ) : (
                        <span className="rp-track">
                          <i className="rp-bar is-a" style={{ width: `${(v ?? 0) * 100}%` }} {...bind(`${name}: ${LABELS[k]} ${fmt(v, k)}${ref !== null && v !== null && name !== "full system" ? ` (${v - ref >= 0 ? "+" : ""}${(v - ref).toFixed(2)} vs full)` : ""}`)} />
                          <span className="rp-val">{fmt(v, k)}</span>
                          {ref !== null ? <b className="rp-ref" style={{ left: `${ref * 100}%` }} aria-hidden="true" /> : null}
                        </span>
                      )}
                    </li>
                  );
                })}
              </ol>
            </div>
          );
        })}
      </div>
      <figcaption>The thin mark is the full system. Rows without a bar were not measured on this footage.</figcaption>
      {node}
    </figure>
  );
}

function Latency({ bySplit, splits }: { bySplit: Record<string, EvalFile>; splits: string[] }) {
  const { bind, node } = useTip();
  const rows = splits.flatMap((s) => Object.entries(bySplit[s]).map(([sys, r]) => ({ s, sys, p50: metric(r, "ttfa_p50_ms"), p95: metric(r, "ttfa_p95_ms") })));
  const max = Math.max(1, ...rows.map((r) => r.p95 ?? r.p50 ?? 0));
  const ticks = [0, 0.25, 0.5, 0.75, 1].map((f) => f * max);
  return (
    <figure className="rp-figure">
      <div className="rp-legend" aria-hidden="true">
        <span className="rp-key is-a">EVORA</span>
        <span className="rp-key is-b">Baseline</span>
      </div>
      <ol className="rp-lat">
        {rows.map((r) => (
          <li key={`${r.s}-${r.sys}`}>
            <span className="rp-row-label">
              {systemName(r.sys)} <em>{r.s.replace("_", " ")}</em>
            </span>
            <span className="rp-lat-track">
              {r.p50 !== null && r.p95 !== null ? <i className={`rp-range is-${r.sys === "ours" ? "a" : "b"}`} style={{ left: `${(r.p50 / max) * 100}%`, width: `${((r.p95 - r.p50) / max) * 100}%` }} /> : null}
              {r.p50 !== null ? (
                <i className={`rp-dot is-${r.sys === "ours" ? "a" : "b"}`} style={{ left: `${(r.p50 / max) * 100}%` }} {...bind(`${systemName(r.sys)}, ${r.s}: median ${fmt(r.p50, "ttfa_p50_ms")}, 95th ${fmt(r.p95, "ttfa_p95_ms")}`)} />
              ) : null}
              <span className="rp-lat-val">
                {fmt(r.p50, "ttfa_p50_ms")} · {fmt(r.p95, "ttfa_p95_ms")}
              </span>
            </span>
          </li>
        ))}
      </ol>
      <div className="rp-lat-axis" aria-hidden="true">
        {ticks.map((t, i) => (
          <span key={i} style={{ left: `${(t / max) * 100}%` }}>
            {fmt(t, "ttfa_p50_ms")}
          </span>
        ))}
      </div>
      {node}
    </figure>
  );
}
