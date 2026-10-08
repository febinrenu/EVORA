"use client";

// Report page (P4.17): the research story from M3's latest report, as
// measured. What the results support and what they do not show sit beside the
// numbers; every value carries its n; capabilities that were not evaluated are
// listed with the reason. Nothing is rounded into a claim.
import { useEffect, useState } from "react";
import { ApiError } from "@/lib/api/client";
import { CAPABILITY, LABELS, SPLITS, fetchReport, metric, splitName, systemName, value, type AblationRow, type EvalFile, type Metric, type Report, type SystemReport } from "./data";

const RATIOS = ["hit@1", "hit@5", "mrr", "camera_accuracy", "negative_precision"] as const;
const TABLE = ["hit@1", "hit@5", "mrr", "camera_accuracy", "timestamp_error_s", "negative_precision", "reask_count", "ttfa_p50_ms"] as const;

const fmt = (v: number | null | undefined, key: string): string => {
  if (v === null || v === undefined || Number.isNaN(v)) return "–";
  if (key === "timestamp_error_s") return `${v.toFixed(1)} s`;
  if (key.endsWith("_ms")) return v < 1000 ? `${Math.round(v)} ms` : `${(v / 1000).toFixed(2)} s`;
  if (key === "reask_count") return String(Math.round(v * 100) / 100);
  return v.toFixed(2);
};
const withN = (m: Metric | null, key: string): string => (m && m.value !== null ? `${fmt(m.value, key)} (n ${m.n})` : "not measured");

export function ReportView({ initial = null }: { initial?: Report | null }) {
  // start from the report baked in at build time, then take the API's if it has one
  const [report, setReport] = useState<Report | null | undefined>(initial ?? undefined);
  const [error, setError] = useState<string | null>(null);
  const [stale, setStale] = useState(false);
  useEffect(() => {
    fetchReport()
      .then((live) => {
        if (live) setReport(live);
        else if (!initial) setReport(null);
        else setStale(true);
      })
      .catch((e: unknown) => {
        if (initial) setStale(true);
        else setError(e instanceof ApiError ? e.message : "The API did not answer.");
      });
  }, [initial]);

  const r = report ?? null;
  const known: readonly string[] = SPLITS;
  const splits = r ? [...known.filter((s) => r.overall[s]), ...Object.keys(r.overall).filter((s) => !known.includes(s))] : [];
  const main = r ? r.overall.test ?? r.overall[splits[0]] : undefined;
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
        {!error && report === undefined ? <p className="rp-note">Loading the latest evaluation…</p> : null}
        {report === null ? (
          <p className="rp-note">
            No evaluation has been published to this machine yet. Run <code>make eval</code>, <code>make ablate</code> and <code>python -m eval.report</code>; this page reads the result.
          </p>
        ) : null}
        {r && main ? (
          <>
            <Provenance r={r} />
            {stale ? <p className="rp-prov">Shown from the copy built into this page; the running API has not published a report.</p> : null}
            {r.supported.length || r.notShown.length ? (
              <div className="rp-claims">
                {r.supported.length ? (
                  <section aria-labelledby="rp-sup">
                    <h2 id="rp-sup" className="rp-claims-head">What the results support</h2>
                    <ul>
                      {r.supported.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </section>
                ) : null}
                {r.notShown.length ? (
                  <section aria-labelledby="rp-not">
                    <h2 id="rp-not" className="rp-claims-head">What they do not show</h2>
                    <ul>
                      {r.notShown.map((s) => (
                        <li key={s}>{s}</li>
                      ))}
                    </ul>
                  </section>
                ) : null}
              </div>
            ) : null}
            <Tiles r={r} main={main} baseline={baseline} />
            <section className="rp-section" aria-labelledby="rp-results">
              <h2 id="rp-results">Results</h2>
              <p className="rp-lead">Same footage, same parsed time window and camera filter for every system; only retrieval differs. n is the number of questions a metric applies to.</p>
              <ResultsTable overall={r.overall} splits={splits} />
            </section>
            {Object.keys(r.capabilities).length ? (
              <section className="rp-section" aria-labelledby="rp-caps">
                <h2 id="rp-caps">By capability</h2>
                <CapabilityTable caps={r.capabilities} splits={splits} />
                {Object.keys(r.notEvaluated).length ? (
                  <>
                    <h3 className="rp-subhead">Not evaluated</h3>
                    <dl className="rp-noteval">
                      {Object.entries(r.notEvaluated).map(([cap, why]) => (
                        <div key={cap}>
                          <dt>{CAPABILITY[cap] ?? cap}</dt>
                          <dd>{why}</dd>
                        </div>
                      ))}
                    </dl>
                  </>
                ) : null}
              </section>
            ) : null}
            {baseline ? (
              <section className="rp-section" aria-labelledby="rp-vs">
                <h2 id="rp-vs">Against the baseline, {splitName(main[systems[0]]?.split ?? "test")} split</h2>
                <PairedBars a={main.ours} b={main[baseline]} bName={systemName(baseline)} />
              </section>
            ) : null}
            {r.ablations.length ? (
              <section className="rp-section" aria-labelledby="rp-abl">
                <h2 id="rp-abl">What each part contributes</h2>
                <p className="rp-lead">Each row turns one contribution off and reruns the frozen test split. A shorter bar than the full system means that part was helping; equal bars mean no effect could be measured on these questions.</p>
                <Ablations rows={r.ablations} />
              </section>
            ) : null}
            <section className="rp-section" aria-labelledby="rp-lat">
              <h2 id="rp-lat">Time to the first answer</h2>
              <p className="rp-lead">Dot is the median, the line runs to the 95th percentile, measured on the same machine in the same session.</p>
              <Latency overall={r.overall} splits={splits} />
            </section>
            {r.limits.length ? (
              <section className="rp-section" aria-labelledby="rp-lim">
                <h2 id="rp-lim">Limits</h2>
                <ul className="rp-limits">
                  {r.limits.map((l) => (
                    <li key={l}>{l}</li>
                  ))}
                </ul>
              </section>
            ) : null}
          </>
        ) : null}
      </main>
    </div>
  );
}

function Provenance({ r }: { r: Report }) {
  const parts: string[] = [];
  if (r.generatedAt) parts.push(`Generated ${r.generatedAt.replace("T", " ").slice(0, 16)}`);
  if (r.commit) parts.push(`code ${r.commit}`);
  if (r.frozen?.code_commit) parts.push(`thresholds frozen at ${r.frozen.code_commit} before ${(r.frozen.frozen_before ?? ["test"]).map(splitName).join(" and ")}`);
  if (!parts.length) return null;
  return <p className="rp-prov">{parts.join(" · ")}.</p>;
}

/** Headline tiles: pooled capability results (with n) when present, else the main split. */
function Tiles({ r, main, baseline }: { r: Report; main: EvalFile; baseline?: string }) {
  const ours = r.pooled.ours;
  const base = baseline ? r.pooled[baseline] : undefined;
  const tiles: { what: string; a: Metric | null; b: Metric | null; key: string; scope: string }[] = [];
  if (ours?.negative) tiles.push({ what: "Says “nothing there” when nothing is there", a: metric(ours.negative, "negative_precision"), b: metric(base?.negative, "negative_precision"), key: "negative_precision", scope: "all splits" });
  tiles.push({ what: "Median distance from the true moment", a: metric(main.ours?.metrics, "timestamp_error_s"), b: baseline ? metric(main[baseline]?.metrics, "timestamp_error_s") : null, key: "timestamp_error_s", scope: `${splitName(main.ours?.split ?? "test")} split` });
  if (ours?.object) tiles.push({ what: "Right clip ranked first, object questions", a: metric(ours.object, "hit@1"), b: metric(base?.object, "hit@1"), key: "hit@1", scope: "all splits" });
  else tiles.push({ what: "Right clip ranked first", a: metric(main.ours?.metrics, "hit@1"), b: baseline ? metric(main[baseline]?.metrics, "hit@1") : null, key: "hit@1", scope: "test split" });
  return (
    <ul className="rp-tiles">
      {tiles.map((t) => (
        <li key={t.what}>
          <span className="rp-tile-what">{t.what}</span>
          <span className="rp-tile-value">{fmt(t.a?.value, t.key)}</span>
          <span className="rp-tile-vs">
            n {t.a?.n ?? 0}, {t.scope}
            {t.b ? ` · ${systemName(baseline ?? "")} ${fmt(t.b.value, t.key)}` : ""}
          </span>
        </li>
      ))}
    </ul>
  );
}

function ResultsTable({ overall, splits }: { overall: Record<string, EvalFile>; splits: string[] }) {
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
            Object.entries(overall[s]).map(([sys, rep]) => (
              <tr key={`${s}-${sys}`} className={sys === "ours" ? "is-ours" : undefined}>
                <th scope="row">{systemName(sys)}</th>
                <td className="rp-text">{splitName(s)}</td>
                <td>{rep.n_queries}</td>
                {TABLE.map((k) => (
                  <td key={k} title={`n = ${rep.metrics[k]?.n ?? 0}`}>
                    {fmt(value(rep.metrics, k), k)}
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

function CapabilityTable({ caps, splits }: { caps: Record<string, Record<string, EvalFile>>; splits: string[] }) {
  const rows: { split: string; sys: string; cap: string; rep: SystemReport }[] = [];
  for (const s of splits)
    for (const [sys, byCap] of Object.entries(caps[s] ?? {}))
      for (const [cap, rep] of Object.entries(byCap)) if (!cap.startsWith("activity")) rows.push({ split: s, sys, cap, rep });
  const cols = ["hit@1", "camera_accuracy", "timestamp_error_s", "negative_precision"] as const;
  return (
    <div className="rp-table-wrap">
      <table className="rp-table">
        <thead>
          <tr>
            <th scope="col">Capability</th>
            <th scope="col">System</th>
            <th scope="col">Split</th>
            {cols.map((k) => (
              <th key={k} scope="col">
                {LABELS[k]}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map(({ split, sys, cap, rep }) => (
            <tr key={`${cap}-${sys}-${split}`} className={sys === "ours" ? "is-ours" : undefined}>
              <th scope="row">{CAPABILITY[cap] ?? cap}</th>
              <td className="rp-text">{systemName(sys)}</td>
              <td className="rp-text">{splitName(split)}</td>
              {cols.map((k) => (
                <td key={k}>{rep.metrics[k]?.value === null || rep.metrics[k] === undefined ? "–" : withN(rep.metrics[k], k)}</td>
              ))}
            </tr>
          ))}
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
      const b = (e.target as HTMLElement).getBoundingClientRect();
      setTip({ x: b.right, y: b.top, text });
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
          const ma = metric(a?.metrics, k);
          const mb = metric(b?.metrics, k);
          return (
            <li key={k}>
              <span className="rp-row-label">{LABELS[k]}</span>
              <span className="rp-bars">
                <span className="rp-track">
                  <i role="img" className="rp-bar is-a" style={{ width: `${(ma?.value ?? 0) * 100}%` }} {...bind(`EVORA, ${LABELS[k]}: ${withN(ma, k)}`)} />
                  <span className="rp-val">{fmt(ma?.value, k)}</span>
                </span>
                <span className="rp-track">
                  <i role="img" className="rp-bar is-b" style={{ width: `${(mb?.value ?? 0) * 100}%` }} {...bind(`${bName}, ${LABELS[k]}: ${withN(mb, k)}`)} />
                  <span className="rp-val">{fmt(mb?.value, k)}</span>
                </span>
              </span>
            </li>
          );
        })}
      </ol>
      <figcaption>Scale 0 to 1 for every row; higher is better. Hover or focus a bar for its n.</figcaption>
      {node}
    </figure>
  );
}

function Ablations({ rows }: { rows: AblationRow[] }) {
  const { bind, node } = useTip();
  const full = rows.find((r) => r.label === "full system");
  const shown = ["hit@1", "negative_precision"] as const;
  return (
    <figure className="rp-figure">
      <div className="rp-abl">
        {shown.map((k) => {
          const ref = value(full?.metrics, k);
          return (
            <div key={k} className="rp-abl-col">
              <h3>{LABELS[k]}</h3>
              <ol>
                {rows.map((row) => {
                  const m = metric(row.metrics, k);
                  const v = m?.value ?? null;
                  const isFull = row.label === "full system";
                  const label = isFull ? "Full system" : row.label.replace(/^no /, "without ");
                  return (
                    <li key={row.label} className={isFull ? "is-full" : undefined}>
                      <span className="rp-row-label">{label}</span>
                      {row.skipped ? (
                        <span className="rp-skip">Not run: {row.skipped.replace(/^skipped: /, "")}</span>
                      ) : (
                        <span className="rp-track">
                          <i
                            role="img"
                            className="rp-bar is-a"
                            style={{ width: `${(v ?? 0) * 100}%` }}
                            {...bind(`${label}: ${LABELS[k]} ${withN(m, k)}${ref !== null && v !== null && !isFull ? `, ${v - ref >= 0 ? "+" : ""}${(v - ref).toFixed(2)} vs full` : ""}`)}
                          />
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

function Latency({ overall, splits }: { overall: Record<string, EvalFile>; splits: string[] }) {
  const { bind, node } = useTip();
  const rows = splits.flatMap((s) => Object.entries(overall[s]).map(([sys, rep]) => ({ s, sys, p50: value(rep.metrics, "ttfa_p50_ms"), p95: value(rep.metrics, "ttfa_p95_ms"), n: rep.metrics.ttfa_p50_ms?.n ?? 0 })));
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
              {systemName(r.sys)} <em>{splitName(r.s)}</em>
            </span>
            <span className="rp-lat-track">
              {r.p50 !== null && r.p95 !== null ? <i className={`rp-range is-${r.sys === "ours" ? "a" : "b"}`} style={{ left: `${(r.p50 / max) * 100}%`, width: `${((r.p95 - r.p50) / max) * 100}%` }} /> : null}
              {r.p50 !== null ? (
                <i role="img" className={`rp-dot is-${r.sys === "ours" ? "a" : "b"}`} style={{ left: `${(r.p50 / max) * 100}%` }} {...bind(`${systemName(r.sys)}, ${splitName(r.s)}: median ${fmt(r.p50, "ttfa_p50_ms")}, 95th ${fmt(r.p95, "ttfa_p95_ms")} (n ${r.n})`)} />
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
