"use client";

// Timeline lanes (P4.10, first pass): one lane per camera across the footage
// day, the focused answer's evidence as red ticks. Clicking a tick opens it.
import { useMemo } from "react";
import { clock } from "./format";
import { useEvora } from "./store";

export function Timeline() {
  const cameras = useEvora((s) => s.cameras);
  const cases = useEvora((s) => s.cases);
  const focus = useEvora((s) => s.focus);
  const setFocus = useEvora((s) => s.setFocus);
  const active = cases.find((c) => c.id === focus?.caseId) ?? cases.findLast((c) => c.evidence.length > 0);

  const range = useMemo(() => {
    if (!cameras.length) return null;
    const a = Math.min(...cameras.map((c) => c.t0));
    const b = Math.max(...cameras.map((c) => c.t0 + (c.duration_s ?? 0)));
    return b > a ? { a, b } : null;
  }, [cameras]);

  if (!range) {
    return (
      <footer className="lt-timeline" aria-label="Timeline">
        <p className="lt-quiet">The timeline fills in as footage is indexed.</p>
      </footer>
    );
  }
  const x = (t: number) => ((t - range.a) / (range.b - range.a)) * 100;
  const ticks = 6;

  return (
    <footer className="lt-timeline" aria-label="Timeline">
      <ol className="lt-lanes">
        {cameras.map((c) => (
          <li key={c.id}>
            <span className="lt-lane-name">{c.name}</span>
            <span className="lt-lane">
              <i className="lt-lane-span" style={{ left: `${x(c.t0)}%`, width: `${x(c.t0 + (c.duration_s ?? 0)) - x(c.t0)}%` }} />
              {active?.evidence
                .filter((ev) => ev.camera_id === c.id)
                .map((ev) => (
                  <button
                    key={ev.id}
                    type="button"
                    className={`lt-tick${focus?.evidenceId === ev.id ? " is-active" : ""}`}
                    style={{ left: `${x(ev.t_peak)}%` }}
                    onClick={() => active && setFocus({ caseId: active.id, evidenceId: ev.id })}
                    aria-label={`${c.name} at ${clock(ev.t_peak)}`}
                  />
                ))}
            </span>
          </li>
        ))}
      </ol>
      <div className="lt-axis" aria-hidden="true">
        {Array.from({ length: ticks + 1 }, (_, i) => (
          <span key={i} style={{ left: `${(i / ticks) * 100}%` }}>
            {clock(range.a + ((range.b - range.a) * i) / ticks).slice(0, 5)}
          </span>
        ))}
      </div>
    </footer>
  );
}
