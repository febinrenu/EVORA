"use client";

// Timeline (P4.10): one lane per camera across the footage, the focused
// answer's evidence as red ticks (click one to open it), and a scrub head.
// Click or drag on a lane to see that camera's frame at that moment; with the
// lanes focused, ←/→ step a second (Shift: ten), J/L five, Home/End jump.
import { useMemo, useRef } from "react";
import { frameUrl } from "@/lib/api/client";
import { clock } from "./format";
import { useEvora } from "./store";
import { Frame } from "./Frame";

export function Timeline() {
  const cameras = useEvora((s) => s.cameras);
  const cases = useEvora((s) => s.cases);
  const focus = useEvora((s) => s.focus);
  const setFocus = useEvora((s) => s.setFocus);
  const playhead = useEvora((s) => s.playhead);
  const setPlayhead = useEvora((s) => s.setPlayhead);
  const active = cases.find((c) => c.id === focus?.caseId) ?? cases.findLast((c) => c.evidence.length > 0);
  const lanes = useRef<HTMLOListElement>(null);
  const dragging = useRef(false);

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
  const clamp = (t: number) => Math.min(range.b, Math.max(range.a, t));

  const scrubTo = (e: React.PointerEvent, cameraId: string) => {
    const lane = (e.currentTarget as HTMLElement).getBoundingClientRect();
    const f = Math.min(1, Math.max(0, (e.clientX - lane.left) / lane.width));
    setPlayhead({ t: range.a + f * (range.b - range.a), cameraId });
  };

  const onKey = (e: React.KeyboardEvent) => {
    const cur = playhead ?? { t: range.a, cameraId: cameras[0].id };
    const step: Record<string, number> = { ArrowRight: 1, ArrowLeft: -1, l: 5, j: -5, L: 5, J: -5 };
    if (e.key in step) setPlayhead({ ...cur, t: clamp(cur.t + step[e.key] * (e.shiftKey && e.key.startsWith("Arrow") ? 10 : 1)) });
    else if (e.key === "Home") setPlayhead({ ...cur, t: range.a });
    else if (e.key === "End") setPlayhead({ ...cur, t: range.b });
    else if (e.key === "ArrowUp" || e.key === "ArrowDown") {
      const i = cameras.findIndex((c) => c.id === cur.cameraId);
      const j = Math.min(cameras.length - 1, Math.max(0, i + (e.key === "ArrowDown" ? 1 : -1)));
      setPlayhead({ ...cur, cameraId: cameras[j].id });
    } else if (e.key === "Escape") setPlayhead(null);
    else return;
    e.preventDefault();
  };

  const headCam = playhead ? cameras.find((c) => c.id === playhead.cameraId) : undefined;
  const inFootage = headCam && playhead ? playhead.t >= headCam.t0 && playhead.t <= headCam.t0 + (headCam.duration_s ?? 0) : false;

  return (
    <footer className="lt-timeline" aria-label="Timeline">
      <div className="lt-tl-body">
        <div className="lt-tl-tracks">
          <ol
            ref={lanes}
            className="lt-lanes"
            tabIndex={0}
            aria-label={playhead ? `Timeline at ${clock(playhead.t)} on ${headCam?.name ?? "a camera"}. Arrow keys move.` : "Timeline. Click a lane or use the arrow keys to look at a moment."}
            onKeyDown={onKey}
          >
            {cameras.map((c) => (
              <li key={c.id} className={playhead?.cameraId === c.id ? "is-current" : undefined}>
                <span className="lt-lane-name">{c.name}</span>
                <span
                  className="lt-lane"
                  onPointerDown={(e) => {
                    if ((e.target as HTMLElement).closest(".lt-tick")) return;
                    dragging.current = true;
                    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
                    scrubTo(e, c.id);
                  }}
                  onPointerMove={(e) => dragging.current && scrubTo(e, c.id)}
                  onPointerUp={() => (dragging.current = false)}
                >
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
                  {playhead ? <i className="lt-head" style={{ left: `${x(playhead.t)}%` }} aria-hidden="true" /> : null}
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
        </div>
        {playhead && headCam ? (
          <aside className="lt-peek" aria-live="polite">
            {inFootage ? <Frame src={frameUrl(headCam.id, playhead.t)} alt={`${headCam.name} at ${clock(playhead.t)}`} /> : <div className="lt-frame lt-frame-missing"><span>No footage at this moment</span></div>}
            <p>
              <b>{headCam.name}</b> {clock(playhead.t)}
            </p>
          </aside>
        ) : null}
      </div>
    </footer>
  );
}
