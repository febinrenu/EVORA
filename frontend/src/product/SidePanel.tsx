"use client";

// Right column: the site plan in cyanotype (P4.11) and the Known places
// ledger. Camera nodes sit at their site_xy; drag one (or focus it and use the
// arrow keys) to put it where the camera really is, which is saved on the
// camera. The focused answer's path is drawn hop by hop with its times.
import { useMemo, useRef, useState } from "react";
import { endpoints } from "@/lib/api/client";
import { clock } from "./format";
import { useEvora } from "./store";
import { KnownPlaces } from "./KnownPlaces";

export function SidePanel() {
  const cameras = useEvora((s) => s.cameras);
  const focus = useEvora((s) => s.focus);
  const cases = useEvora((s) => s.cases);
  const answer = cases.find((c) => c.id === focus?.caseId)?.answer;
  const svg = useRef<SVGSVGElement>(null);
  // the node being moved, before the position is saved
  const [moving, setMoving] = useState<{ id: string; x: number; y: number } | null>(null);
  const saveTimer = useRef(0);

  // cameras without a site position are laid out on a gentle arc so the plan is never empty
  const nodes = useMemo(
    () =>
      cameras.map((c, i) => {
        const xy = c.site_xy ?? [0.18 + (0.64 * i) / Math.max(1, cameras.length - 1), 0.5 + Math.sin(i * 1.3) * 0.18];
        return { id: c.id, name: c.name, x: xy[0] * 100, y: xy[1] * 100, placed: Boolean(c.site_xy) };
      }),
    [cameras],
  );
  const shown = nodes.map((n) => (moving?.id === n.id ? { ...n, x: moving.x, y: moving.y } : n));
  const byId = new Map(shown.map((n) => [n.id, n]));
  const hops = (answer?.path ?? []).map((h) => ({ ...h, node: byId.get(h.camera_id) })).filter((h) => h.node);

  const toPlan = (e: React.PointerEvent): { x: number; y: number } => {
    const r = svg.current?.getBoundingClientRect();
    if (!r) return { x: 50, y: 50 };
    return { x: Math.min(96, Math.max(4, ((e.clientX - r.left) / r.width) * 100)), y: Math.min(96, Math.max(6, ((e.clientY - r.top) / r.height) * 100)) };
  };

  const save = async (id: string, x: number, y: number) => {
    try {
      useEvora.getState().upsertCameras([await endpoints.placeCamera(id, [x / 100, y / 100])]);
    } catch {
      /* the node snaps back to the saved position */
    } finally {
      setMoving((m) => (m?.id === id ? null : m));
    }
  };

  return (
    <aside className="lt-side" aria-label="Site plan and known places">
      <section className="lt-plan" aria-label="Site plan">
        <h2>Site plan</h2>
        {shown.length ? (
          <svg ref={svg} viewBox="0 0 100 100" aria-label={hops.length ? `Path: ${hops.map((h) => `${h.camera_name} ${clock(h.t_in)}`).join(", then ")}` : "Camera positions. Drag a camera to where it is on site."}>
            {hops.length > 1 ? <polyline key={answer?.query_id} className="lt-plan-path" points={hops.map((h) => `${h.node?.x},${h.node?.y}`).join(" ")} pathLength={1} /> : null}
            {shown.map((n) => (
              <g
                key={n.id}
                className="lt-plan-node"
                transform={`translate(${n.x} ${n.y})`}
                tabIndex={0}
                role="button"
                aria-label={`${n.name}${n.placed ? "" : ", not placed yet"}. Drag, or use the arrow keys, to move it.`}
                onPointerDown={(e) => {
                  (e.currentTarget as Element).setPointerCapture(e.pointerId);
                  setMoving({ id: n.id, ...toPlan(e) });
                }}
                onPointerMove={(e) => {
                  if (moving?.id === n.id) setMoving({ id: n.id, ...toPlan(e) });
                }}
                onPointerUp={() => {
                  if (moving?.id === n.id) void save(n.id, moving.x, moving.y);
                }}
                onKeyDown={(e) => {
                  const d: Record<string, [number, number]> = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, -1], ArrowDown: [0, 1] };
                  const step = d[e.key];
                  if (!step) return;
                  e.preventDefault();
                  const k = e.shiftKey ? 5 : 1;
                  const x = Math.min(96, Math.max(4, n.x + step[0] * k));
                  const y = Math.min(96, Math.max(6, n.y + step[1] * k));
                  setMoving({ id: n.id, x, y });
                  window.clearTimeout(saveTimer.current);
                  saveTimer.current = window.setTimeout(() => void save(n.id, x, y), 450);
                }}
              >
                <circle r={2.2} className={n.placed ? undefined : "is-unplaced"} />
                <text y={-4.2}>{n.name}</text>
              </g>
            ))}
            {hops.map((h, i) => (
              <text key={i} className="lt-plan-time" x={h.node?.x} y={(h.node?.y ?? 0) + 7}>
                {clock(h.t_in).slice(0, 5)}
              </text>
            ))}
          </svg>
        ) : (
          <p className="lt-quiet">Cameras appear here once footage is loaded. Drag them into place later.</p>
        )}
      </section>
      <KnownPlaces />
    </aside>
  );
}
