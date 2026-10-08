"use client";

// Right column: the site plan in cyanotype (camera nodes at site_xy, the path
// of the focused answer drawn hop by hop) and the Known places ledger.
import { useMemo } from "react";
import { clock } from "./format";
import { useEvora } from "./store";

export function SidePanel() {
  const cameras = useEvora((s) => s.cameras);
  const memory = useEvora((s) => s.memory);
  const focus = useEvora((s) => s.focus);
  const cases = useEvora((s) => s.cases);
  const answer = cases.find((c) => c.id === focus?.caseId)?.answer;

  // cameras without a site position are laid out on a gentle arc so the plan is never empty
  const nodes = useMemo(
    () =>
      cameras.map((c, i) => {
        const xy = c.site_xy ?? [0.18 + (0.64 * i) / Math.max(1, cameras.length - 1), 0.5 + Math.sin(i * 1.3) * 0.18];
        return { id: c.id, name: c.name, x: xy[0] * 100, y: xy[1] * 100 };
      }),
    [cameras],
  );
  const byId = new Map(nodes.map((n) => [n.id, n]));
  const hops = (answer?.path ?? []).map((h) => ({ ...h, node: byId.get(h.camera_id) })).filter((h) => h.node);
  const places = memory.filter((f) => !f.superseded_by);

  return (
    <aside className="lt-side" aria-label="Site plan and known places">
      <section className="lt-plan" aria-label="Site plan">
        <h2>Site plan</h2>
        {nodes.length ? (
          <svg viewBox="0 0 100 100" role="img" aria-label={hops.length ? `Path: ${hops.map((h) => `${h.camera_name} ${clock(h.t_in)}`).join(", then ")}` : "Camera positions"}>
            {hops.length > 1 ? (
              <polyline
                key={answer?.query_id}
                className="lt-plan-path"
                points={hops.map((h) => `${h.node?.x},${h.node?.y}`).join(" ")}
                pathLength={1}
              />
            ) : null}
            {nodes.map((n) => (
              <g key={n.id} className="lt-plan-node" transform={`translate(${n.x} ${n.y})`}>
                <circle r={2.2} />
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
      <section className="lt-places" aria-label="Known places">
        <h2>Known places</h2>
        {places.length ? (
          <ul>
            {places.map((f) => (
              <li key={f.id}>
                <span className="lt-marker">{f.canonical}</span>
                <span className="lt-place-meta">
                  {describeBinding(f.binding, cameras)}
                  {f.use_count ? `. Used ${f.use_count}×` : ""}
                </span>
                {f.aliases?.length ? <span className="lt-place-aliases">Also: {f.aliases.join(", ")}</span> : null}
                {f.inferred_aliases?.length ? <span className="lt-place-guess">Guessed: {f.inferred_aliases.join(", ")}</span> : null}
              </li>
            ))}
          </ul>
        ) : (
          <p className="lt-quiet">Places you name while asking are kept here, so nothing is asked twice.</p>
        )}
      </section>
    </aside>
  );
}

function describeBinding(binding: Record<string, unknown>, cameras: { id: string; name: string }[]): string {
  if (typeof binding.camera_id === "string") {
    const cam = cameras.find((c) => c.id === binding.camera_id);
    return `${cam?.name ?? binding.camera_id}${binding.zone_id ? ", marked line" : ""}`;
  }
  if (typeof binding.tod_after === "string" || typeof binding.tod_before === "string") return `${binding.tod_after ?? "…"} to ${binding.tod_before ?? "…"}`;
  return "Learned";
}
