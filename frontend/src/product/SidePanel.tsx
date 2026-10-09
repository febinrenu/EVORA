"use client";

// Right column: the site plan in cyanotype (P4.11) and the Known places
// ledger. Camera nodes sit at their site_xy over an optional floor plan; drag
// one (or focus it and use the arrow keys) to put it where the camera really
// is, which is saved on the camera. Click a camera to see its latest frame.
// The focused answer's path is drawn hop by hop with its times, and listed
// beneath the plan as a film strip of the frames at each camera.
import { useEffect, useMemo, useRef, useState } from "react";
import { ApiError, apiUrl, endpoints, frameUrl, liveUrl, type CameraInfo } from "@/lib/api/client";
import { clock, day } from "./format";
import { useEvora } from "./store";
import { Frame } from "./Frame";
import { KnownPlaces } from "./KnownPlaces";

/** a press that moves less than this (plan units, 0..100) is a click, not a drag */
const CLICK_SLOP = 1.2;
/** the floor plan is downscaled to this many pixels on its long side before it is kept */
const PLAN_MAX_PX = 1600;

interface Node {
  id: string;
  name: string;
  x: number;
  y: number;
  placed: boolean;
}

export function SidePanel() {
  const cameras = useEvora((s) => s.cameras);
  const focus = useEvora((s) => s.focus);
  const cases = useEvora((s) => s.cases);
  const workspace = useEvora((s) => s.health?.workspace ?? "default");
  const focusedCase = cases.find((c) => c.id === focus?.caseId);
  const answer = focusedCase?.answer;
  const svg = useRef<SVGSVGElement>(null);
  // the node being moved, before the position is saved
  const [moving, setMoving] = useState<{ id: string; x: number; y: number } | null>(null);
  const press = useRef<{ id: string; x: number; y: number; dragged: boolean } | null>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const saveTimer = useRef(0);
  const floor = useFloorPlan(workspace);

  // cameras without a site position are laid out on a gentle arc so the plan is never empty
  const nodes = useMemo<Node[]>(
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
  const selectedCam = cameras.find((c) => c.id === selected) ?? null;

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

  const toggle = (id: string) => setSelected((s) => (s === id ? null : id));

  return (
    <aside className="lt-side" aria-label="Site plan and known places">
      <section className="lt-plan" aria-label="Site plan">
        <div className="lt-plan-head">
          <h2>Site plan</h2>
          <FloorPlanControls floor={floor} />
        </div>
        {shown.length ? (
          <svg
            ref={svg}
            viewBox="0 0 100 100"
            className={floor.url ? "has-floor" : undefined}
            aria-label={hops.length ? `Path: ${hops.map((h) => `${h.camera_name} ${clock(h.t_in)}`).join(", then ")}` : "Camera positions. Drag a camera to where it is on site; click it to see its latest frame."}
          >
            {floor.url ? <image href={floor.url} x={0} y={0} width={100} height={100} preserveAspectRatio="xMidYMid meet" className="lt-plan-floor" /> : null}
            {hops.length > 1 ? <polyline key={answer?.query_id} className="lt-plan-path" points={hops.map((h) => `${h.node?.x},${h.node?.y}`).join(" ")} pathLength={1} /> : null}
            {shown.map((n) => (
              <g
                key={n.id}
                className={`lt-plan-node${selected === n.id ? " is-selected" : ""}`}
                transform={`translate(${n.x} ${n.y})`}
                tabIndex={0}
                role="button"
                aria-pressed={selected === n.id}
                aria-label={`${n.name}${n.placed ? "" : ", not placed yet"}. Press Enter to see its latest frame; drag, or use the arrow keys, to move it.`}
                onPointerDown={(e) => {
                  (e.currentTarget as Element).setPointerCapture(e.pointerId);
                  press.current = { id: n.id, ...toPlan(e), dragged: false };
                }}
                onPointerMove={(e) => {
                  const p = press.current;
                  if (!p || p.id !== n.id) return;
                  const at = toPlan(e);
                  if (!p.dragged && Math.hypot(at.x - p.x, at.y - p.y) < CLICK_SLOP) return;
                  p.dragged = true;
                  setMoving({ id: n.id, ...at });
                }}
                onPointerUp={() => {
                  const p = press.current;
                  press.current = null;
                  if (!p || p.id !== n.id) return;
                  if (p.dragged && moving?.id === n.id) void save(n.id, moving.x, moving.y);
                  else toggle(n.id);
                }}
                onKeyDown={(e) => {
                  if (e.key === "Enter" || e.key === " ") {
                    e.preventDefault();
                    return toggle(n.id);
                  }
                  if (e.key === "Escape") return setSelected(null);
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
        {floor.error ? (
          <p className="lt-error" role="alert">
            {floor.error}
          </p>
        ) : null}
        {selectedCam ? <CameraCard cam={selectedCam} onClose={() => setSelected(null)} /> : null}
        {hops.length && focusedCase ? <RouteStrip hops={hops} caseId={focusedCase.id} evidenceIds={new Set(focusedCase.evidence.map((e) => e.id))} /> : null}
      </section>
      <KnownPlaces />
    </aside>
  );
}

/** The camera clicked on the plan: its latest frame (live when it is streaming) and what is indexed. */
function CameraCard({ cam, onClose }: { cam: CameraInfo; onClose: () => void }) {
  const live = useEvora((s) => s.live[cam.id]);
  const streaming = cam.status === "live" || live === "running" || live === "retrying" || live === "starting";
  // a moment before the end, so the frame exists even when the duration is rounded up
  const last = cam.t0 + Math.max(0, (cam.duration_s ?? 1) - 1);
  return (
    <div className="lt-plan-card" role="region" aria-label={`${cam.name}, latest frame`}>
      {streaming ? (
        <Frame src={liveUrl(cam.id)} alt={`${cam.name}, live`} osd="● LIVE" />
      ) : (
        <Frame src={frameUrl(cam.id, last)} alt={`${cam.name}, last recorded frame`} osd={`${cam.name.toUpperCase()} ${clock(last)}`} />
      )}
      <div className="lt-plan-card-body">
        <p className="lt-plan-card-name">{cam.name}</p>
        <p className="lt-cam-state">
          {streaming
            ? "Streaming now"
            : `Recorded ${day(cam.t0)} ${clock(cam.t0).slice(0, 5)} to ${clock(cam.t0 + (cam.duration_s ?? 0)).slice(0, 5)}${cam.status === "ready" ? ", indexed" : cam.status === "ingesting" ? ", being indexed" : cam.status === "error" ? ", indexing stopped" : ", not indexed yet"}`}
        </p>
        <button type="button" className="lt-link" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}

/** The route as a film strip: the frame at each camera, in order; a frame opens that step's evidence. */
function RouteStrip({ hops, caseId, evidenceIds }: { hops: { camera_id: string; camera_name: string; t_in: number; t_out: number; evidence_id: string }[]; caseId: string; evidenceIds: Set<string> }) {
  const setFocus = useEvora((s) => s.setFocus);
  const active = useEvora((s) => s.focus?.evidenceId);
  return (
    <ol className="lt-route" aria-label="The route, camera by camera">
      {hops.map((h, i) => {
        const open = evidenceIds.has(h.evidence_id);
        const body = (
          <>
            <Frame src={`/api/media/thumb/${encodeURIComponent(h.evidence_id)}.jpg`} alt={`${h.camera_name} at ${clock(h.t_in)}`} />
            <span className="lt-route-step">
              {i + 1} of {hops.length}
            </span>
            <span className="lt-route-where">
              <b>{h.camera_name}</b> {clock(h.t_in).slice(0, 5)}
              {h.t_out - h.t_in >= 1 ? ` for ${Math.round(h.t_out - h.t_in)} s` : ""}
            </span>
          </>
        );
        return (
          <li key={`${h.evidence_id}-${i}`}>
            {open ? (
              <button type="button" className={active === h.evidence_id ? "is-active" : undefined} aria-pressed={active === h.evidence_id} onClick={() => setFocus({ caseId, evidenceId: h.evidence_id })}>
                {body}
              </button>
            ) : (
              <div>{body}</div>
            )}
          </li>
        );
      })}
    </ol>
  );
}

interface FloorPlan {
  url: string | null;
  error: string | null;
  busy: boolean;
  set: (file: File) => void;
  clear: () => void;
}

/** where this browser kept the picture before the API could (v1.15); moved to the server once */
const legacyKey = (workspace: string) => `evora.floorplan.${workspace}`;

/**
 * The picture of the site under the camera nodes, kept by the API with the
 * workspace (GET/PUT/DELETE /api/site/plan), so every browser shows the same
 * one. It reloads when any browser changes it (note kind="site").
 */
function useFloorPlan(workspace: string): FloorPlan {
  const siteVersion = useEvora((s) => s.siteVersion);
  const [url, setUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  // bumped after this browser's own change, in case the change note is missed
  const [mine, setMine] = useState(0);
  const shown = useRef<string | null>(null);

  useEffect(() => {
    let live = true;
    const show = (next: string | null) => {
      if (!live) {
        if (next) URL.revokeObjectURL(next);
        return;
      }
      if (shown.current) URL.revokeObjectURL(shown.current);
      shown.current = next;
      setUrl(next);
    };
    void (async () => {
      try {
        const res = await fetch(apiUrl("/api/site/plan"), { cache: "no-store" });
        if (res.ok) return show(URL.createObjectURL(await res.blob()));
        if (res.status !== 404) return; // the API could not say: keep what is shown
        let old: string | null = null;
        try {
          old = window.localStorage.getItem(legacyKey(workspace));
        } catch {
          /* storage blocked */
        }
        if (!old) return show(null);
        // nothing on the server yet, but this browser kept one: share it
        await endpoints.putSitePlan(await (await fetch(old)).blob());
        try {
          window.localStorage.removeItem(legacyKey(workspace));
        } catch {
          /* already gone */
        }
        if (live) setMine((v) => v + 1);
      } catch {
        /* the API is not answering: keep what is shown */
      }
    })();
    return () => {
      live = false;
    };
  }, [workspace, siteVersion, mine]);

  // the last picture goes when the panel does
  useEffect(
    () => () => {
      if (shown.current) URL.revokeObjectURL(shown.current);
    },
    [],
  );

  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    setError(null);
    try {
      await fn();
      setMine((v) => v + 1);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The API did not take the picture.");
    } finally {
      setBusy(false);
    }
  };

  const set = (file: File) => {
    setError(null);
    if (!/^image\/(png|jpeg|webp)$/.test(file.type)) return setError("Choose a picture of the site: png, jpg or webp.");
    const img = new Image();
    const src = URL.createObjectURL(file);
    img.onload = () => {
      // downscaled here, so a phone photo of the floor plan is well under the 5 MB limit
      const k = Math.min(1, PLAN_MAX_PX / Math.max(img.naturalWidth, img.naturalHeight));
      const c = document.createElement("canvas");
      c.width = Math.max(1, Math.round(img.naturalWidth * k));
      c.height = Math.max(1, Math.round(img.naturalHeight * k));
      const ctx = c.getContext("2d");
      URL.revokeObjectURL(src);
      if (!ctx) return setError("This browser could not read that picture.");
      ctx.fillStyle = "#fff";
      ctx.fillRect(0, 0, c.width, c.height);
      ctx.drawImage(img, 0, 0, c.width, c.height);
      c.toBlob((blob) => (blob ? void run(() => endpoints.putSitePlan(blob)) : setError("This browser could not read that picture.")), "image/jpeg", 0.85);
    };
    img.onerror = () => {
      URL.revokeObjectURL(src);
      setError("That file could not be read as a picture.");
    };
    img.src = src;
  };

  const clear = () => void run(() => endpoints.deleteSitePlan());

  return { url, error, busy, set, clear };
}

function FloorPlanControls({ floor }: { floor: FloorPlan }) {
  const input = useRef<HTMLInputElement>(null);
  return (
    <span className="lt-plan-tools">
      <input
        ref={input}
        type="file"
        accept="image/png,image/jpeg,image/webp"
        hidden
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) floor.set(f);
          e.target.value = "";
        }}
      />
      <button type="button" className="lt-link" disabled={floor.busy} onClick={() => input.current?.click()} title="A drawing or photo of the site, shared with everyone on this workspace">
        {floor.busy ? "Saving…" : floor.url ? "Replace floor plan" : "Add a floor plan"}
      </button>
      {floor.url ? (
        <button type="button" className="lt-link" disabled={floor.busy} onClick={floor.clear}>
          Remove
        </button>
      ) : null}
    </span>
  );
}
