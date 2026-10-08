"use client";

// Clarify card (P4.8): the yellow-tabbed question, a contact sheet of camera
// frames, an optional "mark the exact spot" step where the operator draws the
// line or area with the grease pencil, or a typed answer. Used by questions
// and by watches; whoever shows it decides what happens with the answer.
import { useRef, useState } from "react";
import { getStroke } from "perfect-freehand";
import type { ClarifyRequest, ClarifyResponse, Zone } from "@/lib/api/client";
import { strokePath } from "@/lib/draw";
import { useEvora, type Case } from "./store";
import { Frame } from "./Frame";

type Option = NonNullable<ClarifyRequest["options"]>[number];

export function ClarifyPanel({ req, onAnswer }: { req: ClarifyRequest; onAnswer: (resp: ClarifyResponse, label: string) => void }) {
  const [typed, setTyped] = useState("");
  const [chosen, setChosen] = useState<Option | null>(null);
  const what = req.referent.text;
  const placeLike = req.kind === "choose_camera" && req.referent.role === "place";

  const answerCamera = (o: Option, zone: Zone | null) =>
    onAnswer({ query_id: req.query_id, camera_id: o.camera_id, zone }, `“${what}” is ${o.camera_name}${zone ? (zone.kind === "line" ? ", marked line" : ", marked area") : ""}`);

  return (
    <div className="lt-clarify" role="group" aria-label="One question before answering">
      <span className="lt-tab">Clarify once</span>
      <p className="lt-clarify-q">{req.question}</p>
      {chosen ? (
        <SpotMarker option={chosen} what={what} onBack={() => setChosen(null)} onDone={(zone) => answerCamera(chosen, zone)} />
      ) : (
        <>
          {req.options?.length ? (
            <ul className="lt-contact">
              {req.options.map((o) => (
                <li key={o.camera_id}>
                  <button type="button" onClick={() => (placeLike && req.allow_region !== false ? setChosen(o) : answerCamera(o, null))}>
                    <Frame src={o.thumb_url} alt={`${o.camera_name}, current frame`} />
                    <span>{o.camera_name}</span>
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
          <form
            className="lt-clarify-typed"
            onSubmit={(e) => {
              e.preventDefault();
              if (typed.trim()) onAnswer({ query_id: req.query_id, text: typed.trim() }, `“${what}” is ${typed.trim()}`);
            }}
          >
            <label>
              <span>Or answer in words</span>
              <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="camera 2, the lobby one…" />
            </label>
            <button type="submit" disabled={!typed.trim()}>
              Save place
            </button>
          </form>
        </>
      )}
    </div>
  );
}

/** The question card inside a case: answers continue the paused query. */
export function ClarifyCard({ c, req }: { c: Case; req: ClarifyRequest }) {
  const clarify = useEvora((s) => s.clarify);
  return <ClarifyPanel req={req} onAnswer={(resp, label) => clarify(c.id, resp, label)} />;
}

type Mode = "line" | "area";

/** Draw the gate line (drag) or the area (click points) on the chosen camera's frame. */
function SpotMarker({ option, what, onBack, onDone }: { option: Option; what: string; onBack: () => void; onDone: (zone: Zone | null) => void }) {
  const [mode, setMode] = useState<Mode>("line");
  const [stroke, setStroke] = useState<number[][]>([]);
  const [area, setArea] = useState<[number, number][]>([]);
  const drawing = useRef(false);
  const svg = useRef<SVGSVGElement>(null);

  const local = (e: React.PointerEvent): [number, number] => {
    const r = svg.current?.getBoundingClientRect();
    if (!r) return [0, 0];
    return [Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)), Math.min(1, Math.max(0, (e.clientY - r.top) / r.height))];
  };

  const line = stroke.length > 1 ? ([stroke[0].slice(0, 2), stroke[stroke.length - 1].slice(0, 2)] as [number, number][]) : null;
  const ready = mode === "line" ? line !== null : area.length >= 3;
  const zone = (): Zone | null =>
    mode === "line" && line ? { id: "new", camera_id: option.camera_id, kind: "line", points: line, direction: "any" } : mode === "area" && area.length >= 3 ? { id: "new", camera_id: option.camera_id, kind: "polygon", points: area, direction: "any" } : null;

  // strokes are drawn in a 1600x900 box so the pencil keeps its weight at any size
  const W = 1600;
  const H = 900;
  const pencil = stroke.length > 1 ? strokePath(getStroke(stroke.map(([x, y, p]) => [x * W, y * H, p]), { size: 14, thinning: 0.5, smoothing: 0.5, streamline: 0.4 })) : "";

  return (
    <div className="lt-spot">
      <div className="lt-spot-tools" role="radiogroup" aria-label="What to mark">
        <button type="button" role="radio" aria-checked={mode === "line"} onClick={() => (setMode("line"), setArea([]))}>
          A line people cross
        </button>
        <button type="button" role="radio" aria-checked={mode === "area"} onClick={() => (setMode("area"), setStroke([]))}>
          An area
        </button>
        <span className="lt-spot-hint">{mode === "line" ? `Drag across ${what}.` : area.length < 3 ? "Click the corners of the area." : "Close it with Save place, or keep adding corners."}</span>
      </div>
      <div className="lt-spot-frame">
        <Frame src={option.thumb_url} alt={`${option.camera_name}: mark ${what}`} />
        <svg
          ref={svg}
          viewBox={`0 0 ${W} ${H}`}
          preserveAspectRatio="none"
          aria-label={`Drawing surface over ${option.camera_name}`}
          onPointerDown={(e) => {
            const [x, y] = local(e);
            if (mode === "line") {
              drawing.current = true;
              (e.target as Element).setPointerCapture(e.pointerId);
              setStroke([[x, y, e.pressure || 0.5]]);
            } else setArea((a) => [...a, [x, y]]);
          }}
          onPointerMove={(e) => {
            if (!drawing.current) return;
            const [x, y] = local(e);
            setStroke((s) => [...s, [x, y, e.pressure || 0.5]]);
          }}
          onPointerUp={() => (drawing.current = false)}
        >
          {pencil ? <path className="lt-pencil" d={pencil} /> : null}
          {line ? <line className="lt-pencil-axis" x1={line[0][0] * W} y1={line[0][1] * H} x2={line[1][0] * W} y2={line[1][1] * H} /> : null}
          {area.length ? <polygon className="lt-pencil-area" points={area.map(([x, y]) => `${x * W},${y * H}`).join(" ")} /> : null}
          {area.map(([x, y], i) => (
            <circle key={i} className="lt-pencil-dot" cx={x * W} cy={y * H} r={9} />
          ))}
        </svg>
      </div>
      <div className="lt-spot-actions">
        <button type="button" className="lt-save" disabled={!ready} onClick={() => onDone(zone())}>
          Save place
        </button>
        <button type="button" onClick={() => onDone(null)}>
          Use the whole view
        </button>
        <button type="button" onClick={() => (stroke.length || area.length ? (setStroke([]), setArea([])) : onBack())}>
          {stroke.length || area.length ? "Clear" : "Choose another camera"}
        </button>
      </div>
    </div>
  );
}
