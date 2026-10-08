// DOM layers that follow WebGL objects: the evidence box, memory-map labels,
// reconstruction card labels and timeline ticks. Each is a frame hook that
// returns immediately unless its act is on screen, and writes only transforms
// and (rarely) text.
import { getStroke } from "perfect-freehand";
import { scaleLinear } from "d3-scale";
import { Box3, Vector3 } from "three";
import { S } from "@/animation/sceneState";
import { ACTS, type ActId } from "@/lib/config/animation";
import { twinCam } from "@/lib/data/site";
import { DAY, ENTITIES, PLACES, RECON_CAMS, clockOf } from "@/lib/data/story";
import { timeWindow } from "@/lib/data/timeWindow";
import { mulberry32 } from "@/lib/math";
import type { Engine } from "@/components/webgl/Engine";

const actIndex = (id: ActId) => ACTS.findIndex((a) => a.id === id);
const near = (id: ActId, slack = 0) => {
  const i = actIndex(id);
  return S.act >= i - slack && S.act <= i + slack;
};

const tmp = { x: 0, y: 0, z: 0 };
const v = new Vector3();

/** SVG path from perfect-freehand's outline polygon. */
export function strokePath(points: number[][]): string {
  if (!points.length) return "";
  const d = points.reduce<(string | number)[]>(
    (acc, [x0, y0], i, arr) => {
      const [x1, y1] = arr[(i + 1) % arr.length];
      acc.push(x0, y0, (x0 + x1) / 2, (y0 + y1) / 2);
      return acc;
    },
    ["M", ...points[0], "Q"],
  );
  return `${d.join(" ")} Z`;
}

/** A slightly irregular hand-drawn ellipse, stable for a seed. */
export function handEllipse(w: number, h: number, seed: number, progress: number, size = 3.2): string {
  const rand = mulberry32(seed);
  const pts: number[][] = [];
  const turns = 1.12;
  const n = 90;
  const start = rand() * Math.PI * 2;
  const count = Math.max(2, Math.floor(n * progress));
  for (let i = 0; i < count; i++) {
    const t = (i / n) * turns * Math.PI * 2 + start;
    const wob = 1 + Math.sin(t * 3 + rand() * 0.6) * 0.035 + (i / n) * 0.06;
    pts.push([w / 2 + Math.cos(t) * (w / 2) * wob, h / 2 + Math.sin(t) * (h / 2) * wob * 0.97, 0.5 + 0.5 * Math.sin((i / n) * Math.PI)]);
  }
  return strokePath(getStroke(pts, { size, thinning: 0.6, smoothing: 0.6, streamline: 0.4, simulatePressure: false, last: progress >= 1 }));
}

/** Inverse of the CCTV barrel (shader samples uv' = c * (1 + k r^2)); two Newton-free fixed-point steps are plenty. */
function unbarrel(x: number, y: number, w: number, h: number, k: number): [number, number] {
  let ux = x / w - 0.5;
  let uy = y / h - 0.5;
  const tx = ux;
  const ty = uy;
  for (let i = 0; i < 3; i++) {
    const r2 = ux * ux + uy * uy;
    ux = tx / (1 + k * r2);
    uy = ty / (1 + k * r2);
  }
  return [(ux + 0.5) * w, (uy + 0.5) * h];
}

export function evidenceBox(root: HTMLElement): (e: Engine) => void {
  const box = root.querySelector<HTMLElement>('[data-act="evidence"] [data-el="box"]');
  const path = box?.querySelector<SVGPathElement>("[data-ring]");
  const svg = box?.querySelector<SVGSVGElement>("svg");
  const bounds = new Box3(new Vector3(-1.05, 0, -2.4), new Vector3(1.05, 1.95, 2.4));
  const corner = new Vector3();
  let lastKey = "";
  return (engine) => {
    if (!box || !path || !svg) return;
    if (!near("evidence") || S.evidence.box <= 0.001 || !engine.twin) {
      if (box.style.opacity !== "0") box.style.opacity = "0";
      return;
    }
    const car = engine.twin.car;
    car.updateMatrixWorld();
    const { width: W, height: H } = engine.viewport;
    let x0 = Infinity;
    let y0 = Infinity;
    let x1 = -Infinity;
    let y1 = -Infinity;
    for (let i = 0; i < 8; i++) {
      corner.set(i & 1 ? bounds.max.x : bounds.min.x, i & 2 ? bounds.max.y : bounds.min.y, i & 4 ? bounds.max.z : bounds.min.z).applyMatrix4(car.matrixWorld);
      engine.project(corner, tmp);
      const [px, py] = unbarrel(tmp.x, tmp.y, W, H, 0.16 * S.twin.cctv);
      x0 = Math.min(x0, px);
      y0 = Math.min(y0, py);
      x1 = Math.max(x1, px);
      y1 = Math.max(y1, py);
    }
    const pad = 18;
    const w = x1 - x0 + pad * 2;
    const h = y1 - y0 + pad * 2;
    box.style.opacity = "1";
    box.style.transform = `translate3d(${x0 - pad}px, ${y0 - pad}px, 0)`;
    box.style.width = `${w}px`;
    box.style.height = `${h}px`;
    box.style.setProperty("--box", String(S.evidence.box));
    const key = `${Math.round(w / 6)}:${Math.round(h / 6)}:${Math.round(S.evidence.box * 60)}`;
    if (key !== lastKey) {
      lastKey = key;
      svg.setAttribute("viewBox", `0 0 ${w} ${h}`);
      path.setAttribute("d", handEllipse(w, h, 4231, S.evidence.box, 3.6));
    }
  };
}

/** Labels pinned to places and entities in the memory map. */
export function memoryLabels(root: HTMLElement): (e: Engine) => void {
  const layer = root.querySelector<HTMLElement>('[data-act="memoryMap"]');
  const pin = (id: string) => layer?.querySelector<HTMLElement>(`[data-label="${id}"]`)?.parentElement ?? null;
  const items = PLACES.map((p) => ({ el: pin(p.id), pos: new Vector3(p.pos[0], 12, p.pos[2]) })).concat(
    ENTITIES.map((e) => ({ el: pin(e.id), pos: new Vector3(e.pos[0], 11.5, e.pos[2]) })),
  );
  return (engine) => {
    if (!near("memoryMap")) return;
    for (const it of items) {
      if (!it.el) continue;
      engine.project(it.pos, tmp);
      const off = tmp.z > 1 || tmp.x < -200 || tmp.x > engine.viewport.width + 200;
      it.el.style.transform = `translate3d(${tmp.x}px, ${tmp.y}px, 0)`;
      it.el.style.visibility = off ? "hidden" : "";
    }
  };
}

export function reconLabels(root: HTMLElement): (e: Engine) => void {
  const layer = root.querySelector<HTMLElement>('[data-act="reconstruct"]');
  const items = RECON_CAMS.map((id) => {
    const c = twinCam(id);
    return { el: layer?.querySelector<HTMLElement>(`[data-card="${id}"]`), pos: new Vector3(c.pos[0], 30 + 7.4, c.pos[2]) };
  });
  return (engine) => {
    if (!near("reconstruct")) return;
    for (const it of items) {
      if (!it.el) continue;
      v.copy(it.pos);
      engine.project(v, tmp);
      it.el.style.transform = `translate3d(${tmp.x}px, ${tmp.y}px, 0)`;
    }
  };
}

const STEPS = [3600, 1800, 900, 600, 300, 120, 60, 30, 15, 10, 5, 2, 1, 0.5, 0.25];

/** Timeline ticks: the same mapping as the particle shader (shared timeWindow). */
export function timelineTicks(root: HTMLElement): (e: Engine) => void {
  const layer = root.querySelector<HTMLElement>('[data-act="timeline"]');
  const ticks = Array.from(layer?.querySelectorAll<HTMLElement>("[data-tick]") ?? []);
  const readout = layer?.querySelector<HTMLElement>('[data-el="span"]');
  const marker = layer?.querySelector<HTMLElement>('[data-el="match"]');
  const scale = scaleLinear();
  const text = new Array<string>(ticks.length).fill("");
  let lastSpan = "";
  return (engine) => {
    if (!near("timeline")) return;
    const { width: W } = engine.viewport;
    const { focus, span } = timeWindow(S.universe.zoom);
    scale.domain([focus - span / 2, focus + span / 2]).range([0, W]);
    const step = STEPS.find((s) => span / s >= 5) ?? 0.25;
    const ms = step < 1;
    const first = Math.ceil((focus - span * 0.55) / step) * step;
    for (let i = 0; i < ticks.length; i++) {
      const t = first + i * step;
      const x = scale(t);
      const el = ticks[i];
      const inView = x > -40 && x < W + 40;
      el.style.visibility = inView ? "" : "hidden";
      if (!inView) continue;
      el.style.transform = `translate3d(${x}px, 0, 0)`;
      const label = step >= 60 ? clockOf(t).slice(0, 5) : clockOf(t, ms);
      if (label !== text[i]) {
        text[i] = label;
        el.textContent = label;
      }
    }
    if (readout) {
      const s = span >= 60 ? `${(span / 3600).toFixed(2)} h` : `${span.toFixed(span < 10 ? 2 : 1)} s`;
      if (s !== lastSpan) {
        lastSpan = s;
        readout.textContent = s;
      }
    }
    if (marker) {
      const x = scale(DAY.matchT);
      const show = S.universe.zoom > 0.78;
      marker.style.transform = `translate3d(${x}px, 0, 0)`;
      marker.style.opacity = show ? String(Math.min(1, (S.universe.zoom - 0.78) * 8)) : "0";
    }
  };
}
