// Grease-pencil marks (PLAN §10.5): the hand-drawn circle around an answer and
// freehand strokes, both from perfect-freehand so they read as one hand.
import { getStroke } from "perfect-freehand";
import { mulberry32 } from "@/lib/math";

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

/** Stable 32-bit seed from an id, so a mark looks the same on every render. */
export function seedOf(id: string): number {
  let h = 2166136261;
  for (let i = 0; i < id.length; i++) h = Math.imul(h ^ id.charCodeAt(i), 16777619);
  return h >>> 0;
}
