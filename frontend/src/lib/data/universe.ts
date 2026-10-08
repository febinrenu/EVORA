// The synthetic event index rendered as the particle universe. Every particle
// is an event with a time, a camera, a class and a funnel level, so each scene
// is a view of the same data rather than a new effect:
//   lattice  — x = time of day, y = camera stratum, z = class band
//   network  — the particle's camera node
//   funnel   — survives filters up to `level`
//   timeline — x = time again, at any zoom
import { DAY } from "@/lib/data/story";
import { gaussian, mulberry32 } from "@/lib/math";
import type { NetworkNode } from "@/lib/data/site";

export const CLASS = { person: 0, vehicle: 1, object: 2, event: 3 } as const;

/** Lattice extents in world units. */
export const LATTICE = { width: 150, height: 58, depth: 26, cameras: 24 } as const;

export interface UniverseData {
  count: number;
  textCount: number;
  /** lattice position, also the bounding position for three */
  position: Float32Array;
  /** random values 0..1 */
  seed: Float32Array;
  /** time 0..1, camera 0..23, class 0..3, funnel level 0..4 */
  meta: Float32Array;
  /** camera node position for the network morph */
  network: Float32Array;
  /** index into the text-target texture, -1 when not a text particle */
  textIndex: Float32Array;
}

// Activity over the 08:00-12:00 footage day: a morning arrival peak, a smaller
// late-morning bump and a quiet floor. Values in day fraction.
function sampleTime(rand: () => number): number {
  const r = rand();
  if (r < 0.42) return clamp01(0.31 + gaussian(rand) * 0.09);
  if (r < 0.62) return clamp01(0.74 + gaussian(rand) * 0.06);
  return rand();
}
const clamp01 = (v: number) => (v < 0 ? 0 : v > 0.99999 ? 0.99999 : v);

export function buildUniverse(count: number, textCount: number, nodes: NetworkNode[]): UniverseData {
  const rand = mulberry32(84_219_03);
  const position = new Float32Array(count * 3);
  const seed = new Float32Array(count * 4);
  const meta = new Float32Array(count * 4);
  const network = new Float32Array(count * 3);
  const textIndex = new Float32Array(count);

  const { width, height, depth, cameras } = LATTICE;
  const matchT = DAY.matchT / DAY.span;
  let i = 0;

  const write = (t: number, cam: number, cls: number, level: number, z: number, yJitter: number) => {
    if (i >= count) return;
    const x = (t - 0.5) * width;
    const y = (cam / (cameras - 1) - 0.5) * height + yJitter;
    position.set([x, y, z], i * 3);
    seed.set([rand(), rand(), rand(), rand()], i * 4);
    meta.set([t, cam, cls, level], i * 4);
    const n = nodes[cam % nodes.length].pos;
    network.set(n, i * 3);
    textIndex[i] = i < textCount ? i : -1;
    i++;
  };

  // A dense pocket around the match time so zooming into seconds still finds
  // events, including the sedan's own track points (red vehicle, CAM_04).
  const pocket = Math.min(Math.floor(count * 0.008), 3200);
  const pocketSpan = 7 / DAY.span;
  for (let k = 0; k < pocket && i < count; k++) {
    const sedan = k < 64;
    const t = sedan ? matchT - 3.2 / DAY.span + (k / 64) * (6.4 / DAY.span) : matchT + (rand() - 0.5) * pocketSpan;
    const cam = sedan ? 3 : Math.floor(rand() * cameras);
    const cls = sedan ? CLASS.vehicle : pickClass(rand);
    write(t, cam, cls, sedan ? 2 : 0, (cls - 1.5) * (depth / 4) + gaussian(rand) * 1.4, gaussian(rand) * 0.18);
  }

  // Tracks: short streaks of consecutive detections give the field its grain.
  while (i < count) {
    const cls = pickClass(rand);
    const cam = Math.floor(rand() * cameras);
    const len = 3 + Math.floor(rand() * rand() * 34);
    const t0 = sampleTime(rand);
    const dt = (0.00015 + rand() * 0.0012) * (cls === CLASS.vehicle ? 0.6 : 1);
    const z0 = (cls - 1.5) * (depth / 4) + gaussian(rand) * 2.1;
    const dz = (rand() - 0.5) * 0.35;
    const level = cls === CLASS.vehicle ? (rand() < 0.065 ? 2 : 1) : 0;
    for (let k = 0; k < len && i < count; k++) {
      write(clamp01(t0 + k * dt), cam, cls, level, z0 + k * dz, gaussian(rand) * 0.16);
    }
  }

  // Exactly seven candidates and two high-confidence matches among the red vehicles.
  const red: number[] = [];
  for (let k = textCount; k < count && red.length < 400; k++) if (meta[k * 4 + 3] === 2) red.push(k);
  const step = Math.max(1, Math.floor(red.length / 7));
  for (let k = 0; k < Math.min(7, red.length); k++) meta[red[k * step] * 4 + 3] = k < 2 ? 4 : 3;

  return { count, textCount, position, seed, meta, network, textIndex };
}

function pickClass(rand: () => number): number {
  const r = rand();
  if (r < 0.41) return CLASS.person;
  if (r < 0.56) return CLASS.vehicle;
  if (r < 0.8) return CLASS.object;
  return CLASS.event;
}
