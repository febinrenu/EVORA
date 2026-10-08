// One coordinate system for the whole story (metres, y up): the abstract camera
// network floats above the same footprint the digital twin later occupies, so
// "CAM_04" is in the same place in every scene.
import { CatmullRomCurve3, Vector3 } from "three";
import { mulberry32 } from "@/lib/math";

export interface CameraPose {
  id: string;
  name: string;
  pos: [number, number, number];
  look: [number, number, number];
}

/** Cameras that exist in the digital twin, with real poses. */
export const TWIN_CAMS: readonly CameraPose[] = [
  { id: "CAM_04", name: "Main gate", pos: [12.5, 6.8, 21], look: [0.5, 0.6, 33] },
  { id: "CAM_07", name: "Driveway", pos: [-11, 6.2, -2], look: [8.5, 0.4, -17] },
  { id: "CAM_12", name: "Rear entrance", pos: [21, 7.2, -30], look: [36, 0.6, -45] },
  { id: "CAM_18", name: "Parking", pos: [-16, 8.5, 36], look: [-38, 0, 20] },
  { id: "CAM_02", name: "Lobby", pos: [-9, 5.2, 4], look: [-24, 1.2, -3] },
  { id: "CAM_09", name: "Loading bay", pos: [50, 6.5, 2], look: [63, 1, -14] },
] as const;

export const twinCam = (id: string): CameraPose => {
  const c = TWIN_CAMS.find((p) => p.id === id);
  if (!c) throw new Error(`unknown twin camera ${id}`);
  return c;
};

/** The red sedan's route: public road, through the gate, up the driveway, to the rear entrance. */
export const CAR_ROUTE = new CatmullRomCurve3(
  [
    new Vector3(-130, 0, 45),
    new Vector3(-60, 0, 45),
    new Vector3(-18, 0, 45),
    new Vector3(-3, 0, 41),
    new Vector3(0, 0, 33),
    new Vector3(0.5, 0, 18),
    new Vector3(2.5, 0, 2),
    new Vector3(9, 0, -16),
    new Vector3(21, 0, -28),
    new Vector3(31, 0, -38),
    new Vector3(36.5, 0, -45.5),
  ],
  false,
  "centripetal",
);

/** Route parameter (arc-length) nearest to a world point. */
function uNear(target: Vector3): number {
  let best = 0;
  let bestD = Infinity;
  const p = new Vector3();
  for (let i = 0; i <= 2000; i++) {
    const u = i / 2000;
    CAR_ROUTE.getPointAt(u, p);
    const d = p.distanceToSquared(target);
    if (d < bestD) {
      bestD = d;
      best = u;
    }
  }
  return best;
}

/** Where along the route each hop camera sees the car best. */
export const HOP_U = {
  CAM_04: uNear(new Vector3(0.3, 0, 32)),
  CAM_07: uNear(new Vector3(8.5, 0, -16.5)),
  CAM_12: uNear(new Vector3(35.5, 0, -44.5)),
} as const;

/** The blue-jacket person walks from the lobby across to the parking rows. */
export const PERSON_ROUTE = new CatmullRomCurve3(
  [new Vector3(-22, 0, -3), new Vector3(-16, 0, 6), new Vector3(-24, 0, 14), new Vector3(-36, 0, 17), new Vector3(-46, 0, 15)],
  false,
  "centripetal",
);

export const isTwinCam = (id: string): boolean => TWIN_CAMS.some((c) => c.id === id);

export interface NetworkNode {
  id: string;
  label: string;
  /** index into the 3x3 footage atlas */
  cell: number;
  pos: [number, number, number];
}

const PEOPLE_FEEDS = ["Terrace", "Terrace east", "Terrace stair", "Terrace north", "Passage", "Passage west", "Passage exit", "Lab hall", "Lab bench"];

/** 24 network nodes: the six twin cameras keep their positions, the rest are placed with blue-noise sampling. */
export function buildNetwork(count = 24): NetworkNode[] {
  const rand = mulberry32(2404);
  const fixed = new Map(TWIN_CAMS.map((c) => [c.id, c]));
  const placed: [number, number][] = TWIN_CAMS.map((c) => [c.pos[0], c.pos[2]]);
  const nodes: NetworkNode[] = [];
  let people = 0;
  for (let i = 0; i < count; i++) {
    const id = `CAM_${String(i + 1).padStart(2, "0")}`;
    const twin = fixed.get(id);
    let x: number;
    let z: number;
    if (twin) {
      [x, , z] = twin.pos;
    } else {
      // best-candidate sampling inside an ellipse over the site footprint
      let bx = 0;
      let bz = 0;
      let bestD = -1;
      for (let k = 0; k < 24; k++) {
        const a = rand() * Math.PI * 2;
        const r = Math.sqrt(rand());
        const cx = 8 + Math.cos(a) * r * 92;
        const cz = -6 + Math.sin(a) * r * 62;
        let d = Infinity;
        for (const [px, pz] of placed) d = Math.min(d, (px - cx) ** 2 + (pz - cz) ** 2);
        if (d > bestD) {
          bestD = d;
          bx = cx;
          bz = cz;
        }
      }
      x = bx;
      z = bz;
      placed.push([x, z]);
    }
    const y = 14 + Math.sin(x * 0.05) * 5 + Math.cos(z * 0.07) * 4 + (twin ? 4 : rand() * 8);
    // People feeds take atlas cells in order, so the first nine non-twin nodes
    // show nine different views and their labels match their footage.
    const cell = twin ? i % 9 : people++ % 9;
    nodes.push({ id, label: twin ? twin.name : PEOPLE_FEEDS[cell], cell, pos: [x, y, z] });
  }
  return nodes;
}

/** Links: a minimum spanning tree plus each node's nearest extra neighbour, so the graph reads as a site, not a mesh. */
export function buildLinks(nodes: NetworkNode[]): [number, number][] {
  const n = nodes.length;
  const dist = (a: number, b: number) => {
    const [ax, ay, az] = nodes[a].pos;
    const [bx, by, bz] = nodes[b].pos;
    return (ax - bx) ** 2 + (ay - by) ** 2 * 0.3 + (az - bz) ** 2;
  };
  const inTree = new Set<number>([0]);
  const links: [number, number][] = [];
  while (inTree.size < n) {
    let best: [number, number] = [0, 0];
    let bestD = Infinity;
    for (const a of inTree)
      for (let b = 0; b < n; b++) {
        if (inTree.has(b)) continue;
        const d = dist(a, b);
        if (d < bestD) {
          bestD = d;
          best = [a, b];
        }
      }
    inTree.add(best[1]);
    links.push(best);
  }
  const key = (a: number, b: number) => (a < b ? `${a}-${b}` : `${b}-${a}`);
  const seen = new Set(links.map(([a, b]) => key(a, b)));
  for (let a = 0; a < n; a++) {
    const order = [...Array(n).keys()].filter((b) => b !== a).sort((p, q) => dist(a, p) - dist(a, q));
    for (const b of order.slice(0, 2)) {
      const k = key(a, b);
      if (!seen.has(k)) {
        seen.add(k);
        links.push([a, b]);
        break;
      }
    }
  }
  return links;
}
