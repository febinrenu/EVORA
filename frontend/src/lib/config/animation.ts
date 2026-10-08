// Central choreography config. Every act length, easing and pacing value lives
// here so the story can be re-timed without hunting through components.
//
// Units: one scroll unit = one viewport height of scrolling. The master
// timeline uses the same units for its duration, so timeline time == scroll.

export type ActId =
  | "opening"
  | "universe"
  | "network"
  | "wall"
  | "ask"
  | "search"
  | "evidence"
  | "trace"
  | "memory"
  | "memoryMap"
  | "timeline"
  | "reconstruct"
  | "finale";

export type ChapterId = "discover" | "search" | "trace" | "remember" | "reconstruct";

export interface ActSpec {
  id: ActId;
  chapter: ChapterId;
  /** scroll length on desktop, in viewport heights */
  length: number;
  /** accessible title used for skip links and the hidden outline */
  title: string;
}

export const ACTS: readonly ActSpec[] = [
  { id: "opening", chapter: "discover", length: 3.2, title: "Everything leaves a trace" },
  { id: "universe", chapter: "discover", length: 7.0, title: "The event universe" },
  { id: "network", chapter: "discover", length: 5.0, title: "The camera network" },
  { id: "wall", chapter: "discover", length: 4.2, title: "The camera wall" },
  { id: "ask", chapter: "search", length: 4.0, title: "Ask anything" },
  { id: "search", chapter: "search", length: 5.2, title: "Search through reality" },
  { id: "evidence", chapter: "search", length: 4.2, title: "The evidence" },
  { id: "trace", chapter: "trace", length: 6.4, title: "Cross-camera trace" },
  { id: "memory", chapter: "remember", length: 5.0, title: "Clarify once" },
  { id: "memoryMap", chapter: "remember", length: 4.4, title: "The memory map" },
  { id: "timeline", chapter: "remember", length: 4.8, title: "Zoom into time" },
  { id: "reconstruct", chapter: "reconstruct", length: 5.0, title: "Reconstruct" },
  { id: "finale", chapter: "reconstruct", length: 4.4, title: "Search the memory of a place" },
] as const;

export const CHAPTERS: readonly { id: ChapterId; label: string }[] = [
  { id: "discover", label: "Discover" },
  { id: "search", label: "Search" },
  { id: "trace", label: "Trace" },
  { id: "remember", label: "Remember" },
  { id: "reconstruct", label: "Reconstruct" },
];

/** Mobile and tablet compress the scroll so the story keeps its rhythm on short screens. */
export const LENGTH_SCALE = { desktop: 1, tablet: 0.85, mobile: 0.72 } as const;

export const EASE = {
  /** default for scrubbed choreography: scroll already supplies the curve */
  scrub: "none",
  enter: "power3.out",
  leave: "power2.in",
  move: "power2.inOut",
  camera: "sine.inOut",
  snap: "expo.out",
} as const;

/** Share of each act left still on purpose. Silence is part of the pacing. */
export const BREATH = 0.15;

export const LENIS = {
  lerp: 0.085,
  wheelMultiplier: 0.9,
  touchMultiplier: 1.4,
} as const;

export const CAMERA = {
  fov: 42,
  near: 0.1,
  far: 900,
  /** distance of camera-attached layers (text, timeline, funnel) */
  hudDepth: 60,
} as const;

/** CCTV look: footage holds frames like a real NVR. */
export const CCTV = {
  fps: 12,
  feedSize: [480, 270] as const,
} as const;

export const DOM = {
  /** pointer lerp for the reticle cursor, per 60 fps frame */
  cursorLerp: 0.22,
  magnetStrength: 0.32,
  magnetRadius: 90,
} as const;
