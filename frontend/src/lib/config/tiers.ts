// What each performance tier is allowed to spend. The detector in
// lib/perf/tier.ts picks a tier; the runtime governor may step it down.

export type Tier = "high" | "medium" | "low";

export interface TierSpec {
  /** total event particles (one draw call) */
  particles: number;
  /** particles that can form typography (title, numbers) */
  textParticles: number;
  dprCap: number;
  /** curl-noise drift in the vertex shader */
  curl: boolean;
  /** live CCTV render targets for the twin; 0 = stills only */
  liveFeeds: number;
  /** antialias on the main renderer */
  antialias: boolean;
  /** camera flights; when false acts cut through black instead */
  flights: boolean;
}

export const TIERS: Record<Tier, TierSpec> = {
  high: { particles: 420_000, textParticles: 90_000, dprCap: 2, curl: true, liveFeeds: 3, antialias: true, flights: true },
  medium: { particles: 160_000, textParticles: 48_000, dprCap: 1.5, curl: false, liveFeeds: 1, antialias: false, flights: true },
  low: { particles: 36_000, textParticles: 18_000, dprCap: 1, curl: false, liveFeeds: 0, antialias: false, flights: false },
};

/** Governor: frame time budget and how long it must be exceeded before stepping down. */
export const GOVERNOR = {
  budgetMs: 20,
  windowFrames: 60,
  /** DPR steps tried before the particle count is cut */
  dprSteps: [2, 1.6, 1.3, 1],
  /** share of particles kept at each cut */
  particleSteps: [1, 0.6, 0.35],
  /** ignore the first frames after a scene activates (shader warm-up) */
  graceFrames: 45,
} as const;
