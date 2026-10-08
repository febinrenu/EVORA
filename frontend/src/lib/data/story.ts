// The single investigation the whole experience tells. Acts 5-12 all read from
// here so timestamps, cameras and scores can never drift apart.
import type { QueryPlan } from "@contracts/ts/evora-types";

export const SITE = "North campus";

export const TOTALS = {
  cameras: 24,
  events: 8_421_903,
  tracks: 1_204_921,
  people: 347_203,
  vehicles: 92_821,
} as const;

export const COUNT_BEATS = [
  { value: TOTALS.cameras, label: "Cameras", cls: -1 },
  { value: TOTALS.events, label: "Events", cls: -1 },
  { value: TOTALS.tracks, label: "Object tracks", cls: 2 },
  { value: TOTALS.people, label: "People", cls: 0 },
  { value: TOTALS.vehicles, label: "Vehicles", cls: 1 },
] as const;

export const QUERY = "Did a red car pass through the main gate in the last hour?";

/** Exactly what the planner returns for the query (contracts v1, PLAN Appendix A). */
export const PLAN: QueryPlan = {
  intent: "exists",
  targets: [{ noun: "car", cls: ["car"], attributes: ["red"], embed_text: "a photo of a red car" }],
  place: { text: "main gate", role: "place" },
  action: "pass_through",
  time: { phrase: "in the last hour" },
  camera_ids: [],
  limit: 10,
  unresolved: [{ text: "main gate", role: "place" }],
  source: "fastpath",
};

/** Words of the query that lift out into structured slots, in order of arrival. */
export const PARSE_SLOTS = [
  { key: "object", label: "Object", value: "Vehicle", words: ["car"], code: 'cls: ["car"]' },
  { key: "colour", label: "Colour", value: "Red", words: ["red"], code: 'attributes: ["red"]' },
  { key: "action", label: "Action", value: "Pass through", words: ["pass", "through"], code: '"pass_through"' },
  { key: "place", label: "Place", value: "Main gate", words: ["main", "gate"], code: 'place: "main gate"' },
  { key: "time", label: "Time", value: "Last hour", words: ["last", "hour?"], code: 'phrase: "in the last hour"' },
] as const;

export const FUNNEL = [
  { value: 8_421_903, label: "Events" },
  { value: 14_291, label: "Vehicles" },
  { value: 183, label: "Red objects" },
  { value: 7, label: "Possible matches" },
  { value: 2, label: "High-confidence matches" },
] as const;

export const MATCH = {
  camera: "CAM_04",
  cameraName: "Main gate",
  clock: "09:14:23.042",
  offset: "12:03.042 into cam04_gate.mp4",
  label: "Red sedan",
  score: 0.94,
  trackId: "cam_04:t000318",
  globalId: "g_0007",
} as const;

export const REASONS = [
  { key: "visual", label: "Visual similarity", detail: "siglip 0.312 vs query", value: 0.91 },
  { key: "colour", label: "Colour match", detail: "red, ΔE 6.1 after white balance", value: 0.96 },
  { key: "shape", label: "Vehicle silhouette", detail: "sedan 0.88 · suv 0.07", value: 0.88 },
  { key: "time", label: "Temporal consistency", detail: "tracked 41 frames, 3.4 s", value: 0.93 },
  { key: "cross", label: "Cross-camera consistency", detail: "re-id 0.81 with CAM_07 at +105 s", value: 0.81 },
] as const;

export interface Hop {
  camera: string;
  clock: string;
  place: string;
  /** position along the car's route, 0..1 */
  at: number;
}

export const HOPS: readonly Hop[] = [
  { camera: "CAM_04", clock: "09:14:23", place: "Main gate", at: 0.3 },
  { camera: "CAM_07", clock: "09:16:08", place: "Driveway", at: 0.56 },
  { camera: "CAM_12", clock: "09:22:41", place: "Rear entrance", at: 0.93 },
] as const;

export const RECON_CAMS = ["CAM_04", "CAM_07", "CAM_12", "CAM_18"] as const;

export const MEMORY_QUERY = "What happened near the main gate?";
export const MEMORY_QUESTION = "Which camera shows the main gate?";

export interface PlaceSpec {
  id: string;
  label: string;
  camera: string;
  /** world position in the digital twin (metres) */
  pos: [number, number, number];
  learned: string;
  uses: number;
}

export const PLACES: readonly PlaceSpec[] = [
  { id: "main-gate", label: "Main gate", camera: "CAM_04", pos: [0, 0, 30], learned: "learned from you at 09:31", uses: 4 },
  { id: "lobby", label: "Lobby", camera: "CAM_02", pos: [-25, 0, -8], learned: "learned 2 days ago", uses: 11 },
  { id: "parking", label: "Parking", camera: "CAM_18", pos: [-34, 0, 18], learned: "learned 2 days ago", uses: 7 },
  { id: "rear-entrance", label: "Rear entrance", camera: "CAM_12", pos: [38, 0, -47], learned: "learned yesterday", uses: 3 },
  { id: "loading-bay", label: "Loading bay", camera: "CAM_09", pos: [62, 0, -14], learned: "learned yesterday", uses: 2 },
] as const;

export interface EntitySpec {
  id: string;
  label: string;
  kind: "vehicle" | "person";
  pos: [number, number, number];
  seen: string;
  events: { clock: string; camera: string; what: string }[];
}

export const ENTITIES: readonly EntitySpec[] = [
  {
    id: "red-sedan",
    label: "Red sedan",
    kind: "vehicle",
    pos: [14, 0, -22],
    seen: "3 cameras · 18 min",
    events: [
      { clock: "09:14:23", camera: "CAM_04", what: "Crossed main gate inbound" },
      { clock: "09:16:08", camera: "CAM_07", what: "Driveway, heading north-east" },
      { clock: "09:22:41", camera: "CAM_12", what: "Stopped at rear entrance" },
      { clock: "09:32:10", camera: "CAM_12", what: "Departed towards loading bay" },
    ],
  },
  {
    id: "blue-jacket",
    label: "Blue jacket person",
    kind: "person",
    pos: [-30, 0, 4],
    seen: "2 cameras · 6 min",
    events: [
      { clock: "09:02:51", camera: "CAM_02", what: "Left lobby, carrying a backpack" },
      { clock: "09:05:37", camera: "CAM_18", what: "Walked through parking row B" },
    ],
  },
  {
    id: "black-suv",
    label: "Black SUV",
    kind: "vehicle",
    pos: [-40, 0, 24],
    seen: "1 camera · 47 min",
    events: [
      { clock: "08:21:14", camera: "CAM_18", what: "Parked, bay 12" },
      { clock: "09:08:02", camera: "CAM_18", what: "Still parked (dwell 46 min)" },
    ],
  },
] as const;

/** Timeline act: 08:00 to 12:00 footage day, zooming onto the match. */
export const DAY = {
  startClock: 8 * 3600,
  span: 4 * 3600,
  /** seconds since 08:00 of the red sedan's best frame */
  matchT: (9 - 8) * 3600 + 14 * 60 + 23.042,
  focusT: (9 - 8) * 3600 + 14 * 60 + 22.4,
  minSpan: 3.2,
} as const;

export function clockOf(secondsSince8: number, ms = false): string {
  const t = DAY.startClock + secondsSince8;
  const h = Math.floor(t / 3600);
  const m = Math.floor((t % 3600) / 60);
  const s = Math.floor(t % 60);
  const base = `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  if (!ms) return base;
  return `${base}.${String(Math.floor((t % 1) * 1000)).padStart(3, "0")}`;
}
