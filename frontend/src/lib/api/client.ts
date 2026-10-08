// Typed access to the evora API (contracts v1). In production the UI is served
// by the API itself, so paths are relative; `npm run dev` points at :8700.
import type {
  PathHop,
  Alert,
  StandingQuery,
  Zone,
  Answer,
  CameraInfo,
  ClarifyRequest,
  ClarifyResponse,
  Evidence,
  IngestJob,
  MemoryFact,
  QueryPlan,
} from "@contracts/ts/evora-types";

export type { PathHop, Alert, StandingQuery, Zone, Answer, CameraInfo, ClarifyRequest, ClarifyResponse, Evidence, IngestJob, MemoryFact, QueryPlan };

// `make up` builds with NEXT_PUBLIC_EVORA_API when it serves the UI on its own port
export const API_BASE = process.env.NEXT_PUBLIC_EVORA_API ?? process.env.NEXT_PUBLIC_API_BASE ?? "";

export const apiUrl = (path: string): string => (path.startsWith("http") ? path : `${API_BASE}${path}`);

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
    readonly body: unknown,
  ) {
    super(message);
  }
}

/** FastAPI errors arrive as {detail: string | [...]}: turn them into a sentence for the UI. */
function detailOf(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    if (Array.isArray(d) && d.length && typeof d[0] === "object" && d[0] && "msg" in d[0]) return String((d[0] as { msg: unknown }).msg);
  }
  return fallback;
}

export async function api<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("content-type")) headers.set("content-type", "application/json");
  const res = await fetch(apiUrl(path), { ...init, headers });
  const text = await res.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    body = text;
  }
  if (!res.ok) throw new ApiError(res.status, detailOf(body, `${res.status} ${res.statusText}`), body);
  return body as T;
}

export interface Health {
  ok: boolean;
  version: string;
  workspace: string;
  profile: string;
  onprem: boolean;
  layers_ready: string[];
  egress_blocked?: number;
  blur?: "applied" | "off" | "unavailable";
  /** IANA zone of the site clock, when the API reports it */
  tz?: string;
}

export interface TrackPoint {
  t: number;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  conf?: number;
}

export interface TrackDetail {
  id: string;
  camera_id: string;
  cls: string;
  t_start: number;
  t_end: number;
  attrs: Record<string, unknown>;
  points: TrackPoint[];
}

export interface PastQuery {
  id: string;
  text: string;
  intent: string | null;
  verdict: string | null;
  count: number | null;
  n_results: number | null;
  confidence: number | null;
  created_at: number;
  answer?: Answer | null;
}

export interface ZoneEvent {
  id: string;
  camera_id: string;
  track_id: string;
  kind: string;
  zone_id: string;
  t: number;
  payload: Record<string, unknown>;
}

export interface DoctorCheck {
  id: string;
  title: string;
  status: string;
  detail: string;
  fix: string | null;
}

export interface Doctor {
  verdict: string;
  ok: boolean;
  checks: DoctorCheck[];
  took_s: number;
}

/** GET returning the body plus one response header (e.g. X-Evora-Reid). */
async function withHeader<T>(path: string, header: string): Promise<{ body: T; header: string | null }> {
  const res = await fetch(apiUrl(path));
  if (!res.ok) throw new ApiError(res.status, `${res.status} ${res.statusText}`, null);
  return { body: (await res.json()) as T, header: res.headers.get(header) };
}

export const endpoints = {
  health: () => api<Health>("/api/health"),
  cameras: () => api<CameraInfo[]>("/api/cameras"),
  renameCamera: (id: string, name: string) => api<CameraInfo>(`/api/cameras/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  replay: (ids: string[], analyze = false, speed = 1) => api<unknown>("/api/live/replay", { method: "POST", body: JSON.stringify({ camera_ids: ids, speed, analyze }) }),
  similar: (trackId: string, k = 12) => withHeader<Evidence[]>(`/api/tracks/${encodeURIComponent(trackId)}/similar?k=${k}`, "X-Evora-Reid"),
  path: (globalId: string) => withHeader<PathHop[]>(`/api/globals/${encodeURIComponent(globalId)}/path`, "X-Evora-Reid"),
  pastQueries: (limit = 20) => api<PastQuery[]>(`/api/queries?limit=${limit}&full=true`),
  zoneEvents: (zoneId: string, limit = 200) => api<ZoneEvent[]>(`/api/zones/${encodeURIComponent(zoneId)}/events?limit=${limit}`),
  doctor: () => api<Doctor>("/api/doctor"),
  signer: () => api<{ algorithm: string; fingerprint: string }>("/api/evidence/signer"),
  stopReplay: (ids: string[]) => api<unknown>("/api/live/replay/stop", { method: "POST", body: JSON.stringify({ camera_ids: ids }) }),
  live: () => api<{ streams: { camera_id: string; state: string; error: string | null }[]; analyzers?: { camera_id: string; state: string; error: string | null }[] }>("/api/live"),
  placeCamera: (id: string, xy: [number, number]) => api<CameraInfo>(`/api/cameras/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ site_xy: xy }) }),
  /** audited, 5-minute token; pass as ?unblur= on media and frame routes */
  unblur: (reason: string, evidenceId?: string) => api<{ token: string; expires_at: number }>("/api/media/unblur", { method: "POST", body: JSON.stringify({ reason, evidence_id: evidenceId }) }),
  upload: (files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    return api<CameraInfo[]>("/api/cameras", { method: "POST", body: form });
  },
  ingest: (cameraIds: string[]) => api<IngestJob[]>("/api/ingest", { method: "POST", body: JSON.stringify({ camera_ids: cameraIds }) }),
  memory: () => api<MemoryFact[]>("/api/memory"),
  patchMemory: (id: string, body: { canonical?: string; aliases?: string[]; confirm_aliases?: string[] }) =>
    api<MemoryFact>(`/api/memory/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify(body) }),
  deleteMemory: (id: string) => api<unknown>(`/api/memory/${encodeURIComponent(id)}`, { method: "DELETE" }),
  watches: () => api<StandingQuery[]>("/api/standing"),
  /** 409 carries {clarify: ClarifyRequest}: answer it, then post the same text again */
  watch: (text: string) => api<StandingQuery>("/api/standing", { method: "POST", body: JSON.stringify({ text }) }),
  setWatch: (id: string, active: boolean) => api<StandingQuery>(`/api/standing/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ active }) }),
  alerts: () => api<Alert[]>("/api/alerts"),
  ack: (id: string) => api<Alert>(`/api/alerts/${encodeURIComponent(id)}/ack`, { method: "POST" }),
  track: (id: string) => api<TrackDetail>(`/api/tracks/${encodeURIComponent(id)}`),
  setOnprem: (onprem: boolean) => api<Record<string, unknown>>("/api/settings", { method: "POST", body: JSON.stringify({ onprem }) }),
  /** the route reads the raw request body */
  voice: (audio: Blob) => api<{ text: string }>("/api/voice", { method: "POST", body: audio, headers: { "content-type": audio.type || "audio/webm" } }),
};

export const frameUrl = (cameraId: string, t: number, unblur?: string): string =>
  apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/frame?t=${t.toFixed(3)}${unblur ? `&unblur=${encodeURIComponent(unblur)}` : ""}`);

export const liveUrl = (cameraId: string): string => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/live.mjpg`);

/** Add an unblur token to a media URL. */
export const withUnblur = (url: string, token?: string | null): string => (token ? `${url}${url.includes("?") ? "&" : "?"}unblur=${encodeURIComponent(token)}` : url);
