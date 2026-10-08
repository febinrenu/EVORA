// Typed access to the evora API (contracts v1). In production the UI is served
// by the API itself, so paths are relative; `npm run dev` points at :8700.
import type {
  Answer,
  CameraInfo,
  ClarifyRequest,
  ClarifyResponse,
  Evidence,
  IngestJob,
  MemoryFact,
  QueryPlan,
} from "@contracts/ts/evora-types";

export type { Answer, CameraInfo, ClarifyRequest, ClarifyResponse, Evidence, IngestJob, MemoryFact, QueryPlan };

export const API_BASE = process.env.NEXT_PUBLIC_API_BASE ?? "";

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

export const endpoints = {
  health: () => api<Health>("/api/health"),
  cameras: () => api<CameraInfo[]>("/api/cameras"),
  renameCamera: (id: string, name: string) => api<CameraInfo>(`/api/cameras/${encodeURIComponent(id)}`, { method: "PATCH", body: JSON.stringify({ name }) }),
  upload: (files: File[]) => {
    const form = new FormData();
    for (const f of files) form.append("files", f, f.name);
    return api<CameraInfo[]>("/api/cameras", { method: "POST", body: form });
  },
  ingest: (cameraIds: string[]) => api<IngestJob[]>("/api/ingest", { method: "POST", body: JSON.stringify({ camera_ids: cameraIds }) }),
  memory: () => api<MemoryFact[]>("/api/memory"),
  track: (id: string) => api<TrackDetail>(`/api/tracks/${encodeURIComponent(id)}`),
  setOnprem: (onprem: boolean) => api<Record<string, unknown>>("/api/settings", { method: "POST", body: JSON.stringify({ onprem }) }),
  /** the route reads the raw request body */
  voice: (audio: Blob) => api<{ text: string }>("/api/voice", { method: "POST", body: audio, headers: { "content-type": audio.type || "audio/webm" } }),
};

export const frameUrl = (cameraId: string, t: number): string => apiUrl(`/api/cameras/${encodeURIComponent(cameraId)}/frame?t=${t.toFixed(3)}`);
