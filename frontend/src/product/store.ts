// Product state for /app. One store; the stream handlers are the only writers
// for cases, the events feed is the only writer for ingest progress.
import { create } from "zustand";
import { endpoints, type Answer, type CameraInfo, type ClarifyRequest, type ClarifyResponse, type Evidence, type Health, type IngestJob, type MemoryFact, type QueryPlan } from "@/lib/api/client";
import { postStream, type StreamMessage } from "@/lib/api/stream";
import { setSiteZone } from "./format";

export type CaseStatus = "planning" | "searching" | "answered" | "clarify" | "error";

export interface Case {
  id: string;
  question: string;
  askedAt: number;
  status: CaseStatus;
  queryId?: string;
  plan?: QueryPlan;
  evidence: Evidence[];
  answer?: Answer;
  clarify?: ClarifyRequest;
  /** clarifications answered for this question, oldest first */
  resolved: string[];
  verified: Record<string, boolean | null>;
  notes: string[];
  error?: string;
  /** ms from asking to the answer event */
  ttfa?: number;
}

export interface Focus {
  caseId: string;
  evidenceId: string;
}

interface State {
  cases: Case[];
  cameras: CameraInfo[];
  jobs: Record<string, IngestJob>;
  memory: MemoryFact[];
  health: Health | null;
  connected: boolean;
  focus: Focus | null;
  history: string[];
  sessionId: string;

  ask: (text: string) => void;
  clarify: (caseId: string, resp: ClarifyResponse, label: string) => void;
  setFocus: (f: Focus | null) => void;
  refreshCameras: () => Promise<void>;
  refreshMemory: () => Promise<void>;
  refreshHealth: () => Promise<void>;
  upsertCameras: (cams: CameraInfo[]) => void;
  setJob: (job: IngestJob) => void;
  setCameraStatus: (id: string, status: CameraInfo["status"]) => void;
  setConnected: (c: boolean) => void;
}

const newId = (): string => (typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : String(Date.now() + Math.random()));

const isEvidence = (d: Record<string, unknown>): d is Evidence & Record<string, unknown> => typeof d.id === "string" && typeof d.camera_id === "string" && typeof d.t_peak === "number";

export const useEvora = create<State>()((set, get) => {
  const patch = (id: string, fn: (c: Case) => Case) => set((s) => ({ cases: s.cases.map((c) => (c.id === id ? fn(c) : c)) }));

  const handle = (id: string, started: number) => (m: StreamMessage) => {
    switch (m.type) {
      case "plan":
        patch(id, (c) => ({ ...c, plan: m.data as unknown as QueryPlan, status: "searching" }));
        break;
      case "evidence":
        if (isEvidence(m.data)) {
          const ev = m.data;
          patch(id, (c) => (c.evidence.some((e) => e.id === ev.id) ? c : { ...c, evidence: [...c.evidence, ev] }));
        }
        break;
      case "answer": {
        const answer = m.data as unknown as Answer;
        patch(id, (c) => ({
          ...c,
          answer,
          queryId: answer.query_id,
          evidence: answer.evidence?.length ? answer.evidence : c.evidence,
          status: "answered",
          ttfa: c.ttfa ?? performance.now() - started,
        }));
        const first = answer.evidence?.[0];
        if (first) set({ focus: { caseId: id, evidenceId: first.id } });
        break;
      }
      case "verified": {
        const eid = String(m.data.evidence_id ?? "");
        if (eid) patch(id, (c) => ({ ...c, verified: { ...c.verified, [eid]: m.data.verified === true ? true : m.data.verified === false ? false : null } }));
        break;
      }
      case "clarify": {
        const req = m.data as unknown as ClarifyRequest;
        patch(id, (c) => ({ ...c, clarify: req, queryId: req.query_id, status: "clarify" }));
        break;
      }
      case "note": {
        const text = typeof m.data.message === "string" ? m.data.message : typeof m.data.text === "string" ? m.data.text : null;
        if (text) patch(id, (c) => ({ ...c, notes: [...c.notes, text] }));
        break;
      }
      case "error": {
        const message = typeof m.data.message === "string" ? m.data.message : "The answer could not be completed.";
        patch(id, (c) => ({ ...c, error: message, status: c.answer ? c.status : "error" }));
        break;
      }
      case "done":
        patch(id, (c) => (c.status === "planning" || c.status === "searching" ? { ...c, status: c.answer ? "answered" : c.error ? "error" : c.clarify ? "clarify" : "answered" } : c));
        break;
    }
  };

  const run = (id: string, path: string, body: unknown) => {
    const started = performance.now();
    postStream(path, body, handle(id, started)).catch((e: unknown) => {
      patch(id, (c) => ({ ...c, status: "error", error: e instanceof Error ? `Connection lost: ${e.message}` : "Connection lost." }));
    });
  };

  return {
    cases: [],
    cameras: [],
    jobs: {},
    memory: [],
    health: null,
    connected: false,
    focus: null,
    history: [],
    sessionId: newId(),

    ask: (text) => {
      const q = text.trim();
      if (!q) return;
      const id = newId();
      set((s) => ({
        cases: [...s.cases, { id, question: q, askedAt: Date.now(), status: "planning", evidence: [], resolved: [], verified: {}, notes: [] }],
        history: [q, ...s.history.filter((h) => h !== q)].slice(0, 30),
      }));
      run(id, "/api/query", { text: q, session_id: get().sessionId });
    },

    clarify: (caseId, resp, label) => {
      patch(caseId, (c) => ({ ...c, clarify: undefined, status: "searching", resolved: [...c.resolved, label] }));
      run(caseId, "/api/clarify", resp);
      // a clarification teaches a place: the ledger should show it
      window.setTimeout(() => void get().refreshMemory(), 1200);
    },

    setFocus: (focus) => set({ focus }),

    refreshCameras: async () => {
      try {
        set({ cameras: await endpoints.cameras() });
      } catch {
        /* keep what we have; the connection badge says why */
      }
    },
    refreshMemory: async () => {
      try {
        set({ memory: await endpoints.memory() });
      } catch {
        /* keep */
      }
    },
    refreshHealth: async () => {
      try {
        const health = await endpoints.health();
        setSiteZone(health.tz);
        set({ health, connected: true });
      } catch {
        set({ connected: false });
      }
    },
    upsertCameras: (cams) =>
      set((s) => {
        const byId = new Map(s.cameras.map((c) => [c.id, c]));
        for (const c of cams) byId.set(c.id, c);
        return { cameras: [...byId.values()] };
      }),
    setJob: (job) => set((s) => ({ jobs: { ...s.jobs, [job.camera_id]: job } })),
    setCameraStatus: (id, status) => set((s) => ({ cameras: s.cameras.map((c) => (c.id === id ? { ...c, status } : c)) })),
    setConnected: (connected) => set({ connected }),
  };
});
