// Product state for /app. One store; the stream handlers are the only writers
// for cases, the events feed is the only writer for ingest progress.
import { create } from "zustand";
import { endpoints, type PathHop, type Alert, type StandingQuery, type Answer, type CameraInfo, type ClarifyRequest, type ClarifyResponse, type Evidence, type Health, type IngestJob, type MemoryFact, type QueryPlan } from "@/lib/api/client";
import { postStream, type StreamMessage } from "@/lib/api/stream";
import { clock, setSiteZone } from "./format";

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
  /** restored from the API's query log (asked before this page was opened) */
  earlier?: boolean;
  /** what the log says when the stored answer itself is not available */
  summary?: { verdict: string | null; results: number | null };
}

export interface Toast {
  id: string;
  alert: Alert;
}

/** Where the timeline scrub head is, and which lane it is on */
export interface Playhead {
  t: number;
  cameraId: string;
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
  watches: StandingQuery[];
  alerts: Alert[];
  toasts: Toast[];
  playhead: Playhead | null;
  /** replay-as-live stream state per camera: starting | running | retrying | failed | stopped */
  live: Record<string, string>;
  /** live analysis state per camera: starting | running | retrying | stopped | error */
  analysis: Record<string, string>;
  /** on-screen clock reading per camera, while it runs or after it failed (absent once done) */
  clock: Record<string, { state: "reading" | "failed"; error?: string }>;
  drawer: boolean;
  /** text to prefill in the watch form ("Watch for this" on an answer) */
  draft: string;

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
  refreshWatches: () => Promise<void>;
  pushAlert: (a: Alert, historical: boolean) => void;
  dismissToast: (id: string) => void;
  ackAlert: (id: string) => Promise<void>;
  setPlayhead: (p: Playhead | null) => void;
  setDrawer: (open: boolean, draft?: string) => void;
  setLive: (cameraId: string, state: string) => void;
  setAnalysis: (cameraId: string, state: string) => void;
  setClock: (cameraId: string, state: { state: "reading" | "failed"; error?: string } | null) => void;
  /** show an alert's evidence as an entry in the case log */
  openAlert: (a: Alert, watchText: string) => void;
  /** query by example: sightings that look like this track */
  findSimilar: (ev: Evidence) => void;
  /** the identity's route across cameras */
  showPath: (ev: Evidence) => void;
  /** bring back questions asked before this page was opened */
  restoreEarlier: () => Promise<void>;
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

  /** an entry the system writes itself (alerts, query by example, paths) */
  const synthetic = (id: string, question: string, status: CaseStatus, answer?: Answer, evidence: Evidence[] = [], notes: string[] = []) => {
    set((s) => {
      const c: Case = { id, question, askedAt: Date.now(), status, evidence, answer, resolved: [], verified: {}, notes };
      const exists = s.cases.some((x) => x.id === id);
      return { cases: exists ? s.cases.map((x) => (x.id === id ? c : x)) : [...s.cases, c], focus: evidence[0] ? { caseId: id, evidenceId: evidence[0].id } : s.focus };
    });
  };
  const answerOf = (id: string, text: string, evidence: Evidence[], path: PathHop[] = []): Answer =>
    ({ query_id: id, text, verdict: evidence.length ? "found" : "not_found", count: evidence.length, evidence, path, confidence: evidence[0]?.score ?? 0, plan: { intent: path.length ? "path" : "list" } }) as unknown as Answer;

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
    watches: [],
    alerts: [],
    toasts: [],
    playhead: null,
    live: {},
    analysis: {},
    clock: {},
    drawer: false,
    draft: "",

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
    refreshWatches: async () => {
      try {
        const [watches, alerts] = await Promise.all([endpoints.watches(), endpoints.alerts()]);
        set({ watches, alerts: alerts.sort((a, b) => b.t - a.t) });
      } catch {
        /* keep */
      }
    },
    pushAlert: (a, historical) =>
      set((s) => ({
        alerts: s.alerts.some((x) => x.id === a.id) ? s.alerts : [a, ...s.alerts],
        // only new events pop a toast; ones found in earlier footage go quietly to the drawer
        toasts: historical || s.toasts.some((t) => t.id === a.id) ? s.toasts : [...s.toasts, { id: a.id, alert: a }].slice(-3),
      })),
    dismissToast: (id) => set((s) => ({ toasts: s.toasts.filter((t) => t.id !== id) })),
    ackAlert: async (id) => {
      set((s) => ({ alerts: s.alerts.map((a) => (a.id === id ? { ...a, acknowledged: true } : a)), toasts: s.toasts.filter((t) => t.id !== id) }));
      try {
        await endpoints.ack(id);
      } catch {
        /* the drawer refresh will show the real state */
      }
    },
    setPlayhead: (playhead) => set({ playhead }),
    findSimilar: (ev) => {
      if (!ev.track_id) return;
      const id = `similar-${ev.id}`;
      const q = `Where else does this appear? (${ev.camera_name} at ${clock(ev.t_peak)})`;
      synthetic(id, q, "searching");
      endpoints
        .similar(ev.track_id)
        .then(({ body, header }) => {
          const hits = body.filter((e) => e.track_id !== ev.track_id);
          const text = hits.length
            ? `${hits.length} sighting${hits.length > 1 ? "s" : ""} look like the same ${ev.camera_name === hits[0].camera_name ? "object" : "object on other cameras"}, best on ${hits[0].camera_name} at ${clock(hits[0].t_peak)}.`
            : header === "pending"
              ? "Cross-camera matching is not ready yet for this footage, so there is nothing to compare with."
              : "Nothing else looks like this in the indexed footage.";
          synthetic(id, q, "answered", answerOf(id, text, hits));
        })
        .catch((e: unknown) => synthetic(id, q, "error", undefined, [], [e instanceof Error ? e.message : "The search did not run."]));
    },
    showPath: (ev) => {
      if (!ev.global_id) return;
      const id = `path-${ev.global_id}`;
      const q = `Where did it go? (from ${ev.camera_name} at ${clock(ev.t_peak)})`;
      synthetic(id, q, "searching");
      endpoints
        .path(ev.global_id)
        .then(({ body, header }) => {
          const cams = get().cameras;
          // one evidence item per hop so each step has a frame and both timestamps
          const evidence: Evidence[] = body.map((h, i) => {
            const cam = cams.find((c) => c.id === h.camera_id);
            const peak = (h.t_in + h.t_out) / 2;
            return { id: h.evidence_id, camera_id: h.camera_id, camera_name: h.camera_name, t_start: h.t_in, t_end: h.t_out, t_peak: peak, offset_s: cam ? peak - cam.t0 : 0, thumb_url: `/api/media/thumb/${h.evidence_id}.jpg`, clip_url: `/api/media/clip/${h.evidence_id}.mp4`, score: 1, global_id: ev.global_id, why: [`seen on ${h.camera_name} from ${clock(h.t_in)} to ${clock(h.t_out)}`], hop: { index: i + 1, of: body.length } } as Evidence;
          });
          const text = body.length > 1 ? `Seen on ${body.length} cameras: ${body.map((h) => `${h.camera_name} ${clock(h.t_in).slice(0, 5)}`).join(", then ")}.` : body.length === 1 ? `Only seen on ${body[0].camera_name}.` : header === "pending" ? "Cross-camera identities are not linked yet for this footage." : "No route is known for this identity.";
          synthetic(id, q, "answered", answerOf(id, text, evidence, body), evidence);
        })
        .catch((e: unknown) => synthetic(id, q, "error", undefined, [], [e instanceof Error ? e.message : "The path could not be loaded."]));
    },
    restoreEarlier: async () => {
      try {
        const past = await endpoints.pastQueries(20);
        const restored: Case[] = past
          .slice()
          .reverse()
          .map((q) => ({
            id: `past-${q.id}`,
            question: q.text,
            askedAt: q.created_at * 1000,
            status: "answered" as const,
            answer: q.answer && typeof q.answer === "object" ? q.answer : undefined,
            evidence: q.answer?.evidence ?? [],
            resolved: [],
            verified: {},
            notes: [],
            earlier: true,
            summary: { verdict: q.verdict, results: q.n_results },
          }));
        set((s) => ({ cases: [...restored.filter((r) => !s.cases.some((c) => c.id === r.id)), ...s.cases] }));
      } catch {
        /* an older API without /api/queries: start empty */
      }
    },
    setLive: (cameraId, state) => set((s) => ({ live: { ...s.live, [cameraId]: state } })),
    setAnalysis: (cameraId, state) => set((s) => ({ analysis: { ...s.analysis, [cameraId]: state } })),
    setClock: (cameraId, state) =>
      set((s) => {
        const clock = { ...s.clock };
        if (state) clock[cameraId] = state;
        else delete clock[cameraId];
        return { clock };
      }),
    setDrawer: (drawer, draft) => set((s) => ({ drawer, draft: draft ?? s.draft })),
    openAlert: (a, watchText) => {
      const id = `alert-${a.id}`;
      set((s) => {
        if (s.cases.some((c) => c.id === id)) return { focus: { caseId: id, evidenceId: a.evidence.id }, drawer: false };
        const answer = {
          query_id: id,
          text: `${a.evidence.camera_name} at ${clock(a.t)}, for the watch “${watchText}”.`,
          verdict: "found",
          evidence: [a.evidence],
          confidence: a.evidence.score,
          plan: { intent: "standing" },
        } as unknown as Answer;
        const c: Case = { id, question: `Alert: ${watchText}`, askedAt: Date.now(), status: "answered", evidence: [a.evidence], answer, resolved: [], verified: {}, notes: [] };
        return { cases: [...s.cases, c], focus: { caseId: id, evidenceId: a.evidence.id }, drawer: false };
      });
    },
  };
});
