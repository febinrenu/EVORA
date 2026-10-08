// The evaluation report as /api/report serves it: M3's eval/reports/report.json
// (eval/report.py `build_report`). An older {eval, ablations} shape is still
// understood. Everything is optional: the page says "not measured" rather than
// inventing a number, and every value travels with its n.
import { api } from "@/lib/api/client";

export interface Metric {
  value: number | null;
  n: number;
  splits?: string[];
}

export interface SystemReport {
  split: string;
  n_queries: number;
  metrics: Record<string, Metric>;
}

/** system name -> report */
export type EvalFile = Record<string, SystemReport>;

export interface AblationRow {
  label: string;
  switch: { id: string; key: string; full: string; ablated: string; metric: string } | null;
  skipped: string | null;
  metrics: Record<string, Metric> | null;
  n_queries: number | null;
}

/** system -> capability -> metric */
export type Pooled = Record<string, Record<string, Record<string, Metric>>>;

export interface Report {
  generatedAt: string | null;
  commit: string | null;
  frozen: { code_commit?: string; frozen_before?: string[]; note?: string } | null;
  /** split -> overall per system */
  overall: Record<string, EvalFile>;
  /** split -> system -> capability -> report */
  capabilities: Record<string, Record<string, EvalFile>>;
  pooled: Pooled;
  ablations: AblationRow[];
  evaluated: string[];
  notEvaluated: Record<string, string>;
  supported: string[];
  notShown: string[];
  limits: string[];
  /** M3's extra checks (low-chance windows, conversations, colour); shown only next to the main results */
  extended: Record<string, unknown>;
}

export const SPLITS = ["test", "dev", "judge_sim"] as const;

type Raw = Record<string, unknown>;
export const obj = (v: unknown): Raw => (v && typeof v === "object" && !Array.isArray(v) ? (v as Raw) : {});
const list = (v: unknown): string[] => (Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : []);

/** Normalise whatever /api/report returns; null when nothing has been measured. */
export function normalise(raw: unknown): Report | null {
  const r = obj(raw);
  if (r.splits && Object.keys(obj(r.splits)).length) {
    const splits = obj(r.splits);
    const overall: Record<string, EvalFile> = {};
    const capabilities: Record<string, Record<string, EvalFile>> = {};
    for (const [split, v] of Object.entries(splits)) {
      overall[split] = obj(obj(v).overall) as EvalFile;
      capabilities[split] = obj(obj(v).capabilities) as Record<string, EvalFile>;
    }
    return {
      generatedAt: typeof r.generated_at === "string" ? r.generated_at : null,
      commit: typeof r.code_commit === "string" ? r.code_commit : null,
      frozen: r.frozen ? (obj(r.frozen) as Report["frozen"]) : null,
      overall,
      capabilities,
      pooled: obj(r.pooled) as Pooled,
      ablations: Array.isArray(r.ablations) ? (r.ablations as AblationRow[]) : [],
      evaluated: list(r.capabilities_evaluated),
      notEvaluated: obj(r.not_evaluated) as Record<string, string>,
      supported: list(r.supported_claims),
      notShown: list(r.not_shown),
      limits: list(r.limits),
      extended: obj(r.extended),
    };
  }
  // older shape: {eval: file | {split: file}, ablations: {label: {switch, skipped, report}}}
  if (!r.eval) return null;
  const e = obj(r.eval);
  const first = Object.values(e)[0];
  const overall: Record<string, EvalFile> =
    first && typeof first === "object" && "metrics" in (first as object) ? { [String(obj(first).split ?? "test")]: e as EvalFile } : (e as Record<string, EvalFile>);
  const ablations: AblationRow[] = Object.entries(obj(r.ablations)).map(([label, row]) => {
    const rr = obj(row);
    const rep = obj(rr.report);
    return { label, switch: (rr.switch as AblationRow["switch"]) ?? null, skipped: typeof rr.skipped === "string" ? rr.skipped : null, metrics: (rep.metrics as AblationRow["metrics"]) ?? null, n_queries: typeof rep.n_queries === "number" ? rep.n_queries : null };
  });
  return { generatedAt: null, commit: null, frozen: null, overall, capabilities: {}, pooled: {}, ablations, evaluated: [], notEvaluated: {}, supported: [], notShown: [], limits: [], extended: {} };
}

export const metric = (m: Record<string, Metric> | undefined | null, key: string): Metric | null => m?.[key] ?? null;
export const value = (m: Record<string, Metric> | undefined | null, key: string): number | null => m?.[key]?.value ?? null;

export const fetchReport = async (): Promise<Report | null> => normalise(await api<unknown>("/api/report"));

export const LABELS: Record<string, string> = {
  "hit@1": "Hit@1",
  "hit@5": "Hit@5",
  mrr: "MRR",
  camera_accuracy: "Camera accuracy",
  negative_precision: "Negative precision",
  timestamp_error_s: "Timestamp error",
  temporal_iou: "Temporal IoU",
  existence_accuracy: "Existence accuracy",
  reask_count: "Re-asks",
  ask_precision: "Ask precision",
  ttfa_p50_ms: "First answer p50",
  ttfa_p95_ms: "First answer p95",
  no_model_share: "Answered without a language model",
};

export const CAPABILITY: Record<string, string> = {
  object: "Finding an object class",
  negative: "Saying “nothing there”",
  count: "Counting",
  colour: "Colour",
  carrying: "Carrying",
  path: "Paths across cameras",
};

export const SYSTEM_NAMES: Record<string, string> = { ours: "EVORA", b0: "Frame baseline (B0)", b1: "Detector baseline (B1)" };
export const systemName = (k: string): string => SYSTEM_NAMES[k] ?? k;
export const splitName = (s: string): string => s.replace("_", " ");
