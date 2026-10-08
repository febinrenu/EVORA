// Shapes of the evaluation reports M3 writes (eval/reports/*.json) and the
// /api/report payload that carries them. Everything is optional: the page must
// say "not measured" rather than invent a number.
import { api } from "@/lib/api/client";

export interface Metric {
  value: number | null;
  n: number;
}

export interface SystemReport {
  split: string;
  n_queries: number;
  metrics: Record<string, Metric>;
}

/** one eval file: system name -> report */
export type EvalFile = Record<string, SystemReport>;

export interface AblationRow {
  switch: { id: string; key: string; full: string; ablated: string; metric: string } | null;
  skipped: string | null;
  report?: SystemReport;
}

export type AblationFile = Record<string, AblationRow>;

export interface ReportPayload {
  /** split -> eval file, e.g. {test, dev, judge_sim}; older APIs send one file or null */
  eval: Record<string, EvalFile> | EvalFile | null;
  ablations: AblationFile | null;
  generated_at?: number;
}

export const SPLITS = ["test", "dev", "judge_sim"] as const;

/** Normalise both payload forms into split -> eval file. */
export function evalBySplit(p: ReportPayload | null): Record<string, EvalFile> {
  if (!p?.eval) return {};
  const e = p.eval as Record<string, unknown>;
  const first = Object.values(e)[0];
  // a single eval file has systems at the top level, each with `metrics`
  if (first && typeof first === "object" && "metrics" in (first as object)) {
    const file = e as EvalFile;
    const split = Object.values(file)[0]?.split ?? "test";
    return { [split]: file };
  }
  return e as Record<string, EvalFile>;
}

export const metric = (r: SystemReport | undefined, key: string): number | null => r?.metrics[key]?.value ?? null;
export const count = (r: SystemReport | undefined, key: string): number => r?.metrics[key]?.n ?? 0;

export const fetchReport = () => api<ReportPayload>("/api/report");

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

export const SYSTEM_NAMES: Record<string, string> = { ours: "EVORA", b0: "Frame baseline (B0)", b1: "Detector baseline (B1)" };
export const systemName = (k: string): string => SYSTEM_NAMES[k] ?? k;
