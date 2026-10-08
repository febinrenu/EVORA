import type { Metadata } from "next";
import { readFile } from "node:fs/promises";
import { join } from "node:path";
import { ReportView } from "@/product/report/ReportView";
import { normalise, type Report } from "@/product/report/data";
import "./report.css";

export const metadata: Metadata = {
  title: "EVORA — what the measurements show",
};

/**
 * The latest report at build time, so the page paints its numbers without
 * waiting for a request; the page then refreshes from /api/report.
 */
async function reportAtBuild(): Promise<{ report: Report | null; at: string | null }> {
  try {
    const raw: unknown = JSON.parse(await readFile(join(process.cwd(), "..", "eval", "reports", "report.json"), "utf8"));
    const report = normalise(raw);
    return { report, at: report?.generatedAt ?? null };
  } catch {
    return { report: null, at: null };
  }
}

export default async function ReportPage() {
  const { report } = await reportAtBuild();
  return <ReportView initial={report} />;
}
