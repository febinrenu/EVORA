import type { Metadata } from "next";
import { ReportView } from "@/product/report/ReportView";
import "./report.css";

export const metadata: Metadata = {
  title: "EVORA — what the measurements show",
};

export default function ReportPage() {
  return <ReportView />;
}
