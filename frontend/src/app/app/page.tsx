import type { Metadata } from "next";
import { AppShell } from "@/product/AppShell";
import "./app.css";

export const metadata: Metadata = {
  title: "EVORA — case log",
};

export default function AppPage() {
  return <AppShell />;
}
