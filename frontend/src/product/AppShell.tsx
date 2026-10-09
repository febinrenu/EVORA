"use client";

// The operator product on the light table (PLAN §10.4): cameras on the left,
// the case log with the ask bar in the centre, site plan and known places on
// the right, timeline lanes across the bottom.
import { useEffect, useState } from "react";
import { subscribeEvents } from "@/lib/api/stream";
import { endpoints, type Alert, type IngestJob } from "@/lib/api/client";
import { useEvora } from "./store";
import { TopBar } from "./TopBar";
import { CameraRail } from "./CameraRail";
import { CaseLog } from "./CaseLog";
import { AskBar } from "./AskBar";
import { SidePanel } from "./SidePanel";
import { Timeline } from "./Timeline";
import { Shortcuts } from "./Shortcuts";
import { SystemCheck } from "./SystemCheck";
import { Toasts, WatchDrawer } from "./WatchDrawer";

export function AppShell() {
  const [help, setHelp] = useState(false);
  const [check, setCheck] = useState(false);
  // Times are formatted in the site's zone, which arrives with /api/health. Reading it
  // here re-renders the whole table when it lands (or changes), so nothing drawn
  // before it keeps the browser's zone next to times drawn after it.
  useEvora((s) => s.health?.tz);

  useEffect(() => {
    // `?debug` exposes the store for UI tests that must not touch real workspaces
    if (new URLSearchParams(window.location.search).has("debug")) Object.assign(window, { __evoraStore: useEvora });
    // everything the event stream would have said: on load, and again after every reconnect
    const catchUp = () => {
      const s = useEvora.getState();
      void s.refreshHealth();
      void s.refreshCameras();
      void s.refreshMemory();
      void s.refreshWatches();
      endpoints
        .live()
        .then((l) => {
          l.streams.forEach((st) => useEvora.getState().setLive(st.camera_id, st.state));
          l.analyzers?.forEach((a) => useEvora.getState().setAnalysis(a.camera_id, a.state));
        })
        .catch(() => undefined);
    };
    catchUp();
    void useEvora.getState().restoreEarlier();
    const health = window.setInterval(() => void useEvora.getState().refreshHealth(), 15000);
    const off = subscribeEvents(
      (n) => {
        const st = useEvora.getState();
        if (n.kind === "ingest" && n.job && typeof n.job === "object") st.setJob(n.job as IngestJob);
        else if (n.kind === "camera" && typeof n.camera_id === "string") {
          st.setCameraStatus(n.camera_id, n.status as never);
          // a clock correction moved the camera's stored times: fetch its new t0
          if (n.status === "ready" || typeof n.clock === "string") void st.refreshCameras();
          if (typeof n.clock === "string") st.clockCorrected(n.camera_id);
        } else if (n.kind === "clock" && typeof n.camera_id === "string") {
          if (n.state === "reading") st.setClock(n.camera_id, { state: "reading" });
          else {
            st.setClock(n.camera_id, n.state === "failed" ? { state: "failed", error: typeof n.error === "string" ? n.error : undefined } : null);
            void st.refreshCameras();
          }
        } else if (n.kind === "site") st.siteChanged();
        else if (n.kind === "privacy") void st.refreshHealth();
        else if (n.kind === "live" && typeof n.camera_id === "string" && typeof n.state === "string") st.setLive(n.camera_id, n.state);
        else if (n.kind === "analysis" && typeof n.camera_id === "string" && typeof n.state === "string") st.setAnalysis(n.camera_id, n.state);
        else if (n.kind === "alert" && n.alert && typeof n.alert === "object") st.pushAlert(n.alert as Alert, n.historical === true);
      },
      (connected) => useEvora.getState().setConnected(connected),
      catchUp,
    );
    return () => {
      window.clearInterval(health);
      off();
    };
  }, []);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const typing = e.target instanceof HTMLElement && (e.target.closest("input, textarea, [contenteditable]") !== null);
      if (typing) {
        if (e.key === "Escape") (e.target as HTMLElement).blur();
        return;
      }
      if (e.key === "/") {
        e.preventDefault();
        document.getElementById("ask-input")?.focus();
      } else if (e.key === "?") setHelp((h) => !h);
      else if (e.key === "Escape") setHelp(false);
      else if (e.key === "[" || e.key === "]") {
        const { cases, focus, setFocus } = useEvora.getState();
        const c = cases.find((x) => x.id === focus?.caseId) ?? cases[cases.length - 1];
        if (!c || !c.evidence.length) return;
        const i = Math.max(0, c.evidence.findIndex((ev) => ev.id === focus?.evidenceId));
        const j = Math.min(c.evidence.length - 1, Math.max(0, i + (e.key === "]" ? 1 : -1)));
        setFocus({ caseId: c.id, evidenceId: c.evidence[j].id });
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  return (
    <div className="lt">
      <TopBar onHelp={() => setHelp(true)} onCheck={() => setCheck(true)} />
      <CameraRail />
      <main className="lt-case" aria-label="Case log">
        <CaseLog />
        <AskBar />
      </main>
      <SidePanel />
      <Timeline />
      <WatchDrawer />
      <Toasts />
      {help ? <Shortcuts onClose={() => setHelp(false)} /> : null}
      {check ? <SystemCheck onClose={() => setCheck(false)} /> : null}
    </div>
  );
}
