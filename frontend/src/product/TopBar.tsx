"use client";

import { useState } from "react";
import { endpoints } from "@/lib/api/client";
import { clock } from "./format";
import { SessionMenu } from "./SessionMenu";
import { useEvora } from "./store";

/** Workspace, the footage clock (end of the latest recording) and the privacy state. */
export function TopBar({ onHelp, onCheck }: { onHelp: () => void; onCheck: () => void }) {
  const health = useEvora((s) => s.health);
  const connected = useEvora((s) => s.connected);
  const cameras = useEvora((s) => s.cameras);
  const unseen = useEvora((s) => s.alerts.filter((a) => !a.acknowledged).length);
  const setDrawer = useEvora((s) => s.setDrawer);
  const [busy, setBusy] = useState(false);
  const end = cameras.reduce((m, c) => Math.max(m, c.t0 + (c.duration_s ?? 0)), 0);

  const toggle = async () => {
    if (!health) return;
    setBusy(true);
    try {
      await endpoints.setOnprem(!health.onprem);
      await useEvora.getState().refreshHealth();
    } finally {
      setBusy(false);
    }
  };

  return (
    <header className="lt-top">
      {/* full page load into the story: it boots its own renderer and scroll */}
      {/* eslint-disable-next-line @next/next/no-html-link-for-pages */}
      <a href="/" className="lt-mark" title="Back to the story">
        EVORA
      </a>
      <SessionMenu current={health?.workspace ?? ""} />
      <span className="lt-clock">{end ? `Footage clock ${clock(end)}` : "No footage yet"}</span>
      <span className="lt-spacer" />
      {!connected ? <span className="lt-offline">Reconnecting to this machine…</span> : null}
      {health?.blur === "unavailable" ? <span className="lt-warn">Face blur is unavailable</span> : null}
      <a href="/report/" className="lt-results">
        Results
      </a>
      <button type="button" className="lt-watch-btn" onClick={() => setDrawer(true)}>
        Watch{unseen ? <span className="lt-badge" aria-label={`${unseen} new alerts`}>{unseen}</span> : null}
      </button>
      <button type="button" className={`lt-privacy${health?.onprem ? " is-on" : ""}`} onClick={() => void toggle()} disabled={!health || busy} aria-pressed={health?.onprem ?? false}>
        {health?.onprem ? "On this machine only" : "Cloud planner allowed"}
      </button>
      <button type="button" className="lt-check" onClick={onCheck}>
        Check this machine
      </button>
      <button type="button" className="lt-help" onClick={onHelp} aria-label="Keyboard shortcuts">
        ?
      </button>
    </header>
  );
}
