"use client";

// Sessions (workspaces): the open one is named in the top bar; this menu opens another or starts a new, empty one.
// A session keeps its cameras, index, learned places and rules on the server, so a reload never loses work; a new
// session is how you start from nothing. Switching restarts the server on the other folder, so we wait for it.
import { useEffect, useRef, useState } from "react";
import { api } from "@/lib/api/client";

type Session = { slug: string; name: string; active: boolean };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

export function SessionMenu({ current }: { current: string }) {
  const [open, setOpen] = useState(false);
  const [sessions, setSessions] = useState<Session[]>([]);
  const [opening, setOpening] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    api<Session[]>("/api/workspaces").then(setSessions).catch(() => setSessions([]));
    const away = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", away);
    return () => document.removeEventListener("mousedown", away);
  }, [open]);

  const waitFor = async (slug: string) => {
    for (let i = 0; i < 120; i++) {
      await sleep(1000);
      try {
        const h = await api<{ workspace: string }>("/api/health");
        if (h.workspace === slug) return true;
      } catch {
        /* the server is restarting */
      }
    }
    return false;
  };

  const go = async (label: string, request: () => Promise<{ slug?: string; switching_to?: string }>) => {
    setError(null);
    setOpen(false);
    setOpening(label);
    try {
      const res = await request();
      const slug = res.slug ?? res.switching_to;
      if (slug && (await waitFor(slug))) {
        window.location.reload();
        return;
      }
      setError("The session did not open. Check the window that runs start.bat.");
    } catch (e) {
      setError(e instanceof Error ? e.message : "The session could not be opened.");
    }
    setOpening(null);
  };

  const fresh = () => go("a new session", () => api("/api/workspaces", { method: "POST", body: JSON.stringify({ activate: true }) }));
  const openOne = (s: Session) => go(`“${s.name}”`, () => api(`/api/workspaces/${encodeURIComponent(s.slug)}/activate`, { method: "POST" }));

  return (
    <div className="lt-session" ref={box}>
      <button type="button" className="lt-site" aria-haspopup="listbox" aria-expanded={open} onClick={() => setOpen((o) => !o)} title="Sessions">
        {current || "No workspace"} <span aria-hidden="true">▾</span>
      </button>
      {open ? (
        <div className="lt-session-pop" role="listbox" aria-label="Sessions">
          <button type="button" className="lt-session-new" onClick={() => void fresh()}>
            New session
          </button>
          <p className="lt-session-hint">Starts empty. The current session is kept and can be opened again.</p>
          <ul>
            {sessions.map((s) => (
              <li key={s.slug}>
                <button type="button" role="option" aria-selected={s.active} disabled={s.active} onClick={() => void openOne(s)}>
                  {s.name}
                  {s.active ? <span className="lt-session-open"> open</span> : null}
                </button>
              </li>
            ))}
          </ul>
        </div>
      ) : null}
      {error ? <span className="lt-warn">{error}</span> : null}
      {opening ? (
        <div className="lt-session-wait" role="status" aria-live="polite">
          Opening {opening}…
        </div>
      ) : null}
    </div>
  );
}
