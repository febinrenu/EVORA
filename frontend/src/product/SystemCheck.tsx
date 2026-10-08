"use client";

// "Check this machine": the quick checks of `make doctor`, run by the API
// against itself. Shows what passes, what is missing and the command that
// fixes it; the UI never runs those commands.
import { useEffect, useRef, useState } from "react";
import { ApiError, endpoints, type Doctor } from "@/lib/api/client";

const STATUS: Record<string, string> = { ok: "Ready", pass: "Ready", warn: "Worth fixing", fail: "Blocking", skip: "Skipped" };

export function SystemCheck({ onClose }: { onClose: () => void }) {
  const [doc, setDoc] = useState<Doctor | null>(null);
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    ref.current?.focus();
    endpoints
      .doctor()
      .then(setDoc)
      .catch((e: unknown) => setError(e instanceof ApiError ? (e.status === 404 ? "This API version cannot check itself yet. Run make doctor in a terminal." : e.message) : "The API did not answer."));
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  const order = (s: string) => (s === "fail" ? 0 : s === "warn" ? 1 : 2);
  const checks = [...(doc?.checks ?? [])].sort((a, b) => order(a.status.toLowerCase()) - order(b.status.toLowerCase()));

  return (
    <div className="lt-keys" role="dialog" aria-modal="true" aria-label="Check this machine" tabIndex={-1} ref={ref} onClick={onClose}>
      <div className="lt-keys-sheet lt-doctor" onClick={(e) => e.stopPropagation()}>
        <h2>Check this machine</h2>
        {error ? <p className="lt-error">{error}</p> : null}
        {!doc && !error ? <p className="lt-quiet">Checking…</p> : null}
        {doc ? (
          <>
            <p className={`lt-doctor-verdict${doc.ok ? " is-ok" : ""}`}>
              {doc.verdict} <span className="lt-quiet">({doc.took_s.toFixed(1)} s)</span>
            </p>
            <ul>
              {checks.map((c) => {
                const st = c.status.toLowerCase();
                return (
                  <li key={c.id} className={`is-${st}`}>
                    <span className="lt-doctor-status">{STATUS[st] ?? c.status}</span>
                    <span>
                      <b>{c.title}</b> {c.detail}
                      {c.fix && st !== "ok" && st !== "pass" ? <code>{c.fix}</code> : null}
                    </span>
                  </li>
                );
              })}
            </ul>
          </>
        ) : null}
        <button type="button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
