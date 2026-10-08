"use client";

// Watches and alerts (P4.14). "Watch for…" compiles a standing query; when it
// names a place EVORA does not know yet, the same clarify-once card appears
// and the watch is saved right after. New alerts arrive on the events feed:
// a toast for live ones, quietly in the drawer for ones found in old footage.
import { useEffect, useState } from "react";
import { ApiError, endpoints, type ClarifyRequest, type ClarifyResponse } from "@/lib/api/client";
import { postStream } from "@/lib/api/stream";
import { clock } from "./format";
import { useEvora } from "./store";
import { ClarifyPanel } from "./ClarifyCard";
import { Frame } from "./Frame";

export function WatchDrawer() {
  const open = useEvora((s) => s.drawer);
  const draft = useEvora((s) => s.draft);
  const watches = useEvora((s) => s.watches);
  const alerts = useEvora((s) => s.alerts);
  const setDrawer = useEvora((s) => s.setDrawer);
  const [text, setText] = useState(draft);
  const [prevDraft, setPrevDraft] = useState(draft);
  const [clarify, setClarify] = useState<ClarifyRequest | null>(null);
  const [status, setStatus] = useState<string | null>(null);
  // the server's refusal ("I could not tell what to watch for") shown as is, next to the form
  const [refusal, setRefusal] = useState<string | null>(null);
  if (draft !== prevDraft) {
    setPrevDraft(draft);
    setText(draft);
  }

  useEffect(() => {
    if (!open) return;
    void useEvora.getState().refreshWatches();
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setDrawer(false);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, setDrawer]);

  const create = async (t: string) => {
    setStatus("Compiling the watch…");
    setRefusal(null);
    try {
      await endpoints.watch(t);
      setText("");
      setClarify(null);
      setStatus("Watching. Alerts appear here and as a notice on screen.");
      await useEvora.getState().refreshWatches();
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body && typeof e.body === "object" && "clarify" in e.body) {
        setClarify((e.body as { clarify: ClarifyRequest }).clarify);
        setStatus(null);
      } else {
        setStatus(null);
        setRefusal(e instanceof ApiError ? e.message : "The watch could not be saved: the API did not answer.");
      }
    }
  };

  const answerClarify = async (resp: ClarifyResponse) => {
    setStatus(clarify?.kind === "time_range" ? "Saving the hours…" : "Saving the answer…");
    let failed: string | null = null;
    await postStream("/api/clarify", resp, (m) => {
      if (m.type === "error") failed = typeof m.data.message === "string" ? m.data.message : "That answer did not save.";
    }).catch(() => (failed = "That answer did not reach the API."));
    if (failed) {
      setStatus(null);
      return setRefusal(failed);
    }
    void useEvora.getState().refreshMemory();
    await create(text);
  };

  const watchText = (id: string) => watches.find((w) => w.id === id)?.text ?? "a watch";

  return (
    <aside className={`lt-drawer${open ? " is-open" : ""}`} aria-label="Watches and alerts" aria-hidden={!open} inert={!open}>
      <header>
        <h2>Watch</h2>
        <button type="button" onClick={() => setDrawer(false)}>
          Close
        </button>
      </header>
      <form
        className="lt-watch-form"
        onSubmit={(e) => {
          e.preventDefault();
          if (text.trim()) void create(text.trim());
        }}
      >
        <label htmlFor="watch-input">Watch for</label>
        <input id="watch-input" value={text} onChange={(e) => setText(e.target.value)} placeholder="anyone entering the parking after 8 pm" />
        <button type="submit" disabled={!text.trim()}>
          Watch for this
        </button>
      </form>
      {refusal ? (
        <p className="lt-error" role="alert">
          {refusal}
        </p>
      ) : null}
      {status ? <p className="lt-watch-status">{status}</p> : null}
      {clarify ? <ClarifyPanel req={clarify} onAnswer={(resp) => void answerClarify(resp)} /> : null}

      <h3>Watching</h3>
      {watches.length ? (
        <ul className="lt-watches">
          {watches.map((w) => (
            <li key={w.id} className={w.active ? undefined : "is-paused"}>
              <span>{typeof w.rule.summary === "string" ? w.rule.summary : w.text}</span>
              <button
                type="button"
                aria-pressed={w.active}
                onClick={async () => {
                  await endpoints.setWatch(w.id, !w.active).catch(() => undefined);
                  void useEvora.getState().refreshWatches();
                }}
              >
                {w.active ? "Pause" : "Resume"}
              </button>
            </li>
          ))}
        </ul>
      ) : (
        <p className="lt-quiet">Nothing yet. A watch keeps checking new footage and replays of old footage.</p>
      )}

      <h3>Alerts</h3>
      {alerts.length ? (
        <ul className="lt-alerts">
          {alerts.map((a) => (
            <li key={a.id} className={a.acknowledged ? "is-acked" : undefined}>
              <button type="button" className="lt-alert-open" onClick={() => useEvora.getState().openAlert(a, watchText(a.standing_query_id))}>
                <Frame src={a.evidence.thumb_url} alt={`${a.evidence.camera_name} at ${clock(a.t)}`} />
                <span>
                  <b>{a.evidence.camera_name}</b> {clock(a.t)}
                  <em>{watchText(a.standing_query_id)}</em>
                </span>
              </button>
              {!a.acknowledged ? (
                <button type="button" onClick={() => void useEvora.getState().ackAlert(a.id)}>
                  Acknowledge
                </button>
              ) : (
                <span className="lt-quiet">Seen</span>
              )}
            </li>
          ))}
        </ul>
      ) : (
        <p className="lt-quiet">No alerts.</p>
      )}
    </aside>
  );
}

/** Live alerts as small notices; they leave on their own after a while. */
export function Toasts() {
  const toasts = useEvora((s) => s.toasts);
  const watches = useEvora((s) => s.watches);
  useEffect(() => {
    if (!toasts.length) return;
    const t = window.setTimeout(() => useEvora.getState().dismissToast(toasts[0].id), 9000);
    return () => window.clearTimeout(t);
  }, [toasts]);
  if (!toasts.length) return null;
  return (
    <ol className="lt-toasts" aria-live="assertive">
      {toasts.map(({ id, alert }) => {
        const text = watches.find((w) => w.id === alert.standing_query_id)?.text ?? "Watch";
        return (
          <li key={id}>
            <p>
              <b>{alert.evidence.camera_name}</b> {clock(alert.t)} · {text}
            </p>
            <div>
              <button type="button" onClick={() => useEvora.getState().openAlert(alert, text)}>
                Open
              </button>
              <button type="button" onClick={() => void useEvora.getState().ackAlert(id)}>
                Acknowledge
              </button>
            </div>
          </li>
        );
      })}
    </ol>
  );
}
