// Server-sent events over POST (the query and clarify routes), parsed from a
// fetch body stream, plus the long-lived GET /api/events feed.
import { apiUrl } from "./client";

export type StreamType = "plan" | "clarify" | "evidence" | "answer" | "verified" | "note" | "error" | "done";

export interface StreamMessage {
  type: StreamType;
  data: Record<string, unknown>;
}

const TYPES = new Set<StreamType>(["plan", "clarify", "evidence", "answer", "verified", "note", "error", "done"]);

function parseBlock(block: string): StreamMessage | null {
  let event = "message";
  const data: string[] = [];
  for (const raw of block.split("\n")) {
    const line = raw.replace(/\r$/, "");
    if (!line || line.startsWith(":")) continue;
    const i = line.indexOf(":");
    const field = i < 0 ? line : line.slice(0, i);
    const value = i < 0 ? "" : line.slice(i + 1).replace(/^ /, "");
    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }
  if (!TYPES.has(event as StreamType)) return null;
  try {
    const parsed: unknown = JSON.parse(data.join("\n") || "{}");
    return { type: event as StreamType, data: parsed && typeof parsed === "object" ? (parsed as Record<string, unknown>) : {} };
  } catch {
    return null;
  }
}

/** POST a JSON body and deliver each SSE message in order. Resolves when the stream ends. */
export async function postStream(path: string, body: unknown, onMessage: (m: StreamMessage) => void, signal?: AbortSignal): Promise<void> {
  const res = await fetch(apiUrl(path), {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let message = `${res.status} ${res.statusText}`;
    try {
      const j: unknown = await res.json();
      if (j && typeof j === "object" && "detail" in j) message = String((j as { detail: unknown }).detail);
    } catch {
      /* keep the status line */
    }
    onMessage({ type: "error", data: { message, status: res.status } });
    onMessage({ type: "done", data: {} });
    return;
  }
  const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value.replace(/\r\n/g, "\n");
    let cut: number;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const msg = parseBlock(buffer.slice(0, cut));
      buffer = buffer.slice(cut + 2);
      if (msg) onMessage(msg);
    }
  }
  const tail = parseBlock(buffer);
  if (tail) onMessage(tail);
}

export interface BusNote {
  kind: string;
  [k: string]: unknown;
}

/**
 * Subscribe to /api/events. EventSource reconnects by itself after a dropped
 * connection, but gives up for good when a restarting server answers with an
 * HTTP error; then it is opened again with a backoff. `onReopen` runs on every
 * reconnect, so the caller can fetch what it missed. Returns an unsubscribe.
 */
export function subscribeEvents(onNote: (n: BusNote) => void, onState?: (connected: boolean) => void, onReopen?: () => void): () => void {
  const BACKOFF_MS = [1000, 2000, 5000, 10000];
  let es: EventSource | null = null;
  let stopped = false;
  let opened = false;
  let tries = 0;
  let timer = 0;
  const open = () => {
    const source = new EventSource(apiUrl("/api/events"));
    es = source;
    source.addEventListener("open", () => {
      if (opened) onReopen?.();
      opened = true;
      tries = 0;
      onState?.(true);
    });
    source.addEventListener("error", () => {
      onState?.(false);
      if (source.readyState === EventSource.CLOSED && !stopped) {
        source.close();
        timer = window.setTimeout(open, BACKOFF_MS[Math.min(tries++, BACKOFF_MS.length - 1)]);
      }
    });
    source.addEventListener("note", (e) => {
      try {
        const n: unknown = JSON.parse((e as MessageEvent<string>).data);
        if (n && typeof n === "object" && "kind" in n) onNote(n as BusNote);
      } catch {
        /* ignore malformed notes */
      }
    });
  };
  open();
  return () => {
    stopped = true;
    window.clearTimeout(timer);
    es?.close();
  };
}
