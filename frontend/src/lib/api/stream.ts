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
    onMessage({ type: "error", data: { message } });
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

/** Subscribe to /api/events. EventSource reconnects by itself; returns an unsubscribe. */
export function subscribeEvents(onNote: (n: BusNote) => void, onState?: (connected: boolean) => void): () => void {
  const es = new EventSource(apiUrl("/api/events"));
  es.addEventListener("open", () => onState?.(true));
  es.addEventListener("error", () => onState?.(false));
  es.addEventListener("note", (e) => {
    try {
      const n: unknown = JSON.parse((e as MessageEvent<string>).data);
      if (n && typeof n === "object" && "kind" in n) onNote(n as BusNote);
    } catch {
      /* ignore malformed notes */
    }
  });
  return () => es.close();
}
