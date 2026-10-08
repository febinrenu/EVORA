"use client";

// Evidence sheet (P4.6): the matched frame enlarged with the circle drawn
// around the object, a film strip of the seconds around it, the caption with
// both timestamps, and the reasons. "Play clip" swaps the frame for the clip
// with the box tracked over it.
import { useEffect, useState } from "react";
import { ApiError, apiUrl, endpoints, frameUrl, withUnblur, type Evidence } from "@/lib/api/client";
import { clock, confidence, fileName, offset, span } from "./format";
import { useEvora } from "./store";
import { Frame } from "./Frame";
import { ClipPlayer } from "./ClipPlayer";

const STRIP = [-2, -1, 0, 1, 2];

/** path steps carry their place on the route instead of a match score */
const isHop = (h: unknown): h is { index: number; of: number } => typeof h === "object" && h !== null && "index" in h && "of" in h;

export function EvidenceSheet({ ev, verified, first }: { ev: Evidence; verified: boolean | null | undefined; first: boolean }) {
  const cam = useEvora((s) => s.cameras.find((c) => c.id === ev.camera_id));
  const [playing, setPlaying] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);
  // faces are blurred everywhere unless someone states a reason; the API logs it
  const [faces, setFaces] = useState<{ token: string; until: number; reason: string } | null>(null);
  const [asking, setAsking] = useState(false);
  const [reason, setReason] = useState("");
  const [faceError, setFaceError] = useState<string | null>(null);
  const token = faces?.token ?? null;
  // the token lapses after five minutes: blur again on time, without a reload
  useEffect(() => {
    if (!faces) return;
    const t = window.setTimeout(() => setFaces(null), Math.max(0, faces.until * 1000 - Date.now()));
    return () => window.clearTimeout(t);
  }, [faces]);

  const showFaces = async () => {
    setFaceError(null);
    try {
      const r = await endpoints.unblur(reason.trim(), ev.id);
      setFaces({ token: r.token, until: r.expires_at, reason: reason.trim() });
      setAsking(false);
      setReason("");
    } catch (e) {
      setFaceError(e instanceof ApiError ? e.message : "The API did not answer.");
    }
  };

  const exportPack = async () => {
    setExporting("Preparing the evidence pack…");
    try {
      const res = await fetch(apiUrl(`/api/evidence/${encodeURIComponent(ev.id)}/pack`), { method: "POST" });
      if (!res.ok) {
        const body: unknown = await res.json().catch(() => null);
        const detail = body && typeof body === "object" && "detail" in body ? String((body as { detail: unknown }).detail) : `${res.status}`;
        setExporting(`Export did not run: ${detail}`);
        return;
      }
      const sha = res.headers.get("X-Evora-Pack-SHA256");
      const url = URL.createObjectURL(await res.blob());
      const a = document.createElement("a");
      a.href = url;
      a.download = `evora_evidence_${ev.id}.zip`;
      a.click();
      URL.revokeObjectURL(url);
      const signer = await endpoints.signer().catch(() => null);
      setExporting(
        [sha ? `Exported. SHA-256 ${sha.slice(0, 16)}…` : "Exported.", signer ? `Signed with key ${signer.fingerprint} (${signer.algorithm}).` : null].filter(Boolean).join(" "),
      );
    } catch {
      setExporting("Export failed: the API did not answer.");
    }
  };

  const osd = `${ev.camera_name.toUpperCase()} ${clock(ev.t_peak, true)}`;
  return (
    <section className="lt-evidence" aria-label={`Evidence from ${ev.camera_name}`}>
      {playing ? (
        <ClipPlayer ev={ev} token={token} onClose={() => setPlaying(false)} />
      ) : (
        <Frame src={withUnblur(ev.thumb_url, token)} alt={`${ev.camera_name} at ${clock(ev.t_peak)}: the matched object, circled`} bbox={ev.bbox} markId={ev.id} animate={first} osd={osd} className="is-hero" />
      )}
      <ol className="lt-film" aria-label="Seconds around the match">
        {STRIP.map((d) => (
          <li key={d} className={d === 0 ? "is-peak" : undefined}>
            <Frame src={frameUrl(ev.camera_id, ev.t_peak + d, token ?? undefined)} alt={`${ev.camera_name} ${d === 0 ? "at the match" : `${d > 0 ? "+" : ""}${d} s`}`} />
          </li>
        ))}
      </ol>
      <div className="lt-caption">
        <p className="lt-where">
          <b>{ev.camera_name}</b>, {span(ev)}
        </p>
        <p className="lt-offset">
          {offset(ev.offset_s)} into {fileName(cam)}
        </p>
        <p className={`lt-score is-${verified === true ? "ok" : verified === false ? "no" : "open"}`}>
          {isHop(ev.hop) ? `Step ${ev.hop.index} of ${ev.hop.of} on the route` : confidence(ev.score)}
          {verified === true ? " · confirmed" : verified === false ? " · rejected on a second look" : ""}
        </p>
        {ev.why?.length ? (
          <details className="lt-why">
            <summary>Why</summary>
            <ul>
              {ev.why.map((w, i) => (
                <li key={i}>{w}</li>
              ))}
            </ul>
          </details>
        ) : null}
        <div className="lt-actions">
          <button type="button" onClick={() => setPlaying((p) => !p)}>
            {playing ? "Show the frame" : "Play clip"}
          </button>
          <button type="button" onClick={() => void exportPack()}>
            Export evidence
          </button>
          {ev.track_id ? (
            <button type="button" onClick={() => useEvora.getState().findSimilar(ev)}>
              Find this elsewhere
            </button>
          ) : null}
          {ev.global_id ? (
            <button type="button" onClick={() => useEvora.getState().showPath(ev)}>
              Show the path
            </button>
          ) : null}
          {token ? (
            <button type="button" onClick={() => setFaces(null)}>
              Blur faces again
            </button>
          ) : (
            <button type="button" onClick={() => setAsking((a) => !a)} aria-expanded={asking}>
              Show faces
            </button>
          )}
        </div>
        {asking && !token ? (
          <form
            className="lt-unblur"
            onSubmit={(e) => {
              e.preventDefault();
              if (reason.trim()) void showFaces();
            }}
          >
            <label>
              <span>Why do you need to see faces? This is logged.</span>
              <input value={reason} maxLength={200} onChange={(e) => setReason(e.target.value)} placeholder="identify the driver for the incident report" autoFocus />
            </label>
            <button type="submit" disabled={!reason.trim()}>
              Show for 5 minutes
            </button>
            {faceError ? <span className="lt-error">{faceError}</span> : null}
          </form>
        ) : null}
        {token && faces ? (
          <p className="lt-unblurred" role="status">
            Faces shown until {clock(faces.until).slice(0, 5)}. Reason logged: “{faces.reason}”.
          </p>
        ) : null}
        {exporting ? <p className="lt-export" aria-live="polite">{exporting}</p> : null}
      </div>
    </section>
  );
}
