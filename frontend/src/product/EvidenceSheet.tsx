"use client";

// Evidence sheet (P4.6): the matched frame enlarged with the circle drawn
// around the object, a film strip of the seconds around it, the caption with
// both timestamps, and the reasons. "Play clip" swaps the frame for the clip
// with the box tracked over it.
import { useState } from "react";
import { apiUrl, frameUrl, type Evidence } from "@/lib/api/client";
import { clock, confidence, fileName, offset, span } from "./format";
import { useEvora } from "./store";
import { Frame } from "./Frame";
import { ClipPlayer } from "./ClipPlayer";

const STRIP = [-2, -1, 0, 1, 2];

export function EvidenceSheet({ ev, verified, first }: { ev: Evidence; verified: boolean | null | undefined; first: boolean }) {
  const cam = useEvora((s) => s.cameras.find((c) => c.id === ev.camera_id));
  const [playing, setPlaying] = useState(false);
  const [exporting, setExporting] = useState<string | null>(null);

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
      setExporting(sha ? `Exported. SHA-256 ${sha.slice(0, 16)}…` : "Exported.");
    } catch {
      setExporting("Export failed: the API did not answer.");
    }
  };

  const osd = `${ev.camera_name.toUpperCase()} ${clock(ev.t_peak, true)}`;
  return (
    <section className="lt-evidence" aria-label={`Evidence from ${ev.camera_name}`}>
      {playing ? (
        <ClipPlayer ev={ev} onClose={() => setPlaying(false)} />
      ) : (
        <Frame src={ev.thumb_url} alt={`${ev.camera_name} at ${clock(ev.t_peak)}: the matched object, circled`} bbox={ev.bbox} markId={ev.id} animate={first} osd={osd} className="is-hero" />
      )}
      <ol className="lt-film" aria-label="Seconds around the match">
        {STRIP.map((d) => (
          <li key={d} className={d === 0 ? "is-peak" : undefined}>
            <Frame src={frameUrl(ev.camera_id, ev.t_peak + d)} alt={`${ev.camera_name} ${d === 0 ? "at the match" : `${d > 0 ? "+" : ""}${d} s`}`} />
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
          {confidence(ev.score)}
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
        </div>
        {exporting ? <p className="lt-export" aria-live="polite">{exporting}</p> : null}
      </div>
    </section>
  );
}
