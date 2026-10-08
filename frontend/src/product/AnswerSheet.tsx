"use client";

import type { Evidence } from "@/lib/api/client";
import { clock, confidence, seconds, verdictLine, watchSentence } from "./format";
import { useEvora, type Case } from "./store";
import { EvidenceSheet } from "./EvidenceSheet";
import { ClarifyCard } from "./ClarifyCard";
import { Frame } from "./Frame";

/** One answer: the verdict, the sentence, the proof, then how it was reached. */
export function AnswerSheet({ c }: { c: Case }) {
  const focus = useEvora((s) => s.focus);
  const setFocus = useEvora((s) => s.setFocus);
  const cameras = useEvora((s) => s.cameras);

  if (c.status === "clarify" && c.clarify) return <ClarifyCard c={c} req={c.clarify} />;

  if (c.status === "planning" || (c.status === "searching" && !c.answer)) {
    return (
      <div className="lt-sheet is-working" aria-busy="true">
        <p className="lt-working">{c.status === "planning" ? "Reading the question…" : c.evidence.length ? `Checking ${c.evidence.length} candidate${c.evidence.length > 1 ? "s" : ""}…` : "Searching the footage…"}</p>
        {c.resolved.length ? <p className="lt-learned">Learned {c.resolved.join(", ")}. This will not be asked again.</p> : null}
        {c.evidence.length ? <Strip evidence={c.evidence} c={c} active={null} onPick={() => undefined} /> : null}
      </div>
    );
  }

  if (c.status === "error" && !c.answer) {
    return (
      <div className="lt-sheet is-error" role="alert">
        <p>{c.error ?? "The answer could not be completed."}</p>
      </div>
    );
  }

  const a = c.answer;
  if (!a && c.earlier) {
    return (
      <div className="lt-sheet">
        <p className="lt-answer">
          {c.summary?.verdict ? `${verdictLine(c.summary.verdict, null, c.summary.results ?? 0)} ` : ""}
          {c.summary?.results !== null && c.summary?.results !== undefined ? `${c.summary.results} result${c.summary.results === 1 ? "" : "s"}. ` : ""}
          The evidence was not kept with this entry.
        </p>
        <div className="lt-sheet-actions">
          <button type="button" onClick={() => useEvora.getState().ask(c.question)}>
            Ask again
          </button>
        </div>
      </div>
    );
  }
  if (!a) return null;
  const evidence = c.evidence;
  const activeId = focus?.caseId === c.id ? focus.evidenceId : evidence[0]?.id;
  const active = evidence.find((e) => e.id === activeId) ?? evidence[0];
  const verifiedCount = Object.values(c.verified).filter((v) => v === true).length;
  const watch = watchSentence(a.plan ?? c.plan, cameras);
  const notes = [...(a.notes ?? []), ...c.notes];
  const partial = a.verdict === "partial";

  return (
    <div className="lt-sheet">
      <p className={`lt-verdict${partial ? " is-partial" : ""}`}>{verdictLine(a.verdict, a.count, evidence.length)}</p>
      {partial ? <p className="lt-partial">Some of this could not be checked. The notes below say what.</p> : null}
      {a.verdict !== "count" && evidence.length > 1 ? <p className="lt-shown">{evidence.length} shown</p> : null}
      <p className="lt-answer">{a.text}</p>
      {c.resolved.length ? <p className="lt-learned">Learned {c.resolved.join(", ")}. This will not be asked again.</p> : null}
      {active ? <EvidenceSheet key={active.id} ev={active} verified={c.verified[active.id]} first={active.id === evidence[0]?.id} /> : null}
      {evidence.length > 1 ? <Strip evidence={evidence} c={c} active={active?.id ?? null} onPick={(id) => setFocus({ caseId: c.id, evidenceId: id })} /> : null}
      {a.path?.length ? (
        <ol className="lt-path" aria-label="Path across cameras">
          {a.path.map((h, i) => (
            <li key={`${h.camera_id}-${i}`}>
              <b>{h.camera_name}</b> {clock(h.t_in)}
            </li>
          ))}
        </ol>
      ) : null}
      {!evidence.length && a.nearest_miss ? (
        <div className="lt-miss">
          <p>Closest: {a.nearest_miss.camera_name} at {clock(a.nearest_miss.t_peak)}, {confidence(a.nearest_miss.score).toLowerCase()}.</p>
          <Frame src={a.nearest_miss.thumb_url} alt={`Closest candidate on ${a.nearest_miss.camera_name}`} bbox={a.nearest_miss.bbox} markId={undefined} />
        </div>
      ) : null}
      {notes.length ? (
        <ul className={`lt-notes${partial ? " is-partial" : ""}`} aria-label="Notes on this answer">
          {notes.map((n, i) => (
            <li key={i}>{n}</li>
          ))}
        </ul>
      ) : null}
      {watch ? (
        <div className="lt-sheet-actions">
          <button type="button" onClick={() => useEvora.getState().setDrawer(true, watch)}>
            Watch for this
          </button>
        </div>
      ) : null}
      <p className="lt-meta">
        {c.ttfa !== undefined ? `First answer in ${seconds(c.ttfa)}` : null}
        {verifiedCount ? ` · ${verifiedCount} confirmed by a second look` : null}
        {a.plan?.source ? ` · planned by ${a.plan.source === "fastpath" ? "the fast path" : a.plan.source === "cache" ? "the plan cache" : a.plan.source === "local_llm" ? "the local model" : "the cloud planner"}` : null}
      </p>
    </div>
  );
}

function Strip({ evidence, c, active, onPick }: { evidence: Evidence[]; c: Case; active: string | null; onPick: (id: string) => void }) {
  return (
    <ol className="lt-strip" aria-label="Other evidence">
      {evidence.map((ev) => (
        <li key={ev.id}>
          <button type="button" className={ev.id === active ? "is-active" : undefined} onClick={() => onPick(ev.id)} aria-pressed={ev.id === active}>
            <Frame src={ev.thumb_url} alt={`${ev.camera_name} at ${clock(ev.t_peak)}`} />
            <span>
              {ev.camera_name} {clock(ev.t_peak)}
              {c.verified[ev.id] === true ? " · confirmed" : c.verified[ev.id] === false ? " · rejected" : ""}
            </span>
          </button>
        </li>
      ))}
    </ol>
  );
}
