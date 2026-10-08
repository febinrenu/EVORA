"use client";

// Case log (P4.5): each question and its answer sheet, newest at the bottom
// next to the ask bar. The answer renders as soon as it arrives and updates in
// place when verification lands.
import { useEffect, useRef } from "react";
import { useEvora } from "./store";
import { AnswerSheet } from "./AnswerSheet";

const EXAMPLES = ["Did anyone carry a large bag through the lobby?", "Did a red car pass through the main gate in the last hour?", "Where did the person in the blue jacket go?"];

export function CaseLog() {
  const cases = useEvora((s) => s.cases);
  const ask = useEvora((s) => s.ask);
  const end = useRef<HTMLDivElement>(null);
  // follow the newest entry as it grows (planning -> clarify or answer)
  const last = cases[cases.length - 1];
  const key = last ? `${cases.length}:${last.status}:${last.evidence.length}` : "";

  useEffect(() => {
    end.current?.scrollIntoView({ block: "end", behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth" });
  }, [key]);

  if (!cases.length) {
    return (
      <div className="lt-empty">
        <p>Ask about the footage. Try: did anyone carry a large bag through the lobby?</p>
        <ul>
          {EXAMPLES.map((q) => (
            <li key={q}>
              <button type="button" onClick={() => ask(q)}>
                {q}
              </button>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  return (
    <div className="lt-log" role="log" aria-live="polite">
      {cases.map((c) => (
        <article key={c.id} className="lt-entry">
          <p className="lt-question">{c.question}</p>
          <AnswerSheet c={c} />
        </article>
      ))}
      <div ref={end} />
    </div>
  );
}
