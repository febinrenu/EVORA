import { Split } from "@/components/ui/Split";
import { FUNNEL, MATCH, PARSE_SLOTS, PLAN, QUERY, REASONS } from "@/lib/data/story";
import { formatInt } from "@/lib/math";

const DIGITS = 7;

/** The plan as the planner returns it, one key per line so it reads at a glance. */
function PlanJson() {
  const keys = ["intent", "targets", "place", "action", "time", "unresolved", "source"] as const;
  const json = [
    "{",
    ...keys.map((k, i) => {
      const v = JSON.stringify(PLAN[k], (key, val: unknown) => (key === "embed_text" ? undefined : val))
        .replace(/,"/g, ', "')
        .replace(/":/g, '": ');
      return `  "${k}": ${v}${i < keys.length - 1 ? "," : ""}`;
    }),
    "}",
  ];
  return (
    <pre className="plan-json mono" data-json aria-label="Query plan">
      {json.map((l, i) => (
        <span key={i}>{l}</span>
      ))}
    </pre>
  );
}

export function Ask() {
  const words = QUERY.split(" ");
  return (
    <section className="act" data-act="ask" aria-label="Ask anything">
      <Split as="h2" className="display ask-head" text="Ask anything." data-el="head" />
      <div className="ask" data-el="bar">
        <span className="ask-glyph" aria-hidden="true" />
        <div className="ask-field">
          <p className="ask-line" data-el="typedLine">
            <span data-el="typed" />
            <span className="caret" aria-hidden="true" />
          </p>
          <p className="ask-line ask-words" data-el="words" aria-label={QUERY}>
            {words.map((w, i) => (
              <span key={i} data-word={w.toLowerCase().replace(/[^a-z?]/g, "")}>
                {w}
                {i < words.length - 1 ? " " : ""}
              </span>
            ))}
          </p>
        </div>
        <span className="ask-status mono" aria-live="polite">
          <span data-status>Listening</span>
          <span data-status>Parsing</span>
          <span data-status>Planned · fast path · 3 ms</span>
        </span>
      </div>
      <div className="slots">
        {PARSE_SLOTS.map((s) => (
          <div className={`slot slot-${s.key}`} data-el={`slot-${s.key}`} key={s.key}>
            <span className="slot-label mono">{s.label}</span>
            <span className="slot-value">
              <span data-value>{s.value}</span>
              <span className="slot-ghost" data-ghost aria-hidden="true">
                {s.words.join(" ").replace("?", "")}
              </span>
            </span>
            <code className="slot-code mono">{s.code}</code>
          </div>
        ))}
      </div>
      <PlanJson />
    </section>
  );
}

export function SearchAct() {
  const chips = ["Vehicle", "Red", "Pass through", "Main gate", "Last hour"];
  return (
    <section className="act" data-act="search" aria-label="Search through reality">
      <p className="sr-only">{FUNNEL.map((f) => `${formatInt(f.value)} ${f.label.toLowerCase()}`).join(", then ")}.</p>
      <ol className="chain mono" data-el="chain" aria-hidden="true">
        {chips.map((c) => (
          <li className="chip" data-chip key={c}>
            <i />
            {c}
          </li>
        ))}
      </ol>
      <div className="odo" data-el="odo" aria-hidden="true">
        {Array.from({ length: DIGITS }, (_, j) => {
          const place = DIGITS - 1 - j;
          return (
            <span key={j} className="odo-group">
              <span className="odo-col">
                <span className="odo-strip" data-strip>
                  {Array.from({ length: 10 }, (_, d) => (
                    <span key={d}>{d}</span>
                  ))}
                </span>
              </span>
              {place === 6 || place === 3 ? (
                <span className="odo-sep" data-sep={place}>
                  ,
                </span>
              ) : null}
            </span>
          );
        })}
      </div>
      <div className="odo-labels" aria-hidden="true">
        {FUNNEL.map((f, i) => (
          <span key={f.label} data-flabel className={i === FUNNEL.length - 1 ? "is-final" : undefined}>
            {f.label}
          </span>
        ))}
      </div>
    </section>
  );
}

export function Evidence() {
  return (
    <section className="act" data-act="evidence" aria-label="The evidence">
      <div className="osd mono" data-el="osd">
        <span className="osd-tl">
          {MATCH.camera} · {MATCH.cameraName.toUpperCase()}
          <b>2026-10-09 {MATCH.clock}</b>
        </span>
        <span className="osd-tr">
          <i className="rec" /> REC · 12 FPS
          <b>{MATCH.offset}</b>
        </span>
        <span className="osd-bl">
          TRACK {MATCH.trackId} · {MATCH.globalId}
        </span>
      </div>
      <div className="match-box" data-el="box" aria-hidden="true">
        <i className="tick tl" />
        <i className="tick tr" />
        <i className="tick bl" />
        <i className="tick br" />
        <svg preserveAspectRatio="none">
          <path data-ring />
        </svg>
        <span className="match-label mono" data-el="boxLabel">
          {MATCH.label} <b>Match {Math.round(MATCH.score * 100)}%</b>
        </span>
      </div>
      <p className="sr-only">
        Best match: {MATCH.label} on {MATCH.camera} ({MATCH.cameraName}) at {MATCH.clock}, {MATCH.offset}, match {Math.round(MATCH.score * 100)} percent.
      </p>
      <aside className="why" data-el="why" aria-label="Why this match">
        <Split as="h3" className="why-head" text="Why this match?" data-el="whyHead" />
        <ol>
          {REASONS.map((r) => (
            <li key={r.key} data-reason className={`reason reason-${r.key}`}>
              <span className="reason-viz" aria-hidden="true">
                <i />
              </span>
              <span className="reason-text">
                <span className="reason-label">{r.label}</span>
                <span className="reason-detail mono">{r.detail}</span>
              </span>
              <span className="reason-score mono">{r.value.toFixed(2)}</span>
              <span className="reason-bar" aria-hidden="true">
                <i data-bar={r.value} />
              </span>
            </li>
          ))}
        </ol>
      </aside>
    </section>
  );
}
