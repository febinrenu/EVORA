// M3's extra checks (report.json `extended`): harder object windows, clarify-once
// conversations and colour against human labels. Each block carries its own
// "what" text and every number its n; blocks that are missing are left out.
import { obj, systemName, splitName } from "./data";

type Raw = Record<string, unknown>;

const num = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) ? v : null);
const two = (v: number | null): string => (v === null ? "–" : v.toFixed(2));
const text = (v: unknown): string | null => (typeof v === "string" && v ? v : null);

const NAMES: Record<string, string> = { null: "Random moment", chance: "Random order" };
const who = (k: string): string => NAMES[k] ?? systemName(k);

/** metric {value, n} cell */
function Cell({ m, unit = "" }: { m: unknown; unit?: string }) {
  const v = num(obj(m).value);
  const n = num(obj(m).n);
  return (
    <td title={n !== null ? `n = ${n}` : undefined}>
      {v === null ? "–" : unit === "s" ? `${v.toFixed(1)} s` : unit === "count" ? String(Math.round(v * 100) / 100) : v.toFixed(2)}
      {n !== null ? <span className="rp-n"> n {n}</span> : null}
    </td>
  );
}

export function ExtraChecks({ extended }: { extended: Raw }) {
  const low = obj(extended.low_chance_object);
  const conv = obj(extended.conversations);
  const colour = obj(extended.colour_retrieval);
  const verify = obj(extended.colour_verification);
  const blocks = [Object.keys(low).length, Object.keys(conv).length, Object.keys(colour).length, Object.keys(verify).length];
  if (!blocks.some(Boolean)) return null;
  return (
    <section className="rp-section" aria-labelledby="rp-extra">
      <h2 id="rp-extra">Extra checks</h2>
      <p className="rp-lead">Harder or narrower question sets run on the same frozen system. They sit next to the main results above and do not replace them.</p>
      {blocks[0] ? <LowChance r={low} /> : null}
      {blocks[1] ? <Conversations r={conv} /> : null}
      {blocks[2] ? <ColourRetrieval r={colour} /> : null}
      {blocks[3] ? <ColourVerification r={verify} /> : null}
    </section>
  );
}

function LowChance({ r }: { r: Raw }) {
  const pooled = obj(r.pooled);
  const systems = ["ours", "null", "b0"].filter((s) => s in pooled).concat(Object.keys(pooled).filter((s) => !["ours", "null", "b0"].includes(s)));
  const byTime = obj(r.timestamp_error_s_by_split);
  const splits = Object.keys(byTime);
  return (
    <div className="rp-extra">
      <h3 className="rp-subhead">Object questions where a random moment is less likely to be right</h3>
      {text(r.what) ? <p className="rp-lead">{text(r.what)}</p> : null}
      <div className="rp-table-wrap">
        <table className="rp-table">
          <thead>
            <tr>
              <th scope="col">System</th>
              <th scope="col">Hit@1</th>
              <th scope="col">Hit@5</th>
              <th scope="col">MRR</th>
            </tr>
          </thead>
          <tbody>
            {systems.map((s) => {
              const m = obj(pooled[s]);
              return (
                <tr key={s} className={s === "ours" ? "is-ours" : undefined}>
                  <th scope="row">{who(s)}</th>
                  <Cell m={m["object.hit@1"]} />
                  <Cell m={m["object.hit@5"]} />
                  <Cell m={m["object.mrr"]} />
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {splits.length ? (
        <div className="rp-table-wrap rp-gap">
          <table className="rp-table">
            <caption className="rp-caption">Median distance from the true moment, by split</caption>
            <thead>
              <tr>
                <th scope="col">Split</th>
                {systems.map((s) => (
                  <th key={s} scope="col">
                    {who(s)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {splits.map((sp) => (
                <tr key={sp}>
                  <th scope="row">{splitName(sp)}</th>
                  {systems.map((s) => (
                    <Cell key={s} m={obj(byTime[sp])[s]} unit="s" />
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}

const CONV: [string, string, string?][] = [
  ["ask_recall", "First mentions that were asked about"],
  ["ask_precision", "Questions asked that were needed"],
  ["reask_count", "Asked again after it was answered", "count"],
];

function Conversations({ r }: { r: Raw }) {
  const ours = obj(obj(r.pooled).ours);
  return (
    <div className="rp-extra">
      <h3 className="rp-subhead">Asking once about a place, then remembering it</h3>
      {text(r.what) ? <p className="rp-lead">{text(r.what)}</p> : null}
      <div className="rp-table-wrap">
        <table className="rp-table">
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col">EVORA</th>
            </tr>
          </thead>
          <tbody>
            {CONV.filter(([k]) => k in ours).map(([k, label, unit]) => (
              <tr key={k}>
                <th scope="row">{label}</th>
                <Cell m={ours[k]} unit={unit} />
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ColourRetrieval({ r }: { r: Raw }) {
  const rows: [string, Raw][] = [["All labelled tracks", obj(r.all)], ...Object.entries(obj(r.by_kind)).map(([k, v]): [string, Raw] => [`${k[0].toUpperCase()}${k.slice(1)}s`, obj(v)])];
  const cols: [string, string][] = [
    ["attributes_on", "Colour used"],
    ["attributes_off", "Colour ignored"],
    ["chance", "Random order"],
  ];
  return (
    <div className="rp-extra">
      <h3 className="rp-subhead">Asking for a colour, scored against human labels</h3>
      {text(r.what) ? <p className="rp-lead">{text(r.what)}</p> : null}
      <div className="rp-table-wrap">
        <table className="rp-table">
          <thead>
            <tr>
              <th scope="col">Tracks</th>
              <th scope="col">n</th>
              {cols.map(([k, label]) => (
                <th key={k} scope="col">
                  {label}, Hit@1
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows
              .filter(([, v]) => Object.keys(v).length)
              .map(([label, v]) => (
                <tr key={label}>
                  <th scope="row">{label}</th>
                  <td>{num(v.n) ?? "–"}</td>
                  {cols.map(([k]) => (
                    <td key={k}>{two(num(obj(v[k])["hit@1"]))}</td>
                  ))}
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function ColourVerification({ r }: { r: Raw }) {
  const v = obj(r.verifier);
  const e = obj(r.effect);
  const rows: [string, number | null, number | null][] = [
    ["Said yes when the colour was right", num(v.yes_on_positives), num(v.n_positives)],
    ["Said no when the colour was wrong", num(v.no_on_negatives), num(v.n_negatives)],
  ];
  return (
    <div className="rp-extra">
      <h3 className="rp-subhead">The second look at colour, against human labels</h3>
      {text(r.what) ? <p className="rp-lead">{text(r.what)}</p> : null}
      <div className="rp-table-wrap">
        <table className="rp-table">
          <thead>
            <tr>
              <th scope="col">Measure</th>
              <th scope="col">Value</th>
              <th scope="col">n</th>
            </tr>
          </thead>
          <tbody>
            {rows.map(([label, val, n]) => (
              <tr key={label}>
                <th scope="row">{label}</th>
                <td>{two(val)}</td>
                <td>{n ?? "–"}</td>
              </tr>
            ))}
            {Object.keys(e).length ? (
              <>
                <tr>
                  <th scope="row">Hit@1 before and after the check</th>
                  <td>
                    {two(num(e.hit1_before))} to {two(num(e.hit1_after))}
                  </td>
                  <td>{num(e.n) ?? "–"}</td>
                </tr>
                <tr>
                  <th scope="row">Share of right colours among the evidence, before and after</th>
                  <td>
                    {two(num(e.precision_before))} to {two(num(e.precision_after))}
                  </td>
                  <td>{num(e.n) ?? "–"}</td>
                </tr>
                {num(e.lost_all_labelled) ? (
                  <tr>
                    <th scope="row">Questions where the check set aside every labelled track</th>
                    <td>{num(e.lost_all_labelled)}</td>
                    <td>{num(e.n_queries) ?? "–"}</td>
                  </tr>
                ) : null}
              </>
            ) : null}
          </tbody>
        </table>
      </div>
    </div>
  );
}
