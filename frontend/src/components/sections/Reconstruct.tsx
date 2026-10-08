import Link from "next/link";
import { Split } from "@/components/ui/Split";
import { HOPS, MATCH, RECON_CAMS } from "@/lib/data/story";
import { MagneticLink } from "@/components/ui/Magnetic";

const LANES = ["People", "Vehicles", "Objects", "Events"];
const TICKS = 16;

export function TimelineAct() {
  return (
    <section className="act" data-act="timeline" aria-label="Zoom into time">
      <Split as="h2" className="display tl-head" text="Zoom into time." data-el="head" />
      <div className="tl-lanes" aria-hidden="true">
        {LANES.map((l, i) => (
          <span key={l} data-lane style={{ top: `${50 - (1.5 - i) * 11.5}%` }} className="mono">
            {l}
          </span>
        ))}
      </div>
      <div className="tl-axis mono" data-el="axis" aria-hidden="true">
        {Array.from({ length: TICKS }, (_, i) => (
          <span key={i} data-tick />
        ))}
      </div>
      <div className="tl-match mono" data-el="match" aria-hidden="true">
        <i />
        <span>
          {MATCH.label} · {MATCH.camera} · {MATCH.clock}
        </span>
      </div>
      <p className="tl-readout mono" data-el="readout">
        <span>Span</span> <b data-el="span">4.20 h</b> <span>Focus 09:14:22</span>
      </p>
      <p className="statement tl-caption" data-el="caption">
        Every event keeps its moment,
        <br />
        down to the frame.
      </p>
    </section>
  );
}

const RECON_NOTE: Record<string, string> = {
  CAM_04: "09:14:23 · Main gate",
  CAM_07: "09:16:08 · Driveway",
  CAM_12: "09:22:41 · Rear entrance",
  CAM_18: "Checked · not seen",
};

export function ReconstructAct() {
  return (
    <section className="act" data-act="reconstruct" aria-label="Reconstruct">
      <Split as="h2" className="display recon-head" text="Reconstruct." data-el="head" />
      <ul className="recon-cards mono" aria-hidden="true">
        {RECON_CAMS.map((id) => (
          <li key={id} data-card={id} className={id === "CAM_18" ? "is-negative" : undefined}>
            <b>{id}</b> {RECON_NOTE[id]}
          </li>
        ))}
      </ul>
      <div className="recon-summary">
        <p className="recon-entity mono">
          {MATCH.label} · {MATCH.globalId}
        </p>
        <ol>
          {HOPS.map((h) => (
            <li key={h.camera} data-summary>
              <span className="mono">{h.clock}</span>
              <b>{h.place}</b>
              <span className="mono">{h.camera}</span>
            </li>
          ))}
        </ol>
        <svg className="recon-rail" data-el="rail" viewBox="0 0 300 24" aria-hidden="true">
          <path d="M6 12 H294" pathLength="1" />
          {/* hop times on a 0..498 s axis: +0, +105, +498 */}
          {[6, 67, 294].map((x) => (
            <circle key={x} cx={x} cy={12} r={4} />
          ))}
        </svg>
        <p className="recon-verdict" data-el="verdict">
          Path reconstructed across three cameras. Every step cites its frame.
        </p>
      </div>
    </section>
  );
}

export function Finale() {
  return (
    <section className="act" data-act="finale" aria-label="Search the memory of a place">
      <p className="finale-l1" data-el="l1">
        Your cameras already saw it.
      </p>
      <Split as="h2" className="display finale-head" text="You just couldn't search it." data-el="head" />
      <div className="finale-brand">
        <Split as="p" className="mega" text="EVORA" data-el="brand" />
        <p className="finale-tagline" data-el="tagline">
          Search the memory of a place.
        </p>
        <div className="cta-wrap" data-el="cta">
          <MagneticLink href="/app" className="cta">
            <span>Enter EVORA</span>
            <span aria-hidden="true" className="cta-arrow">
              →
            </span>
            <svg viewBox="0 0 240 8" preserveAspectRatio="none" aria-hidden="true">
              <path d="M2 5 C 60 2, 120 7, 238 3" pathLength="1" />
            </svg>
          </MagneticLink>
        </div>
      </div>
      <p className="finale-foot mono" data-el="foot">
        <span>HNX26EPS05</span>
        <span>Runs on your own machine</span>
        <span>No face recognition</span>
        <Link href="/app">Open the product</Link>
      </p>
    </section>
  );
}
