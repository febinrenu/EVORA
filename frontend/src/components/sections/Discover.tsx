import { Split } from "@/components/ui/Split";
import { buildNetwork, isTwinCam } from "@/lib/data/site";
import { COUNT_BEATS, TOTALS } from "@/lib/data/story";
import { formatInt } from "@/lib/math";

const COUNT_NOTES = [
  "Fixed, dome and phone cameras on one clock",
  "Every detection, with a time and a place",
  "Things that move, followed frame to frame",
  "Appearance only. No faces, ever",
  "Type, colour and direction",
];

// The wall's solo feed is the centre of the first nine people feeds.
const SOLO = buildNetwork(24).filter((n) => !isTwinCam(n.id))[4];

export function Opening() {
  return (
    <section className="act" data-act="opening" aria-labelledby="h-opening">
      <h1 id="h-opening" className="display title" data-el="title">
        <Split className="line" text="Everything" data-line="Everything" />
        <Split className="line" text="leaves" data-line="leaves" />
        <Split className="line" text="a trace." data-line="a trace." />
      </h1>
      <p className="meta mono" data-el="meta">
        <span>{TOTALS.cameras} cameras</span>
        <span aria-hidden="true">·</span>
        <span>8.4M events</span>
        <span aria-hidden="true">·</span>
        <span>Continuously indexed</span>
      </p>
      <p className="statement fragment" data-el="fragment">
        Every camera sees a fragment.
      </p>
      <div className="cue mono" data-el="cue" aria-hidden="true">
        <span className="cue-text">
          <span className="cue-wait">Calibrating 24 lenses</span>
          <span className="cue-go">Scroll to enter</span>
        </span>
        <i />
      </div>
    </section>
  );
}

export function Universe() {
  return (
    <section className="act" data-act="universe" aria-label="The event universe">
      <p className="sr-only">
        {COUNT_BEATS.map((b) => `${formatInt(b.value)} ${b.label.toLowerCase()}`).join(", ")}.
      </p>
      <div className="counts" aria-hidden="true">
        {COUNT_BEATS.map((b, i) => (
          <div className="count" data-count key={b.label}>
            <span className="count-label">{b.label}</span>
            <span className="count-note mono">{COUNT_NOTES[i]}</span>
          </div>
        ))}
      </div>
    </section>
  );
}

export function Network() {
  return (
    <section className="act" data-act="network" aria-label="The camera network">
      <p className="statement l-left" data-el="l1">
        One place.
        <br />
        Many perspectives.
      </p>
      <Split as="h2" className="display l-centre" text="EVORA connects them." data-el="l2" />
      <p className="legend mono" data-el="legend">
        <span>24 cameras</span>
        <span>41 links</span>
        <span>Travel times learned, not configured</span>
      </p>
    </section>
  );
}

export function Wall() {
  return (
    <section className="act" data-act="wall" aria-label="The camera wall">
      <p className="statement wall-caption" data-el="caption">
        Hours of footage.
        <br />
        No one watching.
      </p>
      <p className="legend mono wall-count" data-el="count">
        <span>9 feeds</span>
        <span>216 hours today</span>
        <span>0 reviewed</span>
      </p>
      <div className="osd mono" data-el="osd" aria-hidden="true">
        <span className="osd-tl">
          {SOLO.id} · {SOLO.label.toUpperCase()}
        </span>
        <span className="osd-tr">
          <i className="rec" /> REC 09:14:21
        </span>
        <span className="osd-br">640×360 · H.264 · 25 FPS</span>
      </div>
    </section>
  );
}
