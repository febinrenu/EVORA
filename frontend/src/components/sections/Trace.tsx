import { HOPS } from "@/lib/data/story";

export function TraceAct() {
  return (
    <section className="act" data-act="trace" aria-label="Cross-camera trace">
      <p className="statement l-left" data-el="l1">
        One object.
        <br />
        Multiple perspectives.
      </p>
      <p className="statement l-right" data-el="l2">
        One continuous
        <br />
        story.
      </p>
      <ol className="hops mono" aria-label="Path">
        {HOPS.map((h, i) => (
          <li key={h.camera} data-hop>
            <span className="hop-cam">{h.camera}</span>
            <span className="hop-clock">{h.clock}</span>
            <span className="hop-place">{h.place}</span>
            {i < HOPS.length - 1 ? <i aria-hidden="true" /> : null}
          </li>
        ))}
      </ol>
      <div className="osd mono" data-el="osd07" aria-hidden="true">
        <span className="osd-tl">
          CAM_07 · DRIVEWAY
          <b>2026-10-09 09:16:08.517</b>
        </span>
        <span className="osd-tr">
          <i className="rec" /> RE-ID 0.81 · +105 S
        </span>
      </div>
      <div className="osd mono" data-el="osd12" aria-hidden="true">
        <span className="osd-tl">
          CAM_12 · REAR ENTRANCE
          <b>2026-10-09 09:22:41.208</b>
        </span>
        <span className="osd-tr">
          <i className="rec" /> RE-ID 0.77 · +393 S
        </span>
      </div>
    </section>
  );
}
