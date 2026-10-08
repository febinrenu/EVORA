"use client";

import { useEffect, useState } from "react";
import { Split } from "@/components/ui/Split";
import { TWIN_CAMS } from "@/lib/data/site";
import { ENTITIES, MEMORY_QUESTION, PLACES, type EntitySpec, type PlaceSpec } from "@/lib/data/story";
import { SitePlan, planPct } from "./SitePlan";

export function MemoryAct() {
  return (
    <section className="act" data-act="memory" data-chosen="0" aria-label="Clarify once">
      <Split as="h2" className="statement memory-head" text="Sometimes the system needs to ask." data-el="l1" />
      <div className="ask ask-small" data-el="bar">
        <span className="ask-glyph" aria-hidden="true" />
        <div className="ask-field">
          <p className="ask-line">
            <span data-el="typed" />
            <span className="caret" aria-hidden="true" />
          </p>
        </div>
      </div>
      <div className="clarify" data-el="card">
        <div className="marker-tab mono">Clarify once</div>
        <p className="clarify-q">{MEMORY_QUESTION}</p>
        <div className="clarify-body" data-el="map">
          <div className="clarify-plan" aria-hidden="true">
            <SitePlan />
            {TWIN_CAMS.map((c) => (
              <span key={c.id} className="plan-dot mono" data-dot={c.id} style={planPct(c.pos[0], c.pos[2])}>
                <i />
                {c.id.slice(4)}
              </span>
            ))}
          </div>
          <div className="clarify-grid" role="group" aria-label="Cameras">
            {TWIN_CAMS.map((c) => (
              <button key={c.id} type="button" className="pick" data-pick={c.id} aria-pressed="false" aria-label={`Camera ${c.id}`} data-cursor="Choose">
                <canvas data-feed={c.id} data-car="none" width={480} height={270} />
                <span className="pick-label mono">{c.id}</span>
              </button>
            ))}
          </div>
        </div>
      </div>
      <div className="inspect" data-el="inspect">
        <div className="inspect-frame">
          <canvas data-inspect-feed width={480} height={270} />
          <svg data-draw viewBox="0 0 480 270" data-cursor="Draw">
            <path data-stroke />
          </svg>
        </div>
        <p className="inspect-hint mono" data-draw-hint>
          Mark the gate line, or let EVORA do it
        </p>
      </div>
      <p className="saved" data-el="saved" role="status">
        <span className="saved-check" aria-hidden="true">
          ✓
        </span>
        <span className="mono">Memory saved</span>
        <b>Main gate</b>
        <span className="mono" data-chosen-cam>
          CAM_04
        </span>
        <span className="mono">Region 02</span>
      </p>
      <p className="statement remember" data-el="remember">
        EVORA will remember this.
        <span>Not for this chat. For this place.</span>
      </p>
    </section>
  );
}

type Selected = { kind: "place"; item: PlaceSpec } | { kind: "entity"; item: EntitySpec } | null;

/** The feed shown for a node on hover: places use their camera, entities their best sighting. */
const feedOf = (s: NonNullable<Selected>): { cam: string; car: string } =>
  s.kind === "place"
    ? { cam: s.item.camera, car: "none" }
    : s.item.id === "red-sedan"
      ? { cam: "CAM_12", car: "12" }
      : s.item.id === "black-suv"
        ? { cam: "CAM_18", car: "none" }
        : { cam: "CAM_02", car: "none" };

export function MemoryMapAct({ onDrawer, paint }: { onDrawer: (open: boolean) => void; paint: (root: ParentNode) => void }) {
  const [hover, setHover] = useState<Selected>(null);
  const [open, setOpen] = useState<Selected>(null);

  useEffect(() => {
    onDrawer(open !== null);
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(null);
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onDrawer]);

  useEffect(() => {
    // repaint feed canvases that just mounted
    paint(document);
  }, [hover, open, paint]);

  const card = hover ?? null;
  return (
    <section className="act" data-act="memoryMap" aria-label="The memory map">
      <Split as="h2" className="statement map-head" text="Memory, not history." data-el="head" />
      <p className="map-sub" data-el="sub">
        Places and things EVORA has learned, pinned to the site they belong to. Hover a marker to see the evidence.
      </p>
      <ul className="map-labels" aria-label="Known places and entities">
        {PLACES.map((p) => (
          <li key={p.id}>
            <button
              type="button"
              className={`map-label is-place${p.id === "main-gate" ? " is-new" : ""}`}
              data-label={p.id}
              data-cursor="Open"
              onPointerEnter={() => setHover({ kind: "place", item: p })}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover({ kind: "place", item: p })}
              onBlur={() => setHover(null)}
              onClick={() => setOpen({ kind: "place", item: p })}
            >
              <span className="map-label-name">{p.label}</span>
              <span className="map-label-meta mono">
                {p.camera} · used {p.uses}×
              </span>
            </button>
          </li>
        ))}
        {ENTITIES.map((e) => (
          <li key={e.id}>
            <button
              type="button"
              className={`map-label is-entity${e.id === "red-sedan" ? " is-match" : ""}`}
              data-label={e.id}
              data-cursor="Open"
              onPointerEnter={() => setHover({ kind: "entity", item: e })}
              onPointerLeave={() => setHover(null)}
              onFocus={() => setHover({ kind: "entity", item: e })}
              onBlur={() => setHover(null)}
              onClick={() => setOpen({ kind: "entity", item: e })}
            >
              <span className="map-label-name">{e.label}</span>
              <span className="map-label-meta mono">{e.seen}</span>
            </button>
          </li>
        ))}
      </ul>
      <div className={`map-card${card ? " is-on" : ""}`} aria-hidden="true">
        {card ? (
          <>
            <canvas data-feed={feedOf(card).cam} data-car={feedOf(card).car} width={480} height={270} />
            <p className="mono">
              {feedOf(card).cam} · {card.kind === "place" ? card.item.learned : card.item.seen}
            </p>
          </>
        ) : null}
      </div>
      <div className={`drawer${open ? " is-open" : ""}`} role="dialog" aria-modal="false" aria-label={open ? open.item.label : "Details"} hidden={!open}>
        {open ? (
          <>
            <button type="button" className="drawer-close mono" onClick={() => setOpen(null)} data-cursor="Close">
              Close
            </button>
            <p className={`drawer-kind mono${open.kind === "place" ? " is-place" : ""}`}>{open.kind === "place" ? "Known place" : "Entity"}</p>
            <h3 className="drawer-title">{open.item.label}</h3>
            <canvas data-feed={feedOf(open).cam} data-car={feedOf(open).car} width={480} height={270} />
            {open.kind === "place" ? (
              <dl className="drawer-facts mono">
                <dt>Camera</dt>
                <dd>{open.item.camera}</dd>
                <dt>Source</dt>
                <dd>{open.item.learned}</dd>
                <dt>Used</dt>
                <dd>{open.item.uses} times</dd>
              </dl>
            ) : (
              <ol className="drawer-events">
                {open.item.events.map((ev) => (
                  <li key={ev.clock + ev.camera}>
                    <span className="mono">{ev.clock}</span>
                    <span className="mono">{ev.camera}</span>
                    <span>{ev.what}</span>
                  </li>
                ))}
              </ol>
            )}
          </>
        ) : null}
      </div>
    </section>
  );
}
