"use client";

// Minimal chrome: wordmark, four links, a sound switch, the chapter rail and
// the lens cursor. Updates are written straight to the DOM from the shared
// ticker; React renders this once.
import { useEffect, useRef, useState } from "react";
import { ACTS, CHAPTERS, DOM, type ActId } from "@/lib/config/animation";
import { audio } from "@/lib/audio/engine";
import { S } from "@/animation/sceneState";
import { damp } from "@/lib/math";
import { MagneticButton } from "./Magnetic";

const NAV: { label: string; act: ActId }[] = [
  { label: "System", act: "network" },
  { label: "Search", act: "ask" },
  { label: "Memory", act: "memory" },
  { label: "Contact", act: "finale" },
];

export function Nav({ go }: { go: (id: ActId) => void }) {
  const [sound, setSound] = useState(false);
  useEffect(() => audio.subscribe(setSound), []);
  return (
    <header className="nav">
      <a className="wordmark" href="#top" onClick={(e) => (e.preventDefault(), go("opening"))} aria-label="EVORA, back to the start" data-cursor="Top">
        EVORA
      </a>
      <nav aria-label="Story">
        <ul>
          {NAV.map((n) => (
            <li key={n.label}>
              <MagneticButton className="nav-link" onClick={() => go(n.act)}>
                {n.label}
              </MagneticButton>
            </li>
          ))}
        </ul>
      </nav>
      <MagneticButton className={`sound mono${sound ? " is-on" : ""}`} onClick={() => void audio.setEnabled(!sound)} label={sound ? "Turn sound off" : "Turn sound on"}>
        <span className="sound-bars" aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
        </span>
        Sound {sound ? "on" : "off"}
      </MagneticButton>
    </header>
  );
}

/** DISCOVER → SEARCH → TRACE → REMEMBER → RECONSTRUCT, drawn as a hairline that fills with the scroll. */
export function ChapterRail({ register }: { register: (fn: () => void) => () => void }) {
  const fill = useRef<HTMLSpanElement>(null);
  const items = useRef<(HTMLLIElement | null)[]>([]);
  useEffect(() => {
    let last = -1;
    let shown = -1;
    return register(() => {
      const p = Math.round(S.progress * 1000) / 1000;
      if (p !== last && fill.current) {
        last = p;
        fill.current.style.transform = `scaleX(${p})`;
      }
      const chapter = CHAPTERS.findIndex((c) => c.id === ACTS[S.act]?.chapter);
      if (chapter !== shown) {
        shown = chapter;
        items.current.forEach((li, i) => li?.toggleAttribute("data-current", i === chapter));
      }
    });
  }, [register]);
  return (
    <div className="rail mono" aria-hidden="true">
      <ol>
        {CHAPTERS.map((c, i) => (
          <li key={c.id} ref={(el) => void (items.current[i] = el)}>
            {c.label}
          </li>
        ))}
      </ol>
      <span className="rail-track">
        <span className="rail-fill" ref={fill} />
      </span>
    </div>
  );
}

/** A lens reticle that trails the pointer and names what a click will do. */
export function Cursor({ register }: { register: (fn: (dt: number) => void) => () => void }) {
  const ring = useRef<HTMLDivElement>(null);
  const label = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    if (window.matchMedia("(pointer: coarse)").matches) return;
    const el = ring.current;
    if (!el) return;
    const target = { x: window.innerWidth / 2, y: window.innerHeight / 2 };
    const pos = { ...target };
    let hot = 0;
    let hotTarget = 0;
    let lastLabel = "";
    const onMove = (e: PointerEvent) => {
      target.x = e.clientX;
      target.y = e.clientY;
      el.dataset.on = "1";
    };
    const onOver = (e: PointerEvent) => {
      const t = (e.target as HTMLElement | null)?.closest<HTMLElement>("[data-cursor], a, button");
      hotTarget = t ? 1 : 0;
      const text = t?.dataset.cursor ?? "";
      if (text !== lastLabel && label.current) {
        lastLabel = text;
        label.current.textContent = text;
      }
    };
    const onLeave = () => (el.dataset.on = "0");
    window.addEventListener("pointermove", onMove, { passive: true });
    window.addEventListener("pointerover", onOver, { passive: true });
    document.documentElement.addEventListener("pointerleave", onLeave);
    const off = register((dt) => {
      pos.x = damp(pos.x, target.x, DOM.cursorLerp, dt);
      pos.y = damp(pos.y, target.y, DOM.cursorLerp, dt);
      hot = damp(hot, hotTarget, 0.18, dt);
      el.style.transform = `translate3d(${pos.x}px, ${pos.y}px, 0)`;
      el.style.setProperty("--hot", hot.toFixed(3));
    });
    return () => {
      off();
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerover", onOver);
      document.documentElement.removeEventListener("pointerleave", onLeave);
    };
  }, [register]);
  return (
    <div className="cursor" ref={ring} aria-hidden="true" data-on="0">
      <i className="cursor-ring" />
      <i className="cursor-dot" />
      <span className="cursor-label mono" ref={label} />
    </div>
  );
}
