"use client";

// Minimal chrome: wordmark, the five chapters (each jumps to where it starts and
// the one on screen is marked), the results page, the way into the product, a
// sound switch, the progress rail and the lens cursor. Updates are written
// straight to the DOM from the shared ticker; React renders this once.
import { useEffect, useRef, useState } from "react";
import { ACTS, CHAPTERS, DOM, type ActId } from "@/lib/config/animation";
import { audio } from "@/lib/audio/engine";
import { S } from "@/animation/sceneState";
import { damp } from "@/lib/math";
import { MagneticButton, MagneticLink } from "./Magnetic";

/** each chapter link lands on the first act of that chapter */
const NAV: { id: string; label: string; act: ActId }[] = CHAPTERS.map((c) => ({ id: c.id, label: c.label, act: ACTS.find((a) => a.chapter === c.id)?.id ?? "opening" }));

type Register = (fn: () => void) => () => void;

export function Nav({ go, register }: { go: (id: ActId) => void; register: Register }) {
  const [sound, setSound] = useState(false);
  const links = useRef<(HTMLButtonElement | null)[]>([]);
  useEffect(() => audio.subscribe(setSound), []);
  useEffect(() => {
    let shown = -1;
    return register(() => {
      const chapter = NAV.findIndex((n) => n.id === ACTS[S.act]?.chapter);
      if (chapter === shown) return;
      shown = chapter;
      links.current.forEach((b, i) => {
        if (!b) return;
        if (i === chapter) b.setAttribute("aria-current", "step");
        else b.removeAttribute("aria-current");
      });
    });
  }, [register]);
  return (
    <header className="nav">
      <a className="wordmark" href="#top" onClick={(e) => (e.preventDefault(), go("opening"))} aria-label="EVORA, back to the start" data-cursor="Top">
        EVORA
      </a>
      <nav aria-label="Chapters">
        <ul>
          {NAV.map((n, i) => (
            <li key={n.id}>
              <button ref={(el) => void (links.current[i] = el)} type="button" className="nav-link" onClick={() => go(n.act)}>
                {n.label}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      {/* full page loads on purpose: leaving the story tears the WebGL context down cleanly */}
      <a className="nav-link nav-results" href="/report/">
        Results
      </a>
      <MagneticLink href="/app/" className="nav-cta">
        Open the app <span aria-hidden="true">→</span>
      </MagneticLink>
      <MagneticButton className={`sound mono${sound ? " is-on" : ""}`} onClick={() => void audio.setEnabled(!sound)} label={sound ? "Turn sound off" : "Turn sound on"}>
        <span className="sound-bars" aria-hidden="true">
          <i />
          <i />
          <i />
          <i />
        </span>
        <span className="sound-word">Sound {sound ? "on" : "off"}</span>
      </MagneticButton>
    </header>
  );
}

/** Where the story is: the scene number and name over a hairline that fills with the scroll. */
export function ChapterRail({ register }: { register: Register }) {
  const fill = useRef<HTMLSpanElement>(null);
  const index = useRef<HTMLSpanElement>(null);
  const title = useRef<HTMLSpanElement>(null);
  useEffect(() => {
    let last = -1;
    let shown = -1;
    return register(() => {
      const p = Math.round(S.progress * 1000) / 1000;
      if (p !== last && fill.current) {
        last = p;
        fill.current.style.transform = `scaleX(${p})`;
      }
      if (S.act !== shown && ACTS[S.act]) {
        shown = S.act;
        if (index.current) index.current.textContent = `${String(S.act + 1).padStart(2, "0")} / ${ACTS.length}`;
        if (title.current) title.current.textContent = ACTS[S.act].title;
      }
    });
  }, [register]);
  return (
    <div className="rail mono" aria-hidden="true">
      <p className="rail-now">
        <span className="rail-index" ref={index}>
          01 / {ACTS.length}
        </span>
        <span className="rail-title" ref={title}>
          {ACTS[0].title}
        </span>
      </p>
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
