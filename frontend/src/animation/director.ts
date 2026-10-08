// The director: Lenis -> GSAP ticker -> one ScrollTrigger -> one master
// timeline -> DOM + WebGL state. There is exactly one animation loop on the
// page (the GSAP ticker); everything else hangs off it.
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import Lenis from "lenis";
import { S } from "@/animation/sceneState";
import { ACTS, LENGTH_SCALE, LENIS, type ActId } from "@/lib/config/animation";
import { damp } from "@/lib/math";
import type { Ctx } from "./timelines/helpers";
import { network, opening, universe, wall } from "./timelines/discover";
import { ask, evidence, search } from "./timelines/search";
import { memory, memoryMap, timeline, trace } from "./timelines/remember";
import { finale, reconstruct } from "./timelines/reconstruct";

const BUILDERS: Record<ActId, (c: Ctx) => void> = {
  opening,
  universe,
  network,
  wall,
  ask,
  search,
  evidence,
  trace,
  memory,
  memoryMap,
  timeline,
  reconstruct,
  finale,
};

export interface DirectorOptions {
  root: HTMLElement;
  track: HTMLElement;
  flights: boolean;
  frame: (dtMs: number) => void;
  onAct?: (index: number) => void;
}

export interface Director {
  lenis: Lenis | null;
  scrollToAct: (id: ActId) => void;
  /** start time of each act on the master timeline, in viewport heights */
  starts: () => number[];
  destroy: () => void;
}

export function createDirector(opts: DirectorOptions): Director {
  gsap.registerPlugin(ScrollTrigger);
  ScrollTrigger.config({ ignoreMobileResize: true });
  const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const coarse = window.matchMedia("(pointer: coarse)").matches;

  const lenis = reduced
    ? null
    : new Lenis({ lerp: LENIS.lerp, wheelMultiplier: LENIS.wheelMultiplier, touchMultiplier: LENIS.touchMultiplier, autoRaf: false });
  lenis?.on("scroll", ScrollTrigger.update);

  // pointer: raw values from the event, smoothed once per frame in the ticker
  const raw = { x: 0, y: 0, t: -10 };
  const onPointer = (e: PointerEvent) => {
    if (e.pointerType === "touch") return;
    raw.x = (e.clientX / window.innerWidth) * 2 - 1;
    raw.y = -((e.clientY / window.innerHeight) * 2 - 1);
    raw.t = S.time;
  };
  window.addEventListener("pointermove", onPointer, { passive: true });

  let lastAct = -1;
  let starts: number[] = [];

  const tick = (time: number, deltaMs: number) => {
    S.time = time;
    lenis?.raf(time * 1000);
    const active = !coarse && time - raw.t < 2.5 ? 1 : 0;
    S.pointer.x = damp(S.pointer.x, raw.x, 0.08, deltaMs);
    S.pointer.y = damp(S.pointer.y, raw.y, 0.08, deltaMs);
    S.pointer.active = damp(S.pointer.active, active, 0.05, deltaMs);
    opts.frame(deltaMs);
  };
  gsap.ticker.add(tick);
  gsap.ticker.lagSmoothing(0);

  const mm = gsap.matchMedia();
  mm.add(
    {
      desktop: "(min-width: 1024px)",
      tablet: "(min-width: 640px) and (max-width: 1023.98px)",
      mobile: "(max-width: 639.98px)",
    },
    (context) => {
      const cond = context.conditions as { desktop: boolean; tablet: boolean; mobile: boolean };
      const scale = cond.mobile ? LENGTH_SCALE.mobile : cond.tablet ? LENGTH_SCALE.tablet : LENGTH_SCALE.desktop;
      const tl = gsap.timeline({ paused: true, defaults: { ease: "none" } });
      let t = 0;
      starts = [];
      for (const spec of ACTS) {
        const len = spec.length * scale;
        starts.push(t);
        BUILDERS[spec.id]({
          gsap,
          tl,
          root: opts.root,
          mobile: cond.mobile,
          tablet: cond.tablet,
          reduced,
          flights: opts.flights && !reduced,
          t0: t,
          len,
          base: spec.length,
          act: spec.id,
        });
        t += len;
      }
      // pad so the last act's final tweens fully resolve before the end of the scroll
      tl.to({}, { duration: 0.0001 }, t);
      opts.track.style.height = `${(t + 1) * 100}lvh`;
      const total = t;

      ScrollTrigger.create({
        trigger: opts.track,
        start: "top top",
        end: "bottom bottom",
        animation: tl,
        scrub: lenis ? true : 0.4,
        invalidateOnRefresh: true,
        onUpdate: (self) => {
          S.progress = self.progress;
          const now = self.progress * total;
          let idx = 0;
          for (let i = 0; i < starts.length; i++) if (now >= starts[i] - 1e-6) idx = i;
          S.act = idx;
          if (idx !== lastAct) {
            lastAct = idx;
            opts.onAct?.(idx);
          }
        },
      });
      ScrollTrigger.refresh();
    },
  );

  const onVisibility = () => {
    if (document.hidden) lenis?.stop();
    else lenis?.start();
  };
  document.addEventListener("visibilitychange", onVisibility);

  return {
    lenis,
    starts: () => starts,
    scrollToAct: (id) => {
      const i = ACTS.findIndex((a) => a.id === id);
      if (i < 0) return;
      // land a little into the act so its layer is already in
      const y = (starts[i] + 0.25) * window.innerHeight;
      if (lenis) lenis.scrollTo(y, { duration: Math.min(3.2, 0.9 + Math.abs(window.scrollY - y) / window.innerHeight / 9) });
      else window.scrollTo({ top: y, behavior: reduced ? "auto" : "smooth" });
    },
    destroy: () => {
      gsap.ticker.remove(tick);
      window.removeEventListener("pointermove", onPointer);
      document.removeEventListener("visibilitychange", onVisibility);
      mm.revert();
      lenis?.destroy();
      ScrollTrigger.getAll().forEach((st) => st.kill());
    },
  };
}
