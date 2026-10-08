// RECONSTRUCT: every layer at once, then silence, then the lights come on.
import { S } from "@/animation/sceneState";
import { HOP_U } from "@/lib/data/site";
import { sfx } from "@/lib/audio/engine";
import { actLayer, all, at, cam, camSet, chars, cue, dur, el, line, setMorph, type Ctx, reveal } from "./helpers";

export function reconstruct(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  tl.set(S.twin, { car: HOP_U.CAM_12, trace: 0, traceAlpha: 0, cctv: 0, iris: 1, look: 0.12, beacons: 0, gate: 0, ghosts: 0, recon: 0 }, at(c, 0));
  tl.set(S.cam, { follow: 0, sway: 1 }, at(c, 0));
  camSet(c, { pos: [78, 96, 104], look: [8, 0, -14] }, 0);
  setMorph(c, "wRecon", 0);
  tl.set(S.universe, { wText: 0, stage: 0, redReveal: 0, highlight: 0, highlightClass: -1, size: 1, burst: 0 }, at(c, 0));

  tl.to(S.twin, { alpha: 1, duration: dur(c, 0.6), ease: "none" }, at(c, 0));
  tl.to(S.universe, { alpha: 0.32, size: 0.8, duration: dur(c, 0.9) }, at(c, 0.3));
  cam(c, { pos: [-86, 84, 96], look: [10, 0, -16] }, 0, 5.0, "none");
  chars(c, el(c, "head"), 0.1, 1.3, { stagger: 0.035 });

  tl.to(S.twin, { traceAlpha: 1, duration: dur(c, 0.4) }, at(c, 0.9));
  tl.to(S.twin, { trace: 1, duration: dur(c, 1.3), ease: "power2.inOut" }, at(c, 0.9));
  tl.to(S.twin, { ghosts: 1, duration: dur(c, 0.8) }, at(c, 1.0));
  tl.to(S.twin, { recon: 1, duration: dur(c, 0.8), ease: "power2.out" }, at(c, 1.4));
  tl.to(S.twin, { beacons: 0.3, gate: 0.6, duration: dur(c, 0.6) }, at(c, 1.4));
  cue(c, 1.45, () => sfx("pulse"));
  reveal(c, all(c, "[data-card]"), { autoAlpha: 0 }, { autoAlpha: 1, stagger: dur(c, 0.08), duration: dur(c, 0.25) }, at(c, 1.7));

  reveal(c, el(c, "rail"), { "--draw": 0 }, { "--draw": 1, duration: dur(c, 0.9), ease: "power2.inOut" }, at(c, 2.0));
  all(c, "[data-summary]").forEach((row, k) => {
    reveal(c, row, { autoAlpha: 0, x: -12 }, { autoAlpha: 1, x: 0, duration: dur(c, 0.25) }, at(c, 2.2 + k * 0.28));
    cue(c, 2.2 + k * 0.28, () => sfx("tick"));
  });
  line(c, el(c, "verdict"), 3.1, 4.25);

  // everything drifts apart and dims
  tl.to(all(c, "[data-card], [data-summary]"), { autoAlpha: 0, duration: dur(c, 0.3) }, at(c, 4.25));
  tl.to(el(c, "rail"), { autoAlpha: 0, duration: dur(c, 0.3) }, at(c, 4.25));
  tl.to(S.universe, { burst: 1, alpha: 0, duration: dur(c, 0.7), ease: "power2.in" }, at(c, 4.3));
  tl.to(S.twin, { alpha: 0, recon: 0, ghosts: 0, duration: dur(c, 0.6), ease: "power2.in" }, at(c, 4.35));
}

export function finale(c: Ctx): void {
  const { tl } = c;
  actLayer(c, { last: true });
  line(c, el(c, "l1"), 0.3, 1.2);
  chars(c, el(c, "head"), 1.3, 2.45, { stagger: 0.022 });
  // lights on: the unlit glass becomes the light table the product lives on
  tl.to(document.documentElement, { "--lights": 1, duration: dur(c, 0.8), ease: "power2.inOut" }, at(c, 2.2));
  tl.to(S, { lights: 1, duration: dur(c, 0.8), ease: "power2.inOut" }, at(c, 2.2));
  chars(c, el(c, "brand"), 2.75, null, { stagger: 0.05 });
  line(c, el(c, "tagline"), 3.05, null);
  line(c, el(c, "cta"), 3.25, null, { y: 10 });
  reveal(c, el(c, "cta"), { "--draw": 0 }, { "--draw": 1, duration: dur(c, 0.55), ease: "power2.inOut" }, at(c, 3.4));
  line(c, el(c, "foot"), 3.55, null, { y: 0 });
}
