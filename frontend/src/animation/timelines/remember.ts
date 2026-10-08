// TRACE + REMEMBER: follow the sedan between cameras through the twin, the
// one clarifying question, the memory map, and the zoom into time.
import { S } from "@/animation/sceneState";
import { HOP_U, twinCam } from "@/lib/data/site";
import { MEMORY_QUERY } from "@/lib/data/story";
import { sfx } from "@/lib/audio/engine";
import { actLayer, all, at, cam, camSet, chars, cue, dur, el, line, morph, setMorph, type Ctx, reveal } from "./helpers";
import { typeInto } from "./search";

export function trace(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  const c07 = twinCam("CAM_07");
  const c12 = twinCam("CAM_12");
  const after04 = HOP_U.CAM_04 + 0.004;

  // out of CAM_04's lens and into the world
  tl.to(S.twin, { cctv: 0, look: 0.06, duration: dur(c, 0.8), ease: "power2.inOut" }, at(c, 0.05));
  tl.to(S.cam, { follow: 1, duration: dur(c, 0.9), ease: "power2.inOut" }, at(c, 0.15));
  tl.to(S.twin, { traceAlpha: 1, duration: dur(c, 0.5) }, at(c, 0.3));
  tl.set(S.twin, { trace: after04 }, at(c, 0));
  tl.to(S.twin, { car: HOP_U.CAM_07, trace: HOP_U.CAM_07, duration: dur(c, 2.3), ease: "power1.inOut" }, at(c, 0.1));
  cue(c, 0.2, () => sfx("whoosh"));

  // into CAM_07: the rig is placed at the camera while hidden by the chase, then the chase releases
  camSet(c, { pos: [...c07.pos], look: [...c07.look] }, 1.5);
  tl.to(S.cam, { follow: 0, duration: dur(c, 0.65), ease: "power2.inOut" }, at(c, 1.85));
  tl.to(S.twin, { cctv: 1, look: 1, duration: dur(c, 0.45), ease: "power2.inOut" }, at(c, 2.15));
  tl.to(S.twin, { car: HOP_U.CAM_07 + 0.025, trace: HOP_U.CAM_07 + 0.025, duration: dur(c, 1.0), ease: "none" }, at(c, 2.4));
  line(c, el(c, "osd07"), 2.6, 3.35, { y: 0 });

  // out again, along the road, into CAM_12
  tl.to(S.twin, { cctv: 0, look: 0.06, duration: dur(c, 0.5), ease: "power2.inOut" }, at(c, 3.4));
  tl.to(S.cam, { follow: 1, duration: dur(c, 0.6), ease: "power2.inOut" }, at(c, 3.45));
  tl.to(S.twin, { car: HOP_U.CAM_12, trace: HOP_U.CAM_12, duration: dur(c, 1.9), ease: "power1.inOut" }, at(c, 3.45));
  cue(c, 3.5, () => sfx("whoosh"));
  camSet(c, { pos: [...c12.pos], look: [...c12.look] }, 4.4);
  tl.to(S.cam, { follow: 0, duration: dur(c, 0.6), ease: "power2.inOut" }, at(c, 4.85));
  tl.to(S.twin, { cctv: 1, look: 1, duration: dur(c, 0.45), ease: "power2.inOut" }, at(c, 5.15));
  line(c, el(c, "osd12"), 5.55, 6.1, { y: 0 });

  line(c, el(c, "l1"), 0.55, 1.75);
  line(c, el(c, "l2"), 3.7, 4.85);
  all(c, "[data-hop]").forEach((hop, i) => {
    reveal(c, hop, { autoAlpha: 0, x: -10 }, { autoAlpha: 1, x: 0, duration: dur(c, 0.25) }, at(c, [0.3, 2.6, 5.55][i]));
  });
  tl.to(S.twin, { alpha: 0, duration: dur(c, 0.3), ease: "none" }, at(c, 6.1));
}

export function memory(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  tl.set(S.twin, { traceAlpha: 0, cctv: 0, look: 0, iris: 1 }, at(c, 0.05));
  chars(c, el(c, "l1"), 0.1, 0.9, { stagger: 0.012 });
  const bar = el(c, "bar");
  reveal(c, bar, { autoAlpha: 0, clipPath: "inset(0 50% 0 50%)" }, { autoAlpha: 1, clipPath: "inset(0 0% 0 0%)", duration: dur(c, 0.35), ease: "power3.inOut" }, at(c, 0.95));
  typeInto(c, el(c, "typed"), MEMORY_QUERY, 1.15, 0.6);
  const card = el(c, "card");
  reveal(c, card, { autoAlpha: 0, y: 24 }, { autoAlpha: 1, y: 0, duration: dur(c, 0.35), ease: "power3.out" }, at(c, 1.85));
  cue(c, 1.9, () => sfx("tick"));
  reveal(c, all(c, "[data-pick]"), { autoAlpha: 0, scale: 0.92 }, { autoAlpha: 1, scale: 1, stagger: dur(c, 0.05), duration: dur(c, 0.3) }, at(c, 2.05));
  reveal(c, el(c, "map"), { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.35) }, at(c, 2.0));
  // if the visitor has not chosen by now, the system shows the gesture itself
  tl.to(S.memory, { auto: 1, duration: dur(c, 0.05), ease: "none" }, at(c, 2.9));
  reveal(c, el(c, "inspect"), { autoAlpha: 0, y: 12 }, { autoAlpha: 1, y: 0, duration: dur(c, 0.3) }, at(c, 3.0));
  tl.to(S.memory, { draw: 1, duration: dur(c, 0.5), ease: "power1.inOut" }, at(c, 3.2));
  reveal(c, el(c, "saved"), { autoAlpha: 0, scale: 0.94 }, { autoAlpha: 1, scale: 1, duration: dur(c, 0.25), ease: "back.out(2)" }, at(c, 3.8));
  cue(c, 3.82, () => sfx("confirm"));
  line(c, el(c, "remember"), 4.0, 4.6);
  tl.to([card, bar, el(c, "inspect")], { autoAlpha: 0, y: -16, duration: dur(c, 0.3), ease: "power2.in" }, at(c, 4.5));
  tl.to(el(c, "saved"), { scale: 0.6, y: -40, autoAlpha: 0, duration: dur(c, 0.35), ease: "power2.in" }, at(c, 4.6));
}

export function memoryMap(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  tl.set(S.twin, { car: HOP_U.CAM_12, trace: HOP_U.CAM_12, cctv: 0, iris: 1, look: 0, ghosts: 0, recon: 0, beacons: 0, gate: 0 }, at(c, 0));
  tl.set(S.cam, { follow: 0, sway: 1 }, at(c, 0));
  camSet(c, { pos: [10, 160, 150], look: [6, 0, -8] }, 0);
  tl.to(S.twin, { alpha: 1, duration: dur(c, 0.6), ease: "none" }, at(c, 0));
  tl.to(S.twin, { gate: 1, duration: dur(c, 0.6), ease: "power2.out" }, at(c, 0.2));
  tl.to(S.twin, { beacons: 1, duration: dur(c, 0.8) }, at(c, 0.5));
  tl.to(S.twin, { traceAlpha: 0.35, duration: dur(c, 0.6) }, at(c, 0.6));
  cam(c, { pos: [-118, 92, 46], look: [4, 0, -10] }, 0, 2.2);
  cam(c, { pos: [70, 104, 118], look: [8, 0, -6] }, 2.2, 1.8);
  chars(c, el(c, "head"), 0.5, 3.6, { stagger: 0.02 });
  line(c, el(c, "sub"), 1.0, 3.6);
  reveal(c, all(c, "[data-label]"), { autoAlpha: 0 }, { autoAlpha: 1, stagger: dur(c, 0.06), duration: dur(c, 0.25) }, at(c, 1.0));
  tl.to(all(c, "[data-label]"), { autoAlpha: 0, duration: dur(c, 0.25) }, at(c, 3.8));
  // drop to ground level and look down the time axis
  cam(c, { pos: [0, 6, 70], look: [0, 4, -60] }, 4.0, 0.4, "power2.in");
  tl.to(S.twin, { alpha: 0, duration: dur(c, 0.3), ease: "none" }, at(c, 4.1));
}

export function timeline(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  setMorph(c, "wLattice", 0);
  tl.set(S.universe, { wText: 0, stage: 0, redReveal: 1, reveal: 1, highlight: 0, highlightClass: -1, zoom: 0, size: 1, burst: 0 }, at(c, 0));
  camSet(c, { pos: [0, 20, 150], look: [0, 0, 0] }, 0);
  tl.to(S.universe, { alpha: 1, duration: dur(c, 0.5) }, at(c, 0));
  morph(c, "wTimeline", 0.3, 1.0, "power2.inOut");
  chars(c, el(c, "head"), 0.25, 1.25, { stagger: 0.02 });
  reveal(c, el(c, "axis"), { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.3) }, at(c, 1.05));
  reveal(c, all(c, "[data-lane]"), { autoAlpha: 0, x: -8 }, { autoAlpha: 1, x: 0, stagger: dur(c, 0.05), duration: dur(c, 0.25) }, at(c, 1.15));
  reveal(c, el(c, "readout"), { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.25) }, at(c, 1.3));
  tl.to(S.universe, { zoom: 1, duration: dur(c, 2.7), ease: "power1.inOut" }, at(c, 1.3));
  line(c, el(c, "caption"), 2.4, 3.9, { y: 8 });
  tl.to([el(c, "axis"), el(c, "readout"), ...all(c, "[data-lane]")], { autoAlpha: 0, duration: dur(c, 0.3) }, at(c, 4.25));
  tl.to(S.universe, { alpha: 0, duration: dur(c, 0.45) }, at(c, 4.3));
}
