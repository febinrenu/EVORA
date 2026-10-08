// DISCOVER: opening title -> event universe -> camera network -> camera wall.
import { S } from "@/animation/sceneState";
import { COUNT_BEATS } from "@/lib/data/story";
import { actLayer, all, at, cam, chars, dur, el, line, morph, sleeves, type Ctx, reveal } from "./helpers";

export function opening(c: Ctx): void {
  const { tl } = c;
  actLayer(c, { first: true });
  const title = el(c, "title");
  const glyphs = Array.from(title.querySelectorAll<HTMLElement>(".ch"));
  const n = glyphs.length;

  tl.to(el(c, "cue"), { autoAlpha: 0, duration: dur(c, 0.15) }, at(c, 0.02));
  tl.to(el(c, "meta"), { autoAlpha: 0, y: -10, duration: dur(c, 0.3) }, at(c, 0.12));
  // the words begin separating before they let go
  tl.to(glyphs, { x: (i: number) => (i - n / 2) * (c.mobile ? 2.4 : 5.5), duration: dur(c, 0.7), ease: "power1.in" }, at(c, 0.05));
  tl.to(title, { "--wdth": 118, duration: dur(c, 0.7), ease: "power1.in" }, at(c, 0.05));
  // DOM type hands over to particles sitting exactly on the same glyphs
  tl.to(S.universe, { alpha: 1, duration: dur(c, 0.3), ease: "none" }, at(c, 0.5));
  tl.to(title, { autoAlpha: 0, duration: dur(c, 0.3), ease: "none" }, at(c, 0.58));
  // letters become particles, particles multiply, the camera backs away
  tl.to(S.universe, { wText: 0, duration: dur(c, 1.35), ease: "power2.inOut" }, at(c, 0.9));
  tl.to(S.universe, { burst: 1, duration: dur(c, 0.55), ease: "power2.out" }, at(c, 0.9));
  tl.to(S.universe, { burst: 0, duration: dur(c, 1.1), ease: "power2.inOut" }, at(c, 1.45));
  tl.to(S.universe, { reveal: 1, duration: dur(c, 1.4), ease: "power1.in" }, at(c, 1.0));
  cam(c, { pos: [0, 22, 168], look: [0, 0, 0] }, 0.9, 2.3, "power2.inOut");
  line(c, el(c, "fragment"), 2.2, 2.95);
}

export function universe(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  cam(c, { pos: [-86, 9, 52], look: [10, 0, 0] }, 0, 1.6);
  cam(c, { pos: [-12, 2, 24], look: [60, -4, -10] }, 1.6, 2.6);
  cam(c, { pos: [64, 16, 40], look: [-10, 0, -8] }, 4.2, 2.0);
  cam(c, { pos: [0, 70, 150], look: [0, 12, 0] }, 6.2, 0.8, "power2.inOut");

  const labels = all(c, "[data-count]");
  const beat = (i: number) => 0.5 + i * 1.25;
  COUNT_BEATS.forEach((b, i) => {
    const t = beat(i);
    if (i === 0) {
      tl.set(S.universe, { textA: 1, textB: 1, textMix: 0 }, at(c, t - 0.1));
      tl.to(S.universe, { wText: 1, duration: dur(c, 0.55), ease: "power2.inOut" }, at(c, t));
    } else {
      // digits flow into the next number instead of counting
      tl.set(S.universe, { textA: i, textB: i + 1, textMix: 0 }, at(c, t) - 0.0001);
      tl.to(S.universe, { textMix: 1, duration: dur(c, 0.55), ease: "power2.inOut" }, at(c, t));
      tl.to(labels[i - 1], { autoAlpha: 0, y: -8, duration: dur(c, 0.18) }, at(c, t));
    }
    tl.fromTo(labels[i], { autoAlpha: 0, y: 10 }, { autoAlpha: 1, y: 0, duration: dur(c, 0.3) }, at(c, t + 0.3));
    // the field answers the number: its class brightens, everything else steps back
    tl.set(S.universe, { highlightClass: b.cls }, at(c, t));
    reveal(c, S.universe, { highlight: 0 }, { highlight: b.cls >= 0 ? 1 : 0, duration: dur(c, 0.5) }, at(c, t + 0.1));
  });
  tl.to(labels[labels.length - 1], { autoAlpha: 0, duration: dur(c, 0.2) }, at(c, 6.2));
  tl.to(S.universe, { wText: 0, highlight: 0, duration: dur(c, 0.6), ease: "power2.inOut" }, at(c, 6.3));
}

export function network(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  morph(c, "wNetwork", 0, 1.3);
  tl.to(S.links, { alpha: 1, duration: dur(c, 0.5) }, at(c, 0.5));
  tl.to(S.links, { draw: 1, duration: dur(c, 1.8), ease: "power1.inOut" }, at(c, 0.6));
  tl.to(S.sleeves, { alpha: 1, duration: dur(c, 0.7) }, at(c, 0.9));
  tl.to(S.universe, { alpha: 0.5, duration: dur(c, 0.6) }, at(c, 1.0));

  cam(c, { pos: [-74, 30, 74], look: [-10, 14, 0] }, 0, 1.4);
  cam(c, { pos: [-8, 20, 6], look: [50, 14, -40] }, 1.4, 1.6);
  cam(c, { pos: [58, 26, -30], look: [0, 16, 0] }, 3.0, 1.3);
  cam(c, { pos: [0, 18, 92], look: [0, 15, 0] }, 4.3, 0.7, "power2.inOut");

  line(c, el(c, "l1"), 1.3, 2.5);
  chars(c, el(c, "l2"), 2.8, 4.1);
  line(c, el(c, "legend"), 1.6, 4.2, { y: 6 });
  tl.to(S.universe, { alpha: 0, duration: dur(c, 0.6) }, at(c, 4.4));
}

export function wall(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  tl.to(S.cam, { sway: 0, duration: dur(c, 0.6) }, at(c, 0));
  sleeves(c, "wGrid", 0, 1.0);
  tl.to(S.links, { alpha: 0, duration: dur(c, 0.6) }, at(c, 0));
  sleeves(c, "wDiagonal", 1.1, 0.7);
  sleeves(c, "wCurve", 1.9, 0.7);
  sleeves(c, "wSolo", 2.7, 0.6, "power3.inOut");
  line(c, el(c, "caption"), 0.45, 1.7);
  line(c, el(c, "count"), 1.9, 2.6, { y: 6 });
  line(c, el(c, "osd"), 3.05, 3.5, { y: 0 });
  // the dominant feed switches off like a monitor
  tl.to(S.sleeves, { collapse: 1, duration: dur(c, 0.3), ease: "power3.in" }, at(c, 3.55));
  tl.to(S.sleeves, { alpha: 0, duration: dur(c, 0.12), ease: "none" }, at(c, 3.85));
}
