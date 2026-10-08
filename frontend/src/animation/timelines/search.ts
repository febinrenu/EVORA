// SEARCH: ask anything -> the query parses itself -> the universe is filtered
// to one point -> that point opens into the evidence frame.
import { S } from "@/animation/sceneState";
import { HOP_U, twinCam } from "@/lib/data/site";
import { FUNNEL, PARSE_SLOTS, QUERY } from "@/lib/data/story";
import { actLayer, all, at, camSet, chars, cue, dur, el, line, morph, setMorph, type Ctx, reveal } from "./helpers";
import { sfx } from "@/lib/audio/engine";

/** Scroll-scrubbed typing: one character per step, written only when it changes. */
export function typeInto(c: Ctx, target: HTMLElement, text: string, inAt: number, length: number): void {
  const proxy = { n: 0 };
  let shown = -1;
  c.tl.fromTo(
    proxy,
    { n: 0 },
    {
      n: text.length,
      duration: dur(c, length),
      ease: "none",
      immediateRender: false,
      onUpdate: () => {
        const k = Math.round(proxy.n);
        if (k === shown) return;
        if (k > shown && shown >= 0) sfx("key");
        shown = k;
        target.textContent = text.slice(0, k);
      },
    },
    at(c, inAt),
  );
}

function centre(r: DOMRect): { x: number; y: number } {
  return { x: r.left + r.width / 2, y: r.top + r.height / 2 };
}

export function ask(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  const head = el(c, "head");
  chars(c, head, 0.12, null, { stagger: 0.03 });
  tl.to(head, { y: () => -window.innerHeight * (c.mobile ? 0.25 : 0.27), scale: c.mobile ? 0.55 : 0.4, duration: dur(c, 0.45), ease: "power3.inOut" }, at(c, 0.95));

  const bar = el(c, "bar");
  tl.fromTo(bar, { autoAlpha: 0, clipPath: "inset(0 50% 0 50%)" }, { autoAlpha: 1, clipPath: "inset(0 0% 0 0%)", duration: dur(c, 0.4), ease: "power3.inOut" }, at(c, 1.1));
  typeInto(c, el(c, "typed"), QUERY, 1.45, 1.0);

  const status = all(c, "[data-status]");
  status.forEach((s, i) => {
    const t = [1.2, 2.45, 3.15][i] ?? 3.2;
    reveal(c, s, { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.12) }, at(c, t));
    if (i < status.length - 1) tl.to(s, { autoAlpha: 0, duration: dur(c, 0.1) }, at(c, [2.45, 3.15][i]));
  });

  // the typed line is swapped for word spans in the identical layout, so words can be lifted out
  tl.to(el(c, "typedLine"), { autoAlpha: 0, duration: 0.001 }, at(c, 2.5));
  reveal(c, el(c, "words"), { autoAlpha: 0 }, { autoAlpha: 1, duration: 0.001 }, at(c, 2.5));

  const words = all(c, "[data-word]");
  PARSE_SLOTS.forEach((slot, k) => {
    const t = 2.6 + k * 0.16;
    const card = el(c, `slot-${slot.key}`);
    const ghost = card.querySelector<HTMLElement>("[data-ghost]");
    const value = card.querySelector<HTMLElement>("[data-value]");
    const source = words.filter((w) => (slot.words as readonly string[]).includes(w.dataset.word ?? ""));
    if (!ghost || !value) return;
    tl.to(source, { color: "var(--bone-48)", duration: dur(c, 0.1) }, at(c, t));
    tl.fromTo(card, { autoAlpha: 0, clipPath: "inset(0 0 100% 0)" }, { autoAlpha: 1, clipPath: "inset(0 0 0% 0)", duration: dur(c, 0.3), ease: "power2.out" }, at(c, t + 0.08));
    // FLIP: the word itself flies from the query into its slot
    tl.fromTo(
      ghost,
      {
        autoAlpha: 1,
        x: () => {
          const from = centre(source.length ? unionRect(source) : ghost.getBoundingClientRect());
          return from.x - centre((ghost.parentElement ?? ghost).getBoundingClientRect()).x;
        },
        y: () => {
          const from = centre(source.length ? unionRect(source) : ghost.getBoundingClientRect());
          return from.y - centre((ghost.parentElement ?? ghost).getBoundingClientRect()).y;
        },
      },
      { x: 0, y: 0, duration: dur(c, 0.36), ease: "power3.inOut", immediateRender: false },
      at(c, t),
    );
    tl.to(ghost, { autoAlpha: 0, duration: dur(c, 0.12) }, at(c, t + 0.4));
    reveal(c, value, { autoAlpha: 0, y: 6 }, { autoAlpha: 1, y: 0, duration: dur(c, 0.2) }, at(c, t + 0.4));
    cue(c, t + 0.36, () => sfx("tick"));
  });
  reveal(c, all(c, "[data-json] span"), { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.05), stagger: dur(c, 0.035) }, at(c, 3.0));
}

function unionRect(nodes: HTMLElement[]): DOMRect {
  const rs = nodes.map((n) => n.getBoundingClientRect());
  const l = Math.min(...rs.map((r) => r.left));
  const t = Math.min(...rs.map((r) => r.top));
  const r = Math.max(...rs.map((r) => r.right));
  const b = Math.max(...rs.map((r) => r.bottom));
  return new DOMRect(l, t, r - l, b - t);
}

/** Roll an odometer to `value`: each column turns to its digit, right to left. */
function odometer(c: Ctx, strips: HTMLElement[], seps: HTMLElement[], value: number, localAt: number, length = 0.42): void {
  const digits = strips.length;
  strips.forEach((strip, j) => {
    const place = digits - 1 - j; // strips render left to right
    const d = Math.floor(value / 10 ** place) % 10;
    const visible = place === 0 || value >= 10 ** place;
    c.tl.to(strip, { yPercent: -d * 10, duration: dur(c, length), ease: "power3.inOut" }, at(c, localAt) + dur(c, 0.03) * place);
    c.tl.to(strip.parentElement, { autoAlpha: visible ? 1 : 0, width: visible ? "0.62em" : "0em", duration: dur(c, length * 0.6), ease: "power2.inOut" }, at(c, localAt));
  });
  seps.forEach((s) => {
    const place = Number(s.dataset.sep);
    c.tl.to(s, { autoAlpha: value >= 10 ** place ? 1 : 0, width: value >= 10 ** place ? "0.28em" : "0em", duration: dur(c, length * 0.6) }, at(c, localAt));
  });
}

export function search(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  // the universe returns, unfiltered
  setMorph(c, "wLattice", 0);
  tl.set(S.universe, { wText: 0, stage: 0, redReveal: 0, reveal: 1, highlight: 0, highlightClass: -1, burst: 0, size: 1 }, at(c, 0));
  camSet(c, { pos: [0, 30, 150], look: [0, 0, 0] }, 0);
  tl.set(S.cam, { sway: 1 }, at(c, 0));
  tl.to(S.universe, { alpha: 1, duration: dur(c, 0.7) }, at(c, 0));
  tl.to(S.cam.pos, { y: 8, z: 70, duration: dur(c, 4.0), ease: "sine.inOut" }, at(c, 0.6));
  morph(c, "wFunnel", 0.8, 0.9);

  const chips = all(c, "[data-chip]");
  reveal(c, chips, { autoAlpha: 0, y: -6 }, { autoAlpha: 1, y: 0, stagger: dur(c, 0.06), duration: dur(c, 0.25) }, at(c, 0.3));

  const strips = all(c, "[data-strip]");
  const seps = all(c, "[data-sep]");
  const labels = all(c, "[data-flabel]");
  reveal(c, el(c, "odo"), { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.3) }, at(c, 0.7));
  odometer(c, strips, seps, FUNNEL[0].value, 0.4, 0.01);
  reveal(c, labels[0], { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.2) }, at(c, 0.8));

  const stages = [1.6, 2.3, 3.0, 3.7];
  const applied = [[0], [1], [2, 3], [4]];
  stages.forEach((t, i) => {
    tl.to(S.universe, { stage: i + 1, duration: dur(c, 0.5), ease: "power2.inOut" }, at(c, t));
    odometer(c, strips, seps, FUNNEL[i + 1].value, t + 0.08);
    tl.to(labels[i], { autoAlpha: 0, duration: dur(c, 0.12) }, at(c, t + 0.05));
    reveal(c, labels[i + 1], { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.2) }, at(c, t + 0.3));
    for (const k of applied[i]) tl.to(chips[k], { "--applied": 1, duration: dur(c, 0.2) }, at(c, t));
    cue(c, t + 0.1, () => sfx("pulse"));
  });
  tl.to(S.universe, { redReveal: 1, duration: dur(c, 0.45) }, at(c, 2.35));

  // everything collapses to a single point, which flares before the frame opens
  tl.to([el(c, "odo"), labels[labels.length - 1], el(c, "chain")], { autoAlpha: 0, duration: dur(c, 0.3) }, at(c, 4.35));
  tl.to(S.universe, { size: 3.2, duration: dur(c, 0.35), ease: "power2.in" }, at(c, 4.5));
  tl.to(S.universe, { alpha: 0, duration: dur(c, 0.3), ease: "power2.in" }, at(c, 4.9));
}

export function evidence(c: Ctx): void {
  const { tl } = c;
  actLayer(c);
  const pose = twinCam("CAM_04");
  tl.set(S.twin, { alpha: 1, look: 1, cctv: 1, iris: 0, car: HOP_U.CAM_04 - 0.012, trace: 0, traceAlpha: 0, beacons: 0, gate: 0, ghosts: 0, recon: 0 }, at(c, 0));
  tl.set(S.cam, { follow: 0, sway: 0, fov: 42 }, at(c, 0));
  camSet(c, { pos: [...pose.pos], look: [...pose.look] }, 0);
  tl.to(S.twin, { iris: 1, duration: dur(c, 0.75), ease: "power2.inOut" }, at(c, 0.05));
  tl.to(S.twin, { car: HOP_U.CAM_04 + 0.004, duration: dur(c, 4.2), ease: "none" }, at(c, 0));
  cue(c, 0.1, () => sfx("open"));

  line(c, el(c, "osd"), 0.55, 3.95, { y: 0 });
  tl.to(S.evidence, { box: 1, duration: dur(c, 0.6), ease: "power2.inOut" }, at(c, 0.8));
  reveal(c, el(c, "boxLabel"), { autoAlpha: 0, x: -8 }, { autoAlpha: 1, x: 0, duration: dur(c, 0.25) }, at(c, 1.3));

  const why = el(c, "why");
  reveal(c, why, { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.2) }, at(c, 1.55));
  chars(c, el(c, "whyHead"), 1.6, null, { stagger: 0.012 });
  all(c, "[data-reason]").forEach((row, k) => {
    const t = 1.85 + k * 0.22;
    reveal(c, row, { autoAlpha: 0, y: 8 }, { autoAlpha: 1, y: 0, duration: dur(c, 0.22) }, at(c, t));
    const bar = row.querySelector("[data-bar]");
    if (bar) reveal(c, bar, { scaleX: 0 }, { scaleX: Number((bar as HTMLElement).dataset.bar), duration: dur(c, 0.4), ease: "power3.out" }, at(c, t + 0.06));
  });
  tl.to(why, { autoAlpha: 0, x: 20, duration: dur(c, 0.3) }, at(c, 3.55));
  tl.to(S.evidence, { box: 0, duration: dur(c, 0.3) }, at(c, 3.8));
  tl.to(el(c, "boxLabel"), { autoAlpha: 0, duration: dur(c, 0.2) }, at(c, 3.8));
}
