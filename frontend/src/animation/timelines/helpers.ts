// Shared vocabulary for act timelines. Every builder receives a Ctx and works
// in "local" desktop units; ctx.at() maps them onto the master timeline so the
// same choreography compresses cleanly on tablets and phones.
import type { gsap as GSAP } from "gsap";

type Timeline = ReturnType<typeof GSAP.timeline>;
import { S, type Vec3Like } from "@/animation/sceneState";
import type { ActId } from "@/lib/config/animation";

export interface Ctx {
  gsap: typeof GSAP;
  tl: Timeline;
  root: HTMLElement;
  mobile: boolean;
  tablet: boolean;
  reduced: boolean;
  flights: boolean;
  /** act start on the master timeline */
  t0: number;
  /** act length on the master timeline */
  len: number;
  /** desktop-unit length of this act (local time domain) */
  base: number;
  act: ActId;
}

/** Local time (desktop units within the act) to master time. */
export const at = (c: Ctx, local: number): number => c.t0 + (local / c.base) * c.len;
export const dur = (c: Ctx, local: number): number => (local / c.base) * c.len;

export function el<T extends HTMLElement = HTMLElement>(c: Ctx, name: string): T {
  const node = c.root.querySelector<T>(`[data-act="${c.act}"] [data-el="${name}"]`);
  if (!node) throw new Error(`missing [data-el="${name}"] in act ${c.act}`);
  return node;
}

export function all<T extends HTMLElement = HTMLElement>(c: Ctx, selector: string): T[] {
  return Array.from(c.root.querySelectorAll<T>(`[data-act="${c.act}"] ${selector}`));
}

export function layer(c: Ctx): HTMLElement {
  const node = c.root.querySelector<HTMLElement>(`[data-act="${c.act}"]`);
  if (!node) throw new Error(`missing act layer ${c.act}`);
  return node;
}

/** Show the act's DOM layer for its span on the master timeline. */
export function actLayer(c: Ctx, opts: { first?: boolean; last?: boolean } = {}): void {
  const node = layer(c);
  if (opts.first) c.gsap.set(node, { autoAlpha: 1 });
  else c.tl.fromTo(node, { autoAlpha: 0 }, { autoAlpha: 1, duration: dur(c, 0.12), ease: "none" }, c.t0);
  if (!opts.last) c.tl.to(node, { autoAlpha: 0, duration: dur(c, 0.12), ease: "none" }, c.t0 + c.len - dur(c, 0.12));
}

/** Fade text in (rising a few pixels), hold, fade out. Times are local. */
export function line(c: Ctx, target: Element | Element[], inAt: number, outAt: number | null, opts: { y?: number; stretch?: boolean } = {}): void {
  const y = opts.y ?? 18;
  c.tl.fromTo(
    target,
    { autoAlpha: 0, y, ...(opts.stretch ? { "--wdth": 75 } : {}) },
    { autoAlpha: 1, y: 0, ...(opts.stretch ? { "--wdth": 125 } : {}), duration: dur(c, 0.38), ease: "power3.out" },
    at(c, inAt),
  );
  if (outAt !== null) {
    c.tl.to(target, { autoAlpha: 0, y: -y, ...(opts.stretch ? { "--wdth": 100 } : {}), duration: dur(c, 0.3), ease: "power2.in" }, at(c, outAt));
  }
}

type Vars = Record<string, unknown>;

/**
 * fromTo that puts every target in its from-state at build time. A staggered
 * fromTo only pre-renders its first target, so the rest would flash visible
 * before their turn.
 */
export function reveal(c: Ctx, targets: object | object[], from: Vars, to: Vars, position: number): void {
  c.gsap.set(targets, from);
  c.tl.fromTo(targets, from, { ...to, immediateRender: false }, position);
}

/** Character reveal for display headlines (spans rendered by <Split />). */
export function chars(c: Ctx, root: Element, inAt: number, outAt: number | null, opts: { stagger?: number } = {}): void {
  const nodes = Array.from(root.querySelectorAll(".ch"));
  reveal(
    c,
    nodes,
    { yPercent: 105, autoAlpha: 0 },
    { yPercent: 0, autoAlpha: 1, duration: dur(c, 0.42), ease: "power3.out", stagger: { each: dur(c, opts.stagger ?? 0.018) } },
    at(c, inAt),
  );
  if (outAt !== null) {
    c.tl.to(nodes, { yPercent: -105, autoAlpha: 0, duration: dur(c, 0.3), ease: "power2.in", stagger: { each: dur(c, 0.008) } }, at(c, outAt));
  }
}

export interface Pose {
  pos: [number, number, number];
  look: [number, number, number];
}

/**
 * Move the camera rig. Without flights (reduced motion, LOW tier) the move
 * becomes a short dip through black and a cut, which keeps the story intact
 * without vestibular motion.
 */
export function cam(c: Ctx, pose: Pose, inAt: number, length: number, ease = "sine.inOut"): void {
  const to = (target: Vec3Like, v: [number, number, number]) => ({ x: v[0], y: v[1], z: v[2] });
  if (c.flights) {
    c.tl.to(S.cam.pos, { ...to(S.cam.pos, pose.pos), duration: dur(c, length), ease }, at(c, inAt));
    c.tl.to(S.cam.look, { ...to(S.cam.look, pose.look), duration: dur(c, length), ease }, at(c, inAt));
  } else {
    const mid = at(c, inAt + length / 2);
    c.tl.to(S, { blackout: 1, duration: dur(c, 0.08), ease: "none" }, mid - dur(c, 0.08));
    c.tl.set(S.cam.pos, to(S.cam.pos, pose.pos), mid);
    c.tl.set(S.cam.look, to(S.cam.look, pose.look), mid);
    c.tl.to(S, { blackout: 0, duration: dur(c, 0.08), ease: "none" }, mid);
  }
}

/** Instant camera placement, for cuts hidden behind black or a full-screen frame. */
export function camSet(c: Ctx, pose: Pose, localAt: number): void {
  c.tl.set(S.cam.pos, { x: pose.pos[0], y: pose.pos[1], z: pose.pos[2] }, at(c, localAt));
  c.tl.set(S.cam.look, { x: pose.look[0], y: pose.look[1], z: pose.look[2] }, at(c, localAt));
}

type MorphKey = "wLattice" | "wNetwork" | "wFunnel" | "wTimeline" | "wRecon";
const MORPHS: MorphKey[] = ["wLattice", "wNetwork", "wFunnel", "wTimeline", "wRecon"];

/** Morph the universe to one target; the weights always sum to one. */
export function morph(c: Ctx, target: MorphKey, inAt: number, length: number, ease = "power1.inOut"): void {
  const vars: Record<string, number> = {};
  for (const k of MORPHS) vars[k] = k === target ? 1 : 0;
  c.tl.to(S.universe, { ...vars, duration: dur(c, length), ease }, at(c, inAt));
}

export function setMorph(c: Ctx, target: MorphKey, localAt: number): void {
  const vars: Record<string, number> = {};
  for (const k of MORPHS) vars[k] = k === target ? 1 : 0;
  c.tl.set(S.universe, vars, at(c, localAt));
}

type SleeveKey = "wNetwork" | "wGrid" | "wDiagonal" | "wCurve" | "wSolo";
const SLEEVES: SleeveKey[] = ["wNetwork", "wGrid", "wDiagonal", "wCurve", "wSolo"];

export function sleeves(c: Ctx, target: SleeveKey, inAt: number, length: number, ease = "power2.inOut"): void {
  const vars: Record<string, number> = {};
  for (const k of SLEEVES) vars[k] = k === target ? 1 : 0;
  c.tl.to(S.sleeves, { ...vars, duration: dur(c, length), ease }, at(c, inAt));
}

/** Fire-and-forget cue (sound, DOM state) when the scroll passes a moment, in either direction. */
export function cue(c: Ctx, localAt: number, fn: () => void): void {
  c.tl.call(fn, undefined, at(c, localAt));
}
