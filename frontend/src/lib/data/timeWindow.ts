import { DAY } from "@/lib/data/story";

/**
 * Visible window of the timeline act for a zoom value 0..1. Shared by the
 * particle shader uniforms and the DOM tick layer so both always agree.
 * The span shrinks logarithmically so every scroll step reveals the next unit
 * of time (hours, minutes, seconds, milliseconds).
 */
export function timeWindow(zoom: number): { focus: number; span: number } {
  const z = Math.min(1, Math.max(0, zoom));
  const from = Math.log(DAY.span * 1.05);
  const to = Math.log(DAY.minSpan);
  const span = Math.exp(from + (to - from) * z);
  const settle = Math.min(1, z * 3.2);
  const focus = DAY.span / 2 + (DAY.focusT - DAY.span / 2) * (settle * settle * (3 - 2 * settle));
  return { focus, span };
}
