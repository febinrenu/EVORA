// Wording and numbers for the product UI (PLAN §10.6): plain verbs, sentence
// case, every timestamp shown twice (clock and offset into the camera's file).
import type { CameraInfo, Evidence } from "@/lib/api/client";

const two = (n: number) => String(Math.floor(n)).padStart(2, "0");

// Times are shown in the site's zone (the footage's own clock), not the
// browser's, so they match the answer text. Falls back to the browser zone.
let siteZone: string | undefined;
let clockFmt = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false });
let dayFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short" });

export function setSiteZone(tz: string | undefined): void {
  if (!tz || tz === siteZone) return;
  try {
    clockFmt = new Intl.DateTimeFormat("en-GB", { hour: "2-digit", minute: "2-digit", second: "2-digit", hour12: false, timeZone: tz });
    dayFmt = new Intl.DateTimeFormat(undefined, { day: "numeric", month: "short", timeZone: tz });
    siteZone = tz;
  } catch {
    /* unknown zone name: keep the browser zone */
  }
}

/** Wall-clock time of an epoch second on the site clock. */
export function clock(epoch: number, withMs = false): string {
  const base = clockFmt.format(new Date(epoch * 1000));
  return withMs ? `${base}.${String(Math.floor((((epoch % 1) + 1) % 1) * 1000)).padStart(3, "0")}` : base;
}

/** Site-clock wall time as "YYYY-MM-DDTHH:MM:SS", the value a datetime-local input takes. */
export function wallInput(epoch: number): string {
  const f = new Intl.DateTimeFormat("en-CA", {
    year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit",
    hourCycle: "h23", timeZone: siteZone,
  });
  const p = Object.fromEntries(f.formatToParts(new Date(epoch * 1000)).map((x) => [x.type, x.value]));
  return `${p.year}-${p.month}-${p.day}T${p.hour}:${p.minute}:${p.second}`;
}

/** Inverse of wallInput: a site-clock wall time back to epoch seconds, or null if it does not parse. */
export function fromWall(value: string): number | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?$/.exec(value);
  if (!m) return null;
  const asUtc = Date.UTC(+m[1], +m[2] - 1, +m[3], +m[4], +m[5], +(m[6] ?? 0)) / 1000;
  // the zone's offset at that moment, refined once so a day of daylight change lands right
  let t = asUtc;
  for (let i = 0; i < 2; i++) {
    const shown = fromParts(wallInput(t));
    t += asUtc - shown;
  }
  return t;
}

function fromParts(v: string): number {
  const [d, h] = v.split("T");
  const [y, mo, da] = d.split("-").map(Number);
  const [hh, mi, se] = h.split(":").map(Number);
  return Date.UTC(y, mo - 1, da, hh, mi, se) / 1000;
}

export function day(epoch: number): string {
  return dayFmt.format(new Date(epoch * 1000));
}

/** "12:03" or "1:02:07" into the file. */
export function offset(seconds: number): string {
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const r = Math.floor(s % 60);
  return h ? `${h}:${two(m)}:${two(r)}` : `${m}:${two(r)}`;
}

export function fileName(cam: CameraInfo | undefined): string {
  if (!cam) return "the recording";
  const parts = cam.source_uri.split(/[\\/]/);
  return parts[parts.length - 1] || cam.name;
}

/** weak < 0.4 <= fair < 0.65 <= strong, and the number is always shown with it */
export function confidenceWord(score: number): "weak" | "fair" | "strong" {
  return score < 0.4 ? "weak" : score < 0.65 ? "fair" : "strong";
}

export function confidence(score: number): string {
  const w = confidenceWord(score);
  return `${w[0].toUpperCase()}${w.slice(1)} match ${score.toFixed(2)}`;
}

export function span(ev: Evidence): string {
  const a = clock(ev.t_start);
  const b = clock(ev.t_end);
  return a === b ? a : `${a} to ${b}`;
}

/** The headline of an answer sheet, from the verdict and count. */
export function verdictLine(verdict: string, count: number | null | undefined, n: number): string {
  switch (verdict) {
    case "yes":
      return n > 1 ? `Yes, ${n === 2 ? "twice" : `${n} times`}.` : "Yes.";
    case "no":
      return "No.";
    case "count":
      return `${count ?? n}.`;
    case "found":
      return n === 1 ? "Found it." : `Found ${n}.`;
    case "not_found":
      return "Not found.";
    case "partial":
      return "Partly.";
    default:
      return "Answered.";
  }
}

export function seconds(ms: number | undefined): string {
  if (ms === undefined) return "";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
