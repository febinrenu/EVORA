// Performance tier detection plus a frame-time governor.
// Detection is a heuristic at boot; the governor corrects it with real frame
// times and only ever steps down (stepping back up mid-scroll would oscillate).
import { GOVERNOR, TIERS, type Tier, type TierSpec } from "@/lib/config/tiers";

export interface DeviceProfile {
  tier: Tier;
  spec: TierSpec;
  webgl2: boolean;
  touch: boolean;
  reducedMotion: boolean;
  dpr: number;
  renderer: string;
  reason: string;
}

const SOFTWARE_GPU = /swiftshader|llvmpipe|software|basic render|microsoft basic/i;
const WEAK_GPU = /intel\(r\) (hd|uhd) graphics [1-6]\d{2}|mali-[gt][0-7]\d|adreno \(tm\) [1-5]\d{2}|powervr|apple gpu/i;
const STRONG_GPU = /nvidia|geforce|rtx|radeon rx|radeon pro|apple m[1-9]|arc a\d/i;

function readRenderer(): { webgl2: boolean; renderer: string } {
  try {
    const canvas = document.createElement("canvas");
    const gl = canvas.getContext("webgl2");
    if (!gl) return { webgl2: false, renderer: "none" };
    const ext = gl.getExtension("WEBGL_debug_renderer_info");
    const renderer = ext ? String(gl.getParameter(ext.UNMASKED_RENDERER_WEBGL)) : String(gl.getParameter(gl.RENDERER));
    gl.getExtension("WEBGL_lose_context")?.loseContext();
    return { webgl2: true, renderer };
  } catch {
    return { webgl2: false, renderer: "error" };
  }
}

export function detectProfile(): DeviceProfile {
  const url = new URL(window.location.href);
  const forced = url.searchParams.get("tier") as Tier | null;
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const touch = window.matchMedia("(pointer: coarse)").matches;
  const dpr = window.devicePixelRatio || 1;
  const { webgl2, renderer } = readRenderer();
  const cores = navigator.hardwareConcurrency || 4;
  const memory = (navigator as Navigator & { deviceMemory?: number }).deviceMemory ?? 8;
  const small = Math.min(window.innerWidth, window.innerHeight) < 600;

  let tier: Tier = "high";
  let reason = "default";
  if (forced && forced in TIERS) {
    tier = forced;
    reason = "forced by ?tier";
  } else if (!webgl2 || SOFTWARE_GPU.test(renderer)) {
    tier = "low";
    reason = "no hardware WebGL2";
  } else if (reducedMotion) {
    tier = "low";
    reason = "prefers-reduced-motion";
  } else if (touch || small || cores <= 4 || memory <= 4 || WEAK_GPU.test(renderer)) {
    tier = STRONG_GPU.test(renderer) && !small ? "high" : "medium";
    reason = "mobile or modest hardware";
  } else if (!STRONG_GPU.test(renderer) && cores < 8) {
    tier = "medium";
    reason = "integrated GPU";
  }
  return { tier, spec: TIERS[tier], webgl2, touch, reducedMotion, dpr, renderer, reason };
}

/** Watches frame times and steps quality down when the budget is exceeded. */
export class Governor {
  private times: number[] = [];
  private grace: number = GOVERNOR.graceFrames;
  private dprIndex = 0;
  private particleIndex = 0;
  private overFrames = 0;

  constructor(
    private readonly dprCap: number,
    private readonly onDpr: (dpr: number) => void,
    private readonly onParticles: (share: number) => void,
  ) {
    const steps = GOVERNOR.dprSteps;
    while (this.dprIndex < steps.length - 1 && steps[this.dprIndex] > dprCap) this.dprIndex++;
  }

  /** Call after a scene change so shader warm-up frames are not held against the device. */
  warmup(): void {
    this.grace = GOVERNOR.graceFrames;
    this.times.length = 0;
    this.overFrames = 0;
  }

  sample(frameMs: number): void {
    if (document.hidden) return;
    if (this.grace > 0) {
      this.grace--;
      return;
    }
    // Ignore huge gaps (tab switch, debugger) — they are not render cost.
    if (frameMs > 250) return;
    this.times.push(frameMs);
    if (this.times.length > GOVERNOR.windowFrames) this.times.shift();
    if (this.times.length < GOVERNOR.windowFrames) return;
    const sorted = [...this.times].sort((a, b) => a - b);
    const p75 = sorted[Math.floor(sorted.length * 0.75)];
    this.overFrames = p75 > GOVERNOR.budgetMs ? this.overFrames + 1 : 0;
    if (this.overFrames < GOVERNOR.windowFrames) return;
    this.overFrames = 0;
    this.times.length = 0;
    this.grace = GOVERNOR.graceFrames;
    // cheapest lever first: fewer particles costs nothing to apply; a DPR
    // change reallocates the drawing buffer, so the engine defers it
    if (this.particleIndex < GOVERNOR.particleSteps.length - 1) {
      this.particleIndex++;
      this.onParticles(GOVERNOR.particleSteps[this.particleIndex]);
    } else if (this.dprIndex < GOVERNOR.dprSteps.length - 1) {
      this.dprIndex++;
      this.onDpr(Math.min(this.dprCap, GOVERNOR.dprSteps[this.dprIndex]));
    }
  }
}
