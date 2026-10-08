// The clarify-once interaction. The visitor can pick a camera and mark the
// gate themselves; if they just keep scrolling, the system demonstrates the
// same gesture. Either way the answer becomes a remembered place.
import { getStroke } from "perfect-freehand";
import { S } from "@/animation/sceneState";
import { sfx } from "@/lib/audio/engine";
import { strokePath } from "@/lib/draw";
import { seenBy, type FeedLibrary } from "./feeds";

const VIEW_W = 480;
const VIEW_H = 270;

export class ClarifyController {
  private choice: string | null = null;
  private byUser = false;
  private userPoints: number[][] = [];
  private drawing = false;
  private lastDraw = -1;
  private readonly picks: HTMLButtonElement[];
  private readonly inspect: HTMLCanvasElement | null;
  private readonly stroke: SVGPathElement | null;
  private readonly svg: SVGSVGElement | null;
  private readonly camLabels: HTMLElement[];
  private readonly hint: HTMLElement | null;
  private readonly cleanup: (() => void)[] = [];
  private dots = new Map<string, HTMLElement>();

  constructor(
    private readonly root: HTMLElement,
    private readonly feeds: FeedLibrary,
  ) {
    const layer = root.querySelector<HTMLElement>('[data-act="memory"]');
    this.picks = Array.from(layer?.querySelectorAll<HTMLButtonElement>("[data-pick]") ?? []);
    this.inspect = layer?.querySelector<HTMLCanvasElement>("[data-inspect-feed]") ?? null;
    this.svg = layer?.querySelector<SVGSVGElement>("[data-draw]") ?? null;
    this.stroke = layer?.querySelector<SVGPathElement>("[data-stroke]") ?? null;
    this.camLabels = Array.from(root.querySelectorAll<HTMLElement>("[data-chosen-cam]"));
    this.hint = layer?.querySelector<HTMLElement>("[data-draw-hint]") ?? null;

    const dots = new Map(Array.from(layer?.querySelectorAll<HTMLElement>("[data-dot]") ?? []).map((d) => [d.dataset.dot ?? "", d]));
    this.dots = dots;
    for (const b of this.picks) {
      const id = b.dataset.pick ?? "";
      const onClick = () => this.choose(id, true);
      const onEnter = () => dots.get(id)?.setAttribute("data-hot", "1");
      const onLeave = () => dots.get(id)?.removeAttribute("data-hot");
      b.addEventListener("click", onClick);
      b.addEventListener("pointerenter", onEnter);
      b.addEventListener("pointerleave", onLeave);
      b.addEventListener("focus", onEnter);
      b.addEventListener("blur", onLeave);
      this.cleanup.push(() => {
        b.removeEventListener("click", onClick);
        b.removeEventListener("pointerenter", onEnter);
        b.removeEventListener("pointerleave", onLeave);
        b.removeEventListener("focus", onEnter);
        b.removeEventListener("blur", onLeave);
      });
    }
    if (this.svg) {
      const svg = this.svg;
      const toLocal = (e: PointerEvent): number[] => {
        const r = svg.getBoundingClientRect();
        return [((e.clientX - r.left) / r.width) * VIEW_W, ((e.clientY - r.top) / r.height) * VIEW_H, e.pressure || 0.5];
      };
      const down = (e: PointerEvent) => {
        if (!this.choice) return;
        this.drawing = true;
        this.byUser = true;
        this.userPoints = [toLocal(e)];
        svg.setPointerCapture(e.pointerId);
        this.render();
      };
      const move = (e: PointerEvent) => {
        if (!this.drawing) return;
        this.userPoints.push(toLocal(e));
        this.render();
      };
      const up = () => {
        if (!this.drawing) return;
        this.drawing = false;
        sfx("confirm");
        this.hint?.setAttribute("data-done", "1");
      };
      svg.addEventListener("pointerdown", down);
      svg.addEventListener("pointermove", move);
      svg.addEventListener("pointerup", up);
      svg.addEventListener("pointercancel", up);
      this.cleanup.push(() => {
        svg.removeEventListener("pointerdown", down);
        svg.removeEventListener("pointermove", move);
        svg.removeEventListener("pointerup", up);
        svg.removeEventListener("pointercancel", up);
      });
    }
  }

  private choose(id: string, user: boolean): void {
    if (user) {
      this.byUser = true;
      sfx("tick");
    }
    this.choice = id || null;
    this.userPoints = [];
    for (const b of this.picks) b.setAttribute("aria-pressed", String(b.dataset.pick === id));
    this.dots.forEach((d, key) => d.toggleAttribute("data-chosen", key === id));
    for (const l of this.camLabels) l.textContent = id ? id : "CAM_04";
    if (this.inspect && id) this.feeds.paintOne(this.inspect, id, "none");
    this.root.querySelector('[data-act="memory"]')?.setAttribute("data-chosen", id ? "1" : "0");
    this.lastDraw = -1;
    this.render();
  }

  /** Auto gesture: the gate line, projected from the twin into the chosen view. */
  private autoPoints(): number[][] {
    const cam = this.choice ?? "CAM_04";
    const isGate = cam === "CAM_04";
    const a = isGate ? seenBy(cam, [-6.6, 0.4, 34.2]) : [0.18, 0.74];
    const b = isGate ? seenBy(cam, [6.6, 0.4, 34.2]) : [0.84, 0.66];
    const pts: number[][] = [];
    for (let i = 0; i <= 24; i++) {
      const t = i / 24;
      const wob = Math.sin(t * 9.0) * 1.2;
      pts.push([(a[0] + (b[0] - a[0]) * t) * VIEW_W, (a[1] + (b[1] - a[1]) * t) * VIEW_H + wob, 0.4 + 0.5 * Math.sin(t * Math.PI)]);
    }
    return pts;
  }

  private render(): void {
    if (!this.stroke) return;
    const pts = this.byUser && this.userPoints.length ? this.userPoints : this.autoPoints();
    const progress = this.byUser && this.userPoints.length ? 1 : S.memory.draw;
    const n = Math.max(0, Math.floor(pts.length * progress));
    this.stroke.setAttribute("d", n < 2 ? "" : strokePath(getStroke(pts.slice(0, n), { size: 7, thinning: 0.55, smoothing: 0.5, streamline: 0.35, last: progress >= 1 })));
  }

  /** Frame hook: auto-choose once the scroll reaches the gesture, undo it if they scroll back. */
  frame(): void {
    const autoOn = S.memory.auto >= 0.5;
    if (!this.byUser) {
      if (autoOn && this.choice !== "CAM_04") this.choose("CAM_04", false);
      else if (!autoOn && this.choice) this.choose("", false);
    }
    const d = Math.round(S.memory.draw * 40);
    if (d !== this.lastDraw) {
      this.lastDraw = d;
      this.render();
    }
  }

  dispose(): void {
    this.cleanup.forEach((f) => f());
  }
}
