// Feed library: what each twin camera recorded at a moment, rendered once
// through the CCTV lens and kept as small canvases. DOM thumbnails paint from
// these with drawImage; the reconstruction cards use the render targets.
import { PerspectiveCamera, Vector3, type WebGLRenderTarget } from "three";
import { HOP_U, TWIN_CAMS, twinCam } from "@/lib/data/site";
import type { Engine } from "@/components/webgl/Engine";

/** car moment keys: where the sedan is when the frame was recorded */
export const CAR_AT = { none: null, "04": HOP_U.CAM_04, "07": HOP_U.CAM_07, "12": HOP_U.CAM_12 } as const;
export type CarKey = keyof typeof CAR_AT;

const JOBS: [string, CarKey][] = [
  ["CAM_04", "none"],
  ["CAM_07", "none"],
  ["CAM_12", "none"],
  ["CAM_18", "none"],
  ["CAM_02", "none"],
  ["CAM_09", "none"],
  ["CAM_04", "04"],
  ["CAM_07", "07"],
  ["CAM_12", "12"],
];

export class FeedLibrary {
  private readonly canvases = new Map<string, HTMLCanvasElement>();
  readonly targets = new Map<string, WebGLRenderTarget>();
  private started = false;
  private done = false;
  private readonly listeners = new Set<() => void>();

  constructor(private readonly engine: Engine) {}

  /** Render every feed once (shaders are already compiled), reading pixels back asynchronously. */
  async build(): Promise<void> {
    if (this.started) return;
    this.started = true;
    for (const [cam, car] of JOBS) {
      const rt = this.engine.snapshot(twinCam(cam), CAR_AT[car]);
      const canvas = document.createElement("canvas");
      await this.engine.readToCanvas(rt, canvas);
      const key = `${cam}:${car}`;
      this.canvases.set(key, canvas);
      this.targets.set(key, rt);
    }
    this.done = true;
    this.listeners.forEach((l) => l());
  }

  onReady(fn: () => void): void {
    if (this.done) fn();
    else this.listeners.add(fn);
  }

  /** Paint every canvas under `root` that asks for a feed via data-feed / data-car. */
  paint(root: ParentNode): void {
    root.querySelectorAll<HTMLCanvasElement>("canvas[data-feed]").forEach((c) => this.paintOne(c));
  }

  paintOne(target: HTMLCanvasElement, cam = target.dataset.feed ?? "", car: CarKey = (target.dataset.car as CarKey) ?? "none"): void {
    const src = this.canvases.get(`${cam}:${car}`) ?? this.canvases.get(`${cam}:none`);
    if (!src) return;
    if (target.width !== src.width) {
      target.width = src.width;
      target.height = src.height;
    }
    target.getContext("2d")?.drawImage(src, 0, 0);
    target.dataset.painted = "1";
  }

  dispose(): void {
    this.targets.forEach((t) => t.dispose());
    this.targets.clear();
  }
}

/** Normalised (0..1, y down) image position of a world point as seen by a twin camera. */
export function seenBy(camId: string, world: [number, number, number]): [number, number] {
  const pose = TWIN_CAMS.find((c) => c.id === camId) ?? twinCam("CAM_04");
  const cam = new PerspectiveCamera(48, 16 / 9, 0.1, 600);
  cam.position.set(...pose.pos);
  cam.lookAt(new Vector3(...pose.look));
  cam.updateMatrixWorld();
  const p = new Vector3(...world).project(cam);
  return [p.x * 0.5 + 0.5, -p.y * 0.5 + 0.5];
}
