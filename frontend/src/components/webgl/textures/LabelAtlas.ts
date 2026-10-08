// On-screen display for every sleeve, drawn into one small canvas texture.
// The clock column is redrawn once a second, and only while sleeves are visible.
import { CanvasTexture, LinearFilter, SRGBColorSpace } from "three";
import type { NetworkNode } from "@/lib/data/site";

export const LABEL = { cols: 4, rows: 6, cellW: 256, cellH: 42 } as const;

export class LabelAtlas {
  readonly texture: CanvasTexture;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private lastSecond = -1;

  constructor(
    private readonly nodes: NetworkNode[],
    private readonly mono: string,
  ) {
    this.canvas = document.createElement("canvas");
    this.canvas.width = LABEL.cols * LABEL.cellW;
    this.canvas.height = LABEL.rows * LABEL.cellH;
    const ctx = this.canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas unavailable");
    this.ctx = ctx;
    this.texture = new CanvasTexture(this.canvas);
    this.texture.colorSpace = SRGBColorSpace;
    this.texture.minFilter = LinearFilter;
    this.texture.generateMipmaps = false;
    this.draw(9 * 3600 + 14 * 60 + 2);
  }

  /** `clock` is seconds since midnight shown on every OSD. */
  tick(clock: number): void {
    const s = Math.floor(clock);
    if (s === this.lastSecond) return;
    this.draw(s);
  }

  private draw(clock: number): void {
    this.lastSecond = Math.floor(clock);
    const { ctx } = this;
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.textBaseline = "middle";
    this.nodes.forEach((n, i) => {
      const x = (i % LABEL.cols) * LABEL.cellW;
      const y = Math.floor(i / LABEL.cols) * LABEL.cellH;
      // a slightly different clock per camera: real NVRs never agree to the frame
      const t = clock + ((i * 7) % 3) - 1;
      const hh = String(Math.floor(t / 3600) % 24).padStart(2, "0");
      const mm = String(Math.floor((t % 3600) / 60)).padStart(2, "0");
      const ss = String(Math.floor(t % 60)).padStart(2, "0");
      ctx.fillStyle = "rgba(6,8,10,0.55)";
      ctx.fillRect(x + 2, y + 4, LABEL.cellW - 4, LABEL.cellH - 8);
      ctx.fillStyle = "#e9eef1";
      ctx.font = `600 15px ${this.mono}`;
      ctx.fillText(n.id.replace("_", " "), x + 10, y + LABEL.cellH / 2);
      ctx.fillStyle = "rgba(220,227,231,0.66)";
      ctx.font = `400 13px ${this.mono}`;
      ctx.fillText(n.label.toUpperCase().slice(0, 13), x + 78, y + LABEL.cellH / 2);
      ctx.textAlign = "right";
      ctx.fillText(`${hh}:${mm}:${ss}`, x + LABEL.cellW - 10, y + LABEL.cellH / 2);
      ctx.textAlign = "left";
    });
    this.texture.needsUpdate = true;
  }

  dispose(): void {
    this.texture.dispose();
  }
}
