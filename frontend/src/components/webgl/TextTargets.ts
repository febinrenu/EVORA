// Samples typography into particle targets. Each row of the target texture is
// one piece of text (the title, then each count); each texel is one particle's
// position in screen fractions (-0.5..0.5, y up). The vertex shader projects
// those onto a camera-attached plane, so the dots land exactly on the DOM type
// they replace.
import { DataTexture, FloatType, NearestFilter, RGBAFormat } from "three";
import { mulberry32 } from "@/lib/math";

export const TEXT_ROWS = 6;
const TEX_W = 4096;
const SAMPLE_SCALE = 0.5;

export interface TextLine {
  text: string;
  /** centre in CSS px */
  cx: number;
  cy: number;
  fontSize: number;
  letterSpacing: number;
  /** when true, cy is the alphabetic baseline instead of the cap centre */
  baseline?: boolean;
}

export interface TextRowSpec {
  lines: TextLine[];
}

export class TextTargets {
  readonly texture: DataTexture;
  readonly texWidth = TEX_W;
  private readonly data: Float32Array;
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;

  constructor(
    private readonly count: number,
    private readonly fontFamily: string,
  ) {
    const height = Math.ceil((count * TEXT_ROWS) / TEX_W);
    this.data = new Float32Array(TEX_W * height * 4);
    this.texture = new DataTexture(this.data, TEX_W, height, RGBAFormat, FloatType);
    this.texture.minFilter = NearestFilter;
    this.texture.magFilter = NearestFilter;
    this.texture.generateMipmaps = false;
    this.canvas = document.createElement("canvas");
    const ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    if (!ctx) throw new Error("2d canvas unavailable");
    this.ctx = ctx;
  }

  /** Re-sample every row for the current viewport. Called at boot and after resize. */
  build(rows: TextRowSpec[], viewW: number, viewH: number): void {
    const w = Math.max(2, Math.round(viewW * SAMPLE_SCALE));
    const h = Math.max(2, Math.round(viewH * SAMPLE_SCALE));
    this.canvas.width = w;
    this.canvas.height = h;
    rows.slice(0, TEXT_ROWS).forEach((row, r) => this.sampleRow(row, r, w, h));
    this.texture.needsUpdate = true;
  }

  private sampleRow(row: TextRowSpec, r: number, w: number, h: number): void {
    const { ctx } = this;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = "#fff";
    ctx.textAlign = "center";
    ctx.textBaseline = "alphabetic";
    for (const line of row.lines) {
      const size = line.fontSize * SAMPLE_SCALE;
      ctx.font = `800 ${size}px ${this.fontFamily}`;
      // Width axis: canvas exposes it as fontStretch where supported.
      (ctx as CanvasRenderingContext2D & { fontStretch?: string }).fontStretch = "expanded";
      (ctx as CanvasRenderingContext2D & { letterSpacing?: string }).letterSpacing = `${line.letterSpacing * SAMPLE_SCALE}px`;
      const m = ctx.measureText(line.text);
      const capCentre = (m.actualBoundingBoxAscent - m.actualBoundingBoxDescent) / 2;
      ctx.fillText(line.text, line.cx * SAMPLE_SCALE, line.cy * SAMPLE_SCALE + (line.baseline ? 0 : capCentre));
    }
    const pixels = ctx.getImageData(0, 0, w, h).data;
    const filled: number[] = [];
    // stride 1 on small canvases keeps thin strokes; alpha threshold keeps edges crisp
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (pixels[(y * w + x) * 4 + 3] > 140) filled.push(x, y);
      }
    }
    const rand = mulberry32(911 + r * 131);
    const base = r * this.count;
    const nFilled = filled.length / 2;
    for (let i = 0; i < this.count; i++) {
      const o = (base + i) * 4;
      if (nFilled === 0) {
        this.data[o + 3] = 0;
        continue;
      }
      const k = Math.floor(rand() * nFilled) * 2;
      const px = filled[k] + rand();
      const py = filled[k + 1] + rand();
      this.data[o] = px / w - 0.5;
      this.data[o + 1] = 0.5 - py / h;
      this.data[o + 2] = rand();
      this.data[o + 3] = 1;
    }
  }

  dispose(): void {
    this.texture.dispose();
  }
}
