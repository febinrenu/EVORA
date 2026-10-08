// Builds the typography the particles form: row 0 is the opening title,
// measured from the DOM so the dots land exactly on the real glyphs; rows 1..5
// are the counts, laid out to fill the frame.
import { COUNT_BEATS } from "@/lib/data/story";
import { formatInt } from "@/lib/math";
import type { TextLine, TextRowSpec } from "@/components/webgl/TextTargets";

let measure: CanvasRenderingContext2D | null = null;

function metricsAt100(text: string, family: string): { width: number; ascent: number } {
  measure ??= document.createElement("canvas").getContext("2d");
  if (!measure) return { width: text.length * 70, ascent: 72 };
  measure.font = `800 100px ${family}`;
  (measure as CanvasRenderingContext2D & { fontStretch?: string }).fontStretch = "expanded";
  const m = measure.measureText(text);
  return { width: m.width, ascent: m.actualBoundingBoxAscent };
}

export function buildTextRows(root: HTMLElement, family: string): TextRowSpec[] {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const title = root.querySelector<HTMLElement>('[data-act="opening"] [data-el="title"]');
  const lines: TextLine[] = [];
  if (title) {
    const style = getComputedStyle(title);
    const fontSize = parseFloat(style.fontSize);
    const spacing = parseFloat(style.letterSpacing) || 0;
    const upper = style.textTransform === "uppercase";
    title.querySelectorAll<HTMLElement>("[data-line]").forEach((ln) => {
      // measure the visible glyph run, not the screen-reader copy; a zero-size
      // probe on the baseline tells us exactly where the DOM sets the glyphs
      const run = ln.querySelector<HTMLElement>(".split") ?? ln;
      const r = run.getBoundingClientRect();
      const probe = document.createElement("span");
      probe.style.cssText = "display:inline-block;width:0;height:0;vertical-align:baseline";
      run.appendChild(probe);
      const baseline = probe.getBoundingClientRect().top;
      probe.remove();
      const text = ln.dataset.line ?? "";
      lines.push({ text: upper ? text.toUpperCase() : text, cx: r.left + r.width / 2, cy: baseline, fontSize, letterSpacing: spacing, baseline: true });
    });
  }
  const rows: TextRowSpec[] = [{ lines }];
  const mobile = W < 640;
  const labels = root.querySelectorAll<HTMLElement>('[data-act="universe"] [data-count]');
  COUNT_BEATS.forEach((beat, i) => {
    const text = formatInt(beat.value);
    const target = W * (mobile ? 0.9 : 0.78);
    const m = metricsAt100(text, family);
    const size = Math.min((100 * target) / m.width, H * (mobile ? 0.3 : 0.4));
    const cy = H * 0.44;
    rows.push({ lines: [{ text, cx: W / 2, cy, fontSize: size, letterSpacing: -size * 0.02 }] });
    // the label sits just under the particle digits, whatever their size
    const label = labels[i];
    if (label) label.style.top = `${cy + (m.ascent * size) / 200 + Math.max(18, H * 0.035)}px`;
  });
  return rows;
}
