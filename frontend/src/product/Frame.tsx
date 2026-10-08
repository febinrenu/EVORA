"use client";

// A recorded frame on the light table: the image (or a plate saying why it is
// missing), the grease-pencil circle around the match, and an OSD stamp.
// Boxes from the API are normalised to the full camera frame, so marks are
// drawn in a layer laid exactly over the drawn image, never over the figure:
// footage that is not 16:9 is letterboxed (contain) or cropped (cover).
import { useEffect, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { handEllipse, seedOf } from "@/lib/draw";
import { apiUrl } from "@/lib/api/client";

interface FrameProps {
  src: string;
  alt: string;
  bbox?: [number, number, number, number] | null;
  /** id that seeds the circle so it never changes shape between renders */
  markId?: string;
  /** draw the circle on mount (the one orchestrated moment, PLAN §10.5) */
  animate?: boolean;
  osd?: string;
  className?: string;
  /** contain shows the whole frame (the default when something is marked on it); cover fills the sleeve */
  fit?: "contain" | "cover";
  /** drawn over the image itself, in a box with the frame's own aspect (for drawing tools) */
  children?: ReactNode;
}

/** thumbnails are rendered on demand: a missing one is asked for again before giving up */
const RETRY_MS = [600, 1500, 3500];
const VIEW_W = 1600;

interface Rect {
  left: number;
  top: number;
  width: number;
  height: number;
}

export function Frame({ src, alt, bbox, markId, animate = false, osd, className, fit, children }: FrameProps) {
  const mode = fit ?? (bbox || children ? "contain" : "cover");
  const [attempt, setAttempt] = useState(0);
  const [loaded, setLoaded] = useState(false);
  const [failed, setFailed] = useState(false);
  const [natural, setNatural] = useState<{ w: number; h: number } | null>(null);
  const [rect, setRect] = useState<Rect | null>(null);
  const [prevSrc, setPrevSrc] = useState(src);
  const figure = useRef<HTMLElement>(null);
  const path = useRef<SVGPathElement>(null);
  if (src !== prevSrc) {
    setPrevSrc(src);
    setAttempt(0);
    setLoaded(false);
    setFailed(false);
  }

  // the image rectangle inside the figure, for the fit in use
  useLayoutEffect(() => {
    const el = figure.current;
    if (!el || !natural) return;
    const measure = () => {
      const fw = el.clientWidth;
      const fh = el.clientHeight;
      if (!fw || !fh) return;
      const s = mode === "contain" ? Math.min(fw / natural.w, fh / natural.h) : Math.max(fw / natural.w, fh / natural.h);
      const width = natural.w * s;
      const height = natural.h * s;
      setRect({ left: (fw - width) / 2, top: (fh - height) / 2, width, height });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [natural, mode]);

  const viewH = natural ? (VIEW_W * natural.h) / natural.w : VIEW_W * (9 / 16);

  // circle geometry in the frame's own view box, padded around the box
  const box = bbox
    ? (() => {
        const [x1, y1, x2, y2] = bbox;
        const padX = (x2 - x1) * 0.28 + 0.02;
        const padY = (y2 - y1) * 0.22 + 0.025;
        return { x: (x1 - padX) * VIEW_W, y: (y1 - padY) * viewH, w: (x2 - x1 + 2 * padX) * VIEW_W, h: (y2 - y1 + 2 * padY) * viewH };
      })()
    : null;
  const showMark = Boolean(box && markId && rect && loaded);

  useEffect(() => {
    const el = path.current;
    if (!el || !box || !markId || !showMark) return;
    const seed = seedOf(markId);
    const draw = (p: number) => el.setAttribute("d", handEllipse(box.w, box.h, seed, p, 9));
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (!animate || reduced) {
      draw(1);
      return;
    }
    // 600 ms ease-out, written straight to the path: no React renders while drawing
    let raf = 0;
    const t0 = performance.now();
    const step = (now: number) => {
      const t = Math.min(1, (now - t0) / 600);
      draw(1 - Math.pow(1 - t, 3));
      if (t < 1) raf = requestAnimationFrame(step);
    };
    raf = requestAnimationFrame(step);
    return () => cancelAnimationFrame(raf);
    // box values are derived from bbox and the frame's aspect; re-run when either changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markId, animate, bbox?.join(","), viewH, showMark]);

  // a failed load waits, then asks again with a fresh URL (the first answer may be cached)
  const [waiting, setWaiting] = useState(false);
  useEffect(() => {
    if (!waiting) return;
    const t = window.setTimeout(() => {
      setWaiting(false);
      setAttempt((a) => a + 1);
    }, RETRY_MS[attempt] ?? 0);
    return () => window.clearTimeout(t);
  }, [waiting, attempt]);

  const url = apiUrl(src);
  const shown = attempt ? `${url}${url.includes("?") ? "&" : "?"}_r=${attempt}` : url;
  const layer: React.CSSProperties | undefined = rect ? { left: rect.left, top: rect.top, width: rect.width, height: rect.height } : undefined;

  return (
    <figure ref={figure} className={`lt-frame is-${mode}${loaded ? "" : " is-loading"}${className ? ` ${className}` : ""}`}>
      {failed ? (
        <div className="lt-frame-missing" role="img" aria-label={`${alt}. The frame is not available.`}>
          <span>Frame not available</span>
        </div>
      ) : waiting ? null : (
        // eslint-disable-next-line @next/next/no-img-element
        <img
          key={shown}
          src={shown}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={(e) => {
            const img = e.currentTarget;
            setLoaded(true);
            if (img.naturalWidth && img.naturalHeight) setNatural({ w: img.naturalWidth, h: img.naturalHeight });
          }}
          onError={() => (attempt < RETRY_MS.length ? setWaiting(true) : setFailed(true))}
        />
      )}
      {showMark && box ? (
        <svg className="lt-mark-layer" style={layer} viewBox={`0 0 ${VIEW_W} ${viewH}`} preserveAspectRatio="none" aria-hidden="true">
          <g transform={`translate(${box.x} ${box.y})`}>
            <path ref={path} />
          </g>
        </svg>
      ) : null}
      {children ? (
        <div className="lt-frame-box" style={layer ?? { inset: 0 }}>
          {children}
        </div>
      ) : null}
      {osd ? <figcaption className="lt-osd">{osd}</figcaption> : null}
    </figure>
  );
}
