"use client";

// A recorded frame on the light table: the image (or a plate saying why it is
// missing), the grease-pencil circle around the match, and an OSD stamp.
import { useEffect, useRef, useState } from "react";
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
}

const VIEW_W = 1600;
const VIEW_H = 900;

export function Frame({ src, alt, bbox, markId, animate = false, osd, className }: FrameProps) {
  const [failed, setFailed] = useState(false);
  const [prevSrc, setPrevSrc] = useState(src);
  const path = useRef<SVGPathElement>(null);
  if (src !== prevSrc) {
    setPrevSrc(src);
    setFailed(false);
  }

  // circle geometry in a 16:9 view box, padded around the box
  const box = bbox
    ? (() => {
        const [x1, y1, x2, y2] = bbox;
        const padX = (x2 - x1) * 0.28 + 0.02;
        const padY = (y2 - y1) * 0.22 + 0.025;
        return { x: (x1 - padX) * VIEW_W, y: (y1 - padY) * VIEW_H, w: (x2 - x1 + 2 * padX) * VIEW_W, h: (y2 - y1 + 2 * padY) * VIEW_H };
      })()
    : null;

  useEffect(() => {
    const el = path.current;
    if (!el || !box || !markId) return;
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
    // box values are derived from bbox; re-run when the mark changes
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [markId, animate, bbox?.join(",")]);

  return (
    <figure className={`lt-frame${className ? ` ${className}` : ""}`}>
      {failed ? (
        <div className="lt-frame-missing" role="img" aria-label={`${alt}. The frame is not available yet.`}>
          <span>Frame not rendered yet</span>
        </div>
      ) : (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={apiUrl(src)} alt={alt} loading="lazy" decoding="async" onError={() => setFailed(true)} />
      )}
      {box && markId ? (
        <svg className="lt-mark-layer" viewBox={`0 0 ${VIEW_W} ${VIEW_H}`} preserveAspectRatio="none" aria-hidden="true">
          <g transform={`translate(${box.x} ${box.y})`}>
            <path ref={path} />
          </g>
        </svg>
      ) : null}
      {osd ? <figcaption className="lt-osd">{osd}</figcaption> : null}
    </figure>
  );
}
