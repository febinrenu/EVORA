"use client";

// Clip with the tracked box drawn over it, synced to playback from the track's
// points (≈4 Hz, interpolated). J/K/L and ←/→ work while it has focus.
import { useEffect, useRef, useState } from "react";
import { apiUrl, endpoints, type Evidence, type TrackPoint } from "@/lib/api/client";

const PRE_ROLL = 3;

export function ClipPlayer({ ev, onClose }: { ev: Evidence; onClose: () => void }) {
  const video = useRef<HTMLVideoElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [points, setPoints] = useState<TrackPoint[]>([]);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    if (!ev.track_id) return;
    let live = true;
    endpoints
      .track(ev.track_id)
      .then((t) => live && setPoints(t.points ?? []))
      .catch(() => undefined);
    return () => {
      live = false;
    };
  }, [ev.track_id]);

  useEffect(() => {
    const v = video.current;
    const c = canvas.current;
    if (!v || !c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;
    let handle = 0;
    // clips start PRE_ROLL seconds before t_start (media service)
    const clipStart = ev.t_start - PRE_ROLL;
    const draw = () => {
      const w = v.clientWidth;
      const h = v.clientHeight;
      if (c.width !== w * devicePixelRatio) {
        c.width = w * devicePixelRatio;
        c.height = h * devicePixelRatio;
      }
      ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
      ctx.clearRect(0, 0, w, h);
      const t = clipStart + v.currentTime;
      let box: [number, number, number, number] | null = null;
      if (points.length) {
        let j = points.findIndex((p) => p.t >= t);
        if (j === -1) j = points.length - 1;
        const a = points[Math.max(0, j - 1)];
        const b = points[j];
        if (t >= points[0].t - 0.5 && t <= points[points.length - 1].t + 0.5) {
          const k = b.t === a.t ? 0 : Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t)));
          box = [a.x1 + (b.x1 - a.x1) * k, a.y1 + (b.y1 - a.y1) * k, a.x2 + (b.x2 - a.x2) * k, a.y2 + (b.y2 - a.y2) * k];
        }
      } else if (ev.bbox && Math.abs(t - ev.t_peak) < 1.5) {
        box = ev.bbox;
      }
      if (box) {
        ctx.strokeStyle = "#C8102E";
        ctx.lineWidth = 2;
        ctx.strokeRect(box[0] * w, box[1] * h, (box[2] - box[0]) * w, (box[3] - box[1]) * h);
      }
      handle = v.requestVideoFrameCallback ? v.requestVideoFrameCallback(draw) : requestAnimationFrame(draw);
    };
    draw();
    return () => {
      if (v.cancelVideoFrameCallback) v.cancelVideoFrameCallback(handle);
      cancelAnimationFrame(handle);
    };
  }, [points, ev.t_start, ev.t_peak, ev.bbox]);

  const onKey = (e: React.KeyboardEvent) => {
    const v = video.current;
    if (!v) return;
    const frame = 1 / 25;
    switch (e.key.toLowerCase()) {
      case "k":
        if (v.paused) void v.play();
        else v.pause();
        break;
      case "l":
        v.playbackRate = v.paused ? 1 : Math.min(4, v.playbackRate * 2);
        void v.play();
        break;
      case "j":
        v.currentTime = Math.max(0, v.currentTime - 2);
        break;
      case "arrowright":
        if (!v.paused) return;
        v.currentTime += frame;
        break;
      case "arrowleft":
        if (!v.paused) return;
        v.currentTime = Math.max(0, v.currentTime - frame);
        break;
      case "escape":
        onClose();
        break;
      default:
        return;
    }
    e.preventDefault();
  };

  if (failed)
    return (
      <div className="lt-frame lt-frame-missing is-hero" role="img" aria-label="The clip is not available yet">
        <span>Clip not rendered yet. The frame and timestamps above still hold.</span>
      </div>
    );

  return (
    <div className="lt-clip" onKeyDown={onKey}>
      <video ref={video} src={apiUrl(ev.clip_url)} controls autoPlay muted playsInline preload="metadata" onError={() => setFailed(true)} aria-label={`Clip from ${ev.camera_name}`} />
      <canvas ref={canvas} aria-hidden="true" />
    </div>
  );
}
