"""P2.3 spike: detector + tracker throughput on this machine.

Usage: python scripts/spike_detect.py <video> [--weights yolo26n.pt yolo26s.pt] [--device cuda] [--imgsz 640] [--max-frames 600]

Prints decode fps, detect+track fps, tracks per second of video, and the speed relative to real time.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(REPO))


def run(video: Path, weights: str, device: str, imgsz: int, max_frames: int) -> dict:
    from evora.perception.decode import probe_video, read_frames
    from evora.perception.detect import weights_path
    from ultralytics import YOLO

    info = probe_video(video)
    t = time.perf_counter()
    frames = []
    for f in read_frames(video):
        frames.append(f)
        if len(frames) >= max_frames:
            break
    decode_s = time.perf_counter() - t

    model = YOLO(weights_path(weights))
    names = {v: k for k, v in model.names.items()}
    wanted = ("person", "car", "bus", "truck", "motorcycle", "bicycle", "backpack", "handbag", "suitcase", "umbrella")
    classes = [names[c] for c in wanted if c in names]
    half = device == "cuda"
    quant = 16 if half else None
    opts = {"classes": classes, "device": device, "quantize": quant, "imgsz": imgsz, "verbose": False}
    model.track(frames[0].image, persist=False, **opts)  # warm-up
    ids: set[int] = set()
    n_boxes = 0
    t = time.perf_counter()
    for f in frames:
        r = model.track(f.image, persist=True, tracker="bytetrack.yaml", **opts)[0]
        if r.boxes is not None and r.boxes.id is not None:
            ids.update(r.boxes.id.int().tolist())
            n_boxes += len(r.boxes)
    infer_s = time.perf_counter() - t
    video_s = frames[-1].pts_s - frames[0].pts_s if len(frames) > 1 else 0.0
    return {
        "weights": weights, "device": device, "imgsz": imgsz, "frames": len(frames),
        "src": f"{info.width}x{info.height}@{info.fps}",
        "decode_fps": len(frames) / decode_s, "track_fps": len(frames) / infer_s,
        "tracks": len(ids), "boxes": n_boxes, "video_s": video_s,
        "video_s_per_wall_s": video_s / infer_s if infer_s else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video", type=Path)
    ap.add_argument("--weights", nargs="*", default=["yolo26n.pt", "yolo26s.pt"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-frames", type=int, default=600)
    args = ap.parse_args()
    for w in args.weights:
        r = run(args.video, w, args.device, args.imgsz, args.max_frames)
        print(
            f"{r['weights']:<12} {r['device']} imgsz={r['imgsz']} src={r['src']} frames={r['frames']}  "
            f"decode={r['decode_fps']:.0f} fps  detect+track={r['track_fps']:.0f} fps  "
            f"{r['video_s_per_wall_s']:.1f} video-s/s  tracks={r['tracks']} boxes={r['boxes']}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
