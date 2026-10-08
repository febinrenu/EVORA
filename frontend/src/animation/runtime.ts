// Process-wide handles for the running experience, so hooks can reach the
// director and the shared frame loop without prop drilling or context churn.
import type { Director } from "./director";

type FrameFn = (dtMs: number) => void;

export const runtime = {
  director: null as Director | null,
  frames: new Set<FrameFn>(),
};

export function onFrame(fn: FrameFn): () => void {
  runtime.frames.add(fn);
  return () => {
    runtime.frames.delete(fn);
  };
}
