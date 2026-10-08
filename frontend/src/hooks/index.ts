"use client";

import { useEffect, useRef, useSyncExternalStore } from "react";
import type Lenis from "lenis";
import { onFrame, runtime } from "@/animation/runtime";
import { S } from "@/animation/sceneState";
import { ACTS, LENGTH_SCALE, type ActId } from "@/lib/config/animation";
import { detectProfile, type DeviceProfile } from "@/lib/perf/tier";

export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (cb) => {
      const m = window.matchMedia(query);
      m.addEventListener("change", cb);
      return () => m.removeEventListener("change", cb);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

export const useReducedMotion = (): boolean => useMediaQuery("(prefers-reduced-motion: reduce)");

let cachedProfile: DeviceProfile | null = null;
const noop = () => () => {};

/** Device tier, detected once on the client (null during server render). */
export function usePerformanceTier(): DeviceProfile | null {
  return useSyncExternalStore(
    noop,
    () => (cachedProfile ??= detectProfile()),
    () => null,
  );
}

export function useLenis(): Lenis | null {
  return runtime.director?.lenis ?? null;
}

/**
 * Run `fn(progress)` on the shared ticker while an act is on screen, with the
 * act's local progress 0..1. No React state is touched per frame.
 */
export function useScrollScene(id: ActId, fn: (progress: number) => void): void {
  const cb = useRef(fn);
  useEffect(() => {
    cb.current = fn;
  });
  useEffect(() => {
    const index = ACTS.findIndex((a) => a.id === id);
    return onFrame(() => {
      if (S.act !== index) return;
      const starts = runtime.director?.starts() ?? [];
      const scale = window.innerWidth < 640 ? LENGTH_SCALE.mobile : window.innerWidth < 1024 ? LENGTH_SCALE.tablet : LENGTH_SCALE.desktop;
      const total = ACTS.reduce((s, a) => s + a.length * scale, 0);
      const now = S.progress * total;
      const start = starts[index] ?? 0;
      const len = ACTS[index].length * scale;
      cb.current(Math.min(1, Math.max(0, (now - start) / len)));
    });
  }, [id]);
}
