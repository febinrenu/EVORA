"use client";

// Boots the experience: detects the device tier, starts the single ticker,
// lazily loads the WebGL engine, and wires the DOM layers to it. React renders
// the markup once; everything that moves is driven outside React.
import { useCallback, useEffect, useRef } from "react";
import { createDirector, type Director } from "@/animation/director";
import { ClarifyController } from "@/animation/clarify";
import { FeedLibrary } from "@/animation/feeds";
import { evidenceBox, memoryLabels, reconLabels, timelineTicks } from "@/animation/projectors";
import { buildTextRows } from "@/animation/textRows";
import { onFrame, runtime } from "@/animation/runtime";
import { S } from "@/animation/sceneState";
import { ACTS, type ActId } from "@/lib/config/animation";
import { detectProfile } from "@/lib/perf/tier";
import type { Engine } from "@/components/webgl/Engine";
import { Network, Opening, Universe, Wall } from "./sections/Discover";
import { Ask, Evidence, SearchAct } from "./sections/Search";
import { TraceAct } from "./sections/Trace";
import { MemoryAct, MemoryMapAct } from "./sections/Memory";
import { Finale, ReconstructAct, TimelineAct } from "./sections/Reconstruct";
import { ChapterRail, Cursor, Nav } from "./ui/Chrome";

type FrameFn = (dt: number) => void;

function fontFamily(variable: string, fallback: string): string {
  const v = getComputedStyle(document.documentElement).getPropertyValue(variable).trim();
  return v || fallback;
}

export function Experience() {
  const rootRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const trackRef = useRef<HTMLDivElement>(null);
  const directorRef = useRef<Director | null>(null);
  const feedsRef = useRef<FeedLibrary | null>(null);
  const register = useCallback((fn: FrameFn) => onFrame(fn), []);
  const go = useCallback((id: ActId) => directorRef.current?.scrollToAct(id), []);
  const paint = useCallback((root: ParentNode) => feedsRef.current?.paint(root), []);
  const onDrawer = useCallback((open: boolean) => {
    const lenis = directorRef.current?.lenis;
    if (open) lenis?.stop();
    else lenis?.start();
  }, []);

  useEffect(() => {
    const root = rootRef.current;
    const canvas = canvasRef.current;
    const track = trackRef.current;
    if (!root || !canvas || !track) return;
    let disposed = false;
    let engine: Engine | null = null;
    let clarify: ClarifyController | null = null;
    let director: Director | null = null;
    let resizeTimer = 0;

    if ("scrollRestoration" in history) history.scrollRestoration = "manual";
    window.scrollTo(0, 0);

    const profile = detectProfile();
    root.dataset.tier = profile.tier;
    root.dataset.webgl = profile.webgl2 ? "1" : "0";
    const mobile = window.innerWidth < 640;
    const fonts = {
      display: fontFamily("--font-archivo", "sans-serif"),
      mono: fontFamily("--font-geist-mono", "monospace"),
    };

    const boot = async () => {
      await document.fonts.ready;
      if (disposed) return;
      if (profile.webgl2) {
        const { Engine } = await import("@/components/webgl/Engine");
        if (disposed) return;
        engine = new Engine({ canvas, spec: profile.spec, mobile, fonts });
        engine.setSize(window.innerWidth, window.innerHeight);
        engine.buildText(buildTextRows(root, fonts.display));
        // `?debug` exposes the live state for profiling scripts
        if (new URLSearchParams(window.location.search).has("debug")) Object.assign(window, { __evora: { S, engine } });
        for (const hook of [evidenceBox(root), memoryLabels(root), reconLabels(root), timelineTicks(root)]) engine.addHook(hook);
        const feeds = new FeedLibrary(engine);
        feedsRef.current = feeds;
        clarify = new ClarifyController(root, feeds);
      }
      director = createDirector({
        root,
        track,
        flights: profile.spec.flights,
        frame: (dt) => {
          engine?.frame(dt);
          clarify?.frame();
          runtime.frames.forEach((f) => f(dt));
        },
        onAct: (i) => root.style.setProperty("--act", String(i)),
      });
      directorRef.current = director;
      runtime.director = director;

      // Calibration: every shader and every feed snapshot is prepared while the
      // title is on screen, so nothing compiles mid-scroll. Scrolling unlocks
      // when it is done (or after a safety timeout on very slow devices).
      if (engine) {
        director.lenis?.stop();
        const timeout = new Promise<void>((resolve) => window.setTimeout(resolve, 6000));
        const prepare = async () => {
          if (!engine) return;
          await engine.precompile();
          const feeds = feedsRef.current;
          if (!feeds || disposed) return;
          await feeds.build();
          if (disposed || !engine?.twin) return;
          feeds.paint(root);
          const t = feeds.targets;
          engine.twin.setCardTextures(["CAM_04:04", "CAM_07:07", "CAM_12:12", "CAM_18:none"].map((k) => t.get(k)?.texture).filter((x) => x !== undefined));
        };
        await Promise.race([prepare().catch(() => undefined), timeout]);
        if (disposed) return;
        director.lenis?.start();
      }
      root.dataset.ready = "1";
    };
    void boot();

    const onResize = () => {
      window.clearTimeout(resizeTimer);
      resizeTimer = window.setTimeout(() => {
        if (!engine || disposed) return;
        engine.setSize(window.innerWidth, window.innerHeight);
        engine.buildText(buildTextRows(root, fonts.display));
      }, 180);
    };
    window.addEventListener("resize", onResize);

    return () => {
      disposed = true;
      window.clearTimeout(resizeTimer);
      window.removeEventListener("resize", onResize);
      director?.destroy();
      clarify?.dispose();
      feedsRef.current?.dispose();
      feedsRef.current = null;
      engine?.dispose();
      directorRef.current = null;
      runtime.director = null;
    };
  }, []);

  return (
    <div className="exp" ref={rootRef} data-ready="0">
      <canvas className="stage" ref={canvasRef} aria-hidden="true" />
      <div className="grain" aria-hidden="true" />
      <nav className="skip" aria-label="Chapters">
        <ul>
          {ACTS.map((a) => (
            <li key={a.id}>
              <button type="button" onClick={() => go(a.id)}>
                {a.title}
              </button>
            </li>
          ))}
        </ul>
      </nav>
      <Nav go={go} />
      <main className="overlay" id="top">
        <Opening />
        <Universe />
        <Network />
        <Wall />
        <Ask />
        <SearchAct />
        <Evidence />
        <TraceAct />
        <MemoryAct />
        <MemoryMapAct onDrawer={onDrawer} paint={paint} />
        <TimelineAct />
        <ReconstructAct />
        <Finale />
      </main>
      <ChapterRail register={register} />
      <Cursor register={register} />
      <div className="track" ref={trackRef} aria-hidden="true" />
    </div>
  );
}
