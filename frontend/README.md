# evora frontend

**Live: https://evora-ivory.vercel.app** — the story and the results page. The product at `/app` needs the local API, so
there it only shows the interface; run `start.bat` for the working system.

Next.js (App Router) app, built as a static export, with three routes:

- `/` — the cinematic story of the system (scroll experience, WebGL).
- `/app` — the operator product on the light table (PLAN §10): cameras, questions and answers with circled evidence, the
  site plan, known places, watches and alerts, the timeline.
- `/report` — the evaluation results, baked in at build time from `../eval/reports/report.json` and refreshed from
  `/api/report` when the API is there.

From the repo root on Windows, `start.bat` does everything (installs uv and packages, builds this UI into `dist/` when sources changed, starts the API that serves it on http://127.0.0.1:8700). `start.bat dev` runs the API on fixtures plus `npm run dev` on :5173; `start.bat setup` adds the perception stack and model weights.

```
npm install
npm run footage   # optional: builds public/footage/ from EPFL sequences (needs ffmpeg); procedural feeds otherwise
npm run dev       # http://localhost:5173 (the API allows this origin)
npm run check     # tsc --noEmit + eslint
npm run build     # static export into dist/, served by the API
npm run deploy    # publish to Vercel (preview); npm run deploy -- --prod for the live site
```

`npm run deploy` builds here and uploads the static output, because the report file it bakes in lives outside this
folder. It needs the Vercel CLI logged in, and `npm run footage` first for the camera wall.

## How the experience is put together

- One loop: the GSAP ticker drives Lenis, the master ScrollTrigger timeline and the renderer (`src/animation/director.ts`).
- One mutable scene state (`src/animation/sceneState.ts`) that the timeline writes and the renderer reads; no React state changes while scrolling.
- One canvas and one renderer (`src/components/webgl/Engine.ts`). Every particle scene is a morph of a single 420k-point mesh whose particles are synthetic events (time, camera, class, funnel level).
- The digital twin (`DigitalTwin.ts`) is procedural; CCTV views are the same scene through a lens post pass.
- Act choreography lives in `src/animation/timelines/*`, all timings in local units mapped by `src/lib/config/animation.ts`.
- Story data (cameras, timestamps, scores) is in `src/lib/data/story.ts` and `site.ts`; the query plan shown on screen is typed against `contracts/ts`.

## Performance tiers

Detected at boot (`src/lib/perf/tier.ts`): `high`, `medium`, `low` set particle count, DPR cap, shader detail and camera flights. A governor steps quality down if frame times stay over budget. Shaders are compiled and drawn once during the opening calibration so nothing compiles mid-scroll.

Profiling switches: `?tier=high|medium|low` forces a tier, `?gov=0` pins quality.

## Footage

`npm run footage` reads short segments of nine EPFL multi-camera pedestrian sequences (research use) and tiles them into one 3x3 atlas video, so every feed on the page costs one decoder. The output is git-ignored, and it is
published with the hosted site (the owner has permission to show it).
