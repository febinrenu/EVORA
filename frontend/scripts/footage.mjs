#!/usr/bin/env node
// Builds the camera-wall footage from EPFL multi-camera pedestrian sequences
// (research use). Only a short segment of each source is read over HTTP, so no
// full download is needed. Output is git-ignored and regenerated on demand:
//
//   npm run footage            # builds public/footage/{atlas.mp4,poster.jpg,manifest.json}
//   npm run footage -- --force # rebuild even if the atlas exists
//
// The site falls back to procedural feeds when the atlas is missing.
import { spawn } from "node:child_process";
import { existsSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { join, dirname } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const OUT = join(ROOT, "public", "footage");
const TMP = join(OUT, ".cells");

const BASE = "https://documents.epfl.ch/groups/c/cv";
// Nine views across three sites; `ss` skips the empty opening of each sequence.
const SOURCES = [
  { id: "terrace-c0", url: `${BASE}/cvlab-pom-video3/www/terrace1-c0.avi`, ss: 34 },
  { id: "terrace-c1", url: `${BASE}/cvlab-pom-video3/www/terrace1-c1.avi`, ss: 34 },
  { id: "terrace-c2", url: `${BASE}/cvlab-pom-video3/www/terrace1-c2.avi`, ss: 34 },
  { id: "terrace-c3", url: `${BASE}/cvlab-pom-video3/www/terrace1-c3.avi`, ss: 34 },
  { id: "passageway-c0", url: `${BASE}/cvlab-pom-video2/www/passageway1-c0.avi`, ss: 20 },
  { id: "passageway-c1", url: `${BASE}/cvlab-pom-video2/www/passageway1-c1.avi`, ss: 20 },
  { id: "passageway-c2", url: `${BASE}/cvlab-pom-video2/www/passageway1-c2.avi`, ss: 20 },
  { id: "lab-c0", url: `${BASE}/cvlab-pom-video1/www/6p-c0.avi`, ss: 30 },
  { id: "lab-c1", url: `${BASE}/cvlab-pom-video1/www/6p-c1.avi`, ss: 30 },
];

const CELL_W = 640;
const CELL_H = 360;
const SECONDS = 12;
const FPS = 25;

function run(args) {
  return new Promise((resolve, reject) => {
    const p = spawn("ffmpeg", ["-hide_banner", "-loglevel", "error", "-y", ...args], { stdio: "inherit" });
    p.on("error", reject);
    p.on("exit", (code) => (code === 0 ? resolve() : reject(new Error(`ffmpeg exited ${code}`))));
  });
}

async function main() {
  const force = process.argv.includes("--force");
  const atlas = join(OUT, "atlas.mp4");
  if (existsSync(atlas) && !force) {
    console.log("footage: atlas exists, use --force to rebuild");
    return;
  }
  mkdirSync(TMP, { recursive: true });

  // 1) Normalise each view: crop 5:4 sources to 16:9, mild CCTV grade, constant fps.
  const cells = [];
  for (const s of SOURCES) {
    const out = join(TMP, `${s.id}.mp4`);
    cells.push(out);
    if (existsSync(out) && !force) continue;
    console.log(`footage: ${s.id}`);
    await run([
      "-ss", String(s.ss), "-i", s.url, "-t", String(SECONDS), "-an",
      "-vf",
      [
        `scale=${CELL_W}:-2:flags=lanczos`,
        `crop=${CELL_W}:${CELL_H}`,
        `fps=${FPS}`,
        "eq=saturation=0.55:contrast=1.08:gamma=0.96",
        "format=yuv420p",
      ].join(","),
      "-c:v", "libx264", "-preset", "medium", "-crf", "20", out,
    ]);
  }

  // 2) Tile into one 3x3 atlas: one decoder drives every feed on the page.
  const inputs = cells.flatMap((c) => ["-i", c]);
  const layout = [];
  for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) layout.push(`${c * CELL_W}_${r * CELL_H}`);
  console.log("footage: atlas");
  await run([
    ...inputs,
    "-filter_complex", `xstack=inputs=9:layout=${layout.join("|")}:fill=black,format=yuv420p`,
    "-t", String(SECONDS), "-r", String(FPS),
    "-c:v", "libx264", "-preset", "slow", "-crf", "27", "-profile:v", "high",
    "-g", String(FPS * 2), "-movflags", "+faststart", atlas,
  ]);
  await run(["-ss", "1", "-i", atlas, "-frames:v", "1", "-q:v", "4", join(OUT, "poster.jpg")]);

  writeFileSync(
    join(OUT, "manifest.json"),
    JSON.stringify(
      {
        atlas: "/footage/atlas.mp4",
        poster: "/footage/poster.jpg",
        cols: 3,
        rows: 3,
        cell: [CELL_W, CELL_H],
        seconds: SECONDS,
        sources: SOURCES.map((s) => ({ id: s.id, url: s.url, start_s: s.ss })),
        licence: "EPFL CVLab multi-camera pedestrians dataset, research use",
      },
      null,
      2,
    ),
  );
  rmSync(TMP, { recursive: true, force: true });
  console.log(`footage: done -> ${OUT}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
