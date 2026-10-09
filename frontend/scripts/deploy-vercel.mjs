// Publish the story (/), the product shell (/app) and the report (/report) to Vercel
// as a static site. The report's numbers are baked in at build time from
// ../eval/reports/report.json, which a build on Vercel would not have, so the site is
// built here and the output folder is uploaded as it is (footage included).
//
//   npm run deploy            preview deployment
//   npm run deploy -- --prod  production (https://evora-ivory.vercel.app)
//
// Needs the Vercel CLI, logged in. EVORA_VERCEL_PROJECT picks another project name.
import { execSync } from "node:child_process";
import { existsSync, rmSync, writeFileSync } from "node:fs";

const OUT = ".vercel-out";
const prod = process.argv.includes("--prod");
const project = process.env.EVORA_VERCEL_PROJECT ?? "evora";
const run = (cmd, cwd = ".") => execSync(cmd, { stdio: "inherit", cwd, env: { ...process.env, EVORA_DIST_DIR: OUT } });

if (!existsSync("public/footage/atlas.mp4")) console.warn("public/footage is missing: the camera wall will use procedural feeds (npm run footage builds it).");
run("npm run build");
// /app and /report also answer without the trailing slash
writeFileSync(`${OUT}/vercel.json`, `${JSON.stringify({ $schema: "https://openapi.vercel.sh/vercel.json", trailingSlash: true }, null, 2)}\n`);
run(`vercel link --yes --project ${project}`, OUT);
// linking writes the project's environment (a token) next to the site: it must never be uploaded
rmSync(`${OUT}/.env.local`, { force: true });
run(`vercel deploy --yes${prod ? " --prod" : ""}`, OUT);
