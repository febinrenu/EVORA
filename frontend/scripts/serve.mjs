#!/usr/bin/env node
// Serve the static export (dist/) without any dependency, offline:
//   node scripts/serve.mjs [--port 3000] [--hostname 127.0.0.1] [--dir dist]
// Used by `npm run start` (and `make up` when the API is not serving dist/).
// Folders resolve to their index.html (trailingSlash export); range requests
// are honoured so videos seek and loop.
import { createReadStream, existsSync, statSync } from "node:fs";
import { createServer } from "node:http";
import { extname, join, normalize, resolve, sep } from "node:path";
import { fileURLToPath } from "node:url";

const arg = (name, fallback) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
};
const root = resolve(fileURLToPath(new URL(`../${arg("dir", "dist")}`, import.meta.url)));
const port = Number(arg("port", "3000"));
const host = arg("hostname", "127.0.0.1");

const TYPES = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json",
  ".txt": "text/plain; charset=utf-8",
  ".woff2": "font/woff2",
  ".mp4": "video/mp4",
  ".jpg": "image/jpeg",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".ico": "image/x-icon",
};

function resolveFile(urlPath) {
  const clean = normalize(decodeURIComponent(urlPath.split("?")[0])).replace(/^([/\\])+/, "");
  const full = resolve(join(root, clean));
  if (full !== root && !full.startsWith(root + sep)) return null; // no escaping dist/
  if (existsSync(full) && statSync(full).isFile()) return full;
  const index = join(full, "index.html");
  if (existsSync(index)) return index;
  return null;
}

if (!existsSync(join(root, "index.html"))) {
  console.error("dist/ has no build yet: run `npm run build` first");
  process.exit(1);
}

createServer((req, res) => {
  const file = resolveFile(req.url ?? "/") ?? join(root, "404.html");
  const status = file.endsWith("404.html") && !req.url?.startsWith("/404") ? 404 : 200;
  const size = statSync(file).size;
  const type = TYPES[extname(file)] ?? "application/octet-stream";
  const range = /bytes=(\d*)-(\d*)/.exec(req.headers.range ?? "");
  if (range && status === 200) {
    const start = range[1] ? Number(range[1]) : 0;
    const end = range[2] ? Math.min(Number(range[2]), size - 1) : size - 1;
    res.writeHead(206, { "content-type": type, "content-range": `bytes ${start}-${end}/${size}`, "accept-ranges": "bytes", "content-length": end - start + 1 });
    createReadStream(file, { start, end }).pipe(res);
    return;
  }
  res.writeHead(status, { "content-type": type, "content-length": size, "accept-ranges": "bytes", "cache-control": file.includes(`${sep}_next${sep}`) ? "public, max-age=31536000, immutable" : "no-cache" });
  createReadStream(file).pipe(res);
}).listen(port, host, () => console.log(`evora UI on http://${host}:${port}`));
