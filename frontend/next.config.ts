import type { NextConfig } from "next";

const production = process.env.NODE_ENV === "production";

// The production build is a static export into frontend/dist, which the API
// serves on :8700 (one process, one origin, works offline). `next dev` keeps
// its own .next folder so a dev session never breaks the served build.
const nextConfig: NextConfig = {
  output: "export",
  // EVORA_DIST_DIR builds elsewhere (audits) without replacing the served dist/
  distDir: process.env.EVORA_DIST_DIR ?? (production ? "dist" : ".next"),
  // /app/ -> app/index.html, which plain static serving resolves without rewrites
  trailingSlash: true,
  images: { unoptimized: true },
};

export default nextConfig;
