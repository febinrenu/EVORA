#!/usr/bin/env bash
# Normalize raw videos to H.264 mp4, max 1280 px wide, original fps, +faststart.
# Usage: scripts/data/transcode.sh <raw_dir> <norm_dir>
# Example: scripts/data/transcode.sh data/raw/epfl data/norm/epfl
set -euo pipefail
src="${1:?raw dir}"; dst="${2:?norm dir}"
mkdir -p "$dst"
shopt -s nullglob nocaseglob
for f in "$src"/*.avi "$src"/*.mp4 "$src"/*.mov "$src"/*.mkv; do
  out="$dst/$(basename "${f%.*}").mp4"
  if [ -s "$out" ] && [ "$out" -nt "$f" ]; then echo "skip $out"; continue; fi
  echo "transcode $f -> $out"
  ffmpeg -nostdin -hide_banner -loglevel error -y -i "$f" \
    -vf "scale=min(1280\,iw):-2" -c:v libx264 -preset veryfast -crf 21 \
    -pix_fmt yuv420p -an -movflags +faststart -f mp4 "$out.part"
  mv "$out.part" "$out"
done
