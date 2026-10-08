// Video "sleeves": camera feeds mounted like negatives in a sleeve. One
// instanced draw call; each instance picks its atlas cell and OSD label.

export const sleeveVertex = /* glsl */ `
attribute float aCell;
attribute float aLabel;
attribute float aAlpha;
attribute float aHover;
attribute float aSolo;

uniform float uCollapse;
uniform float uTime;

varying vec2 vUv;
varying float vCell;
varying float vLabel;
varying float vAlpha;
varying float vHover;
varying float vSolo;

void main() {
  vUv = uv;
  vCell = aCell;
  vLabel = aLabel;
  vAlpha = aAlpha;
  vHover = aHover;
  vSolo = aSolo;
  vec3 p = position;
  // hovering ripples the glass a touch toward the viewer
  p.z += aHover * sin(uv.x * 9.0 + uTime * 3.0) * sin(uv.y * 7.0) * 0.03;
  // the solo feed collapses to a scanline like a monitor switching off
  p.y *= mix(1.0, 0.0035, uCollapse * aSolo);
  p.x *= mix(1.0, 1.0 + 0.12 * uCollapse, aSolo);
  gl_Position = projectionMatrix * modelViewMatrix * instanceMatrix * vec4(p, 1.0);
}
`;

export const sleeveFragment = /* glsl */ `
precision highp float;
uniform sampler2D uVideo;
uniform sampler2D uLabels;
uniform float uHasVideo;
uniform float uTime;
uniform float uAlpha;
uniform float uCollapse;
uniform vec2 uLabelGrid;   // cols, rows
uniform float uSoloAmt;

varying vec2 vUv;
varying float vCell;
varying float vLabel;
varying float vAlpha;
varying float vHover;
varying float vSolo;

float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }

// Procedural feed for clones without footage: a ground plane in perspective,
// slow figures, sensor noise. Good enough at sleeve size.
vec3 synthetic(vec2 uv, float cell) {
  float horizon = 0.62 + 0.06 * sin(cell * 1.7);
  vec3 col = mix(vec3(0.20, 0.22, 0.23), vec3(0.34, 0.36, 0.36), uv.y);
  if (uv.y < horizon) {
    float depth = (horizon - uv.y) / horizon;
    vec2 g = vec2((uv.x - 0.5) / (depth + 0.05), 1.0 / (depth + 0.05));
    float grid = smoothstep(0.96, 1.0, abs(fract(g.x * 0.5) - 0.5) * 2.0) + smoothstep(0.96, 1.0, abs(fract(g.y * 0.25 + cell) - 0.5) * 2.0);
    col = mix(vec3(0.27, 0.28, 0.28), vec3(0.22, 0.23, 0.23), depth) + grid * 0.03;
  }
  for (int i = 0; i < 4; i++) {
    float fi = float(i);
    float x = fract(0.15 + fi * 0.27 + uTime * (0.012 + 0.006 * fi) * (mod(fi, 2.0) * 2.0 - 1.0) + cell * 0.13);
    float y = 0.18 + 0.32 * fract(fi * 0.37 + cell * 0.21);
    float h = 0.22 * (1.0 - y);
    vec2 d = (uv - vec2(x, y + h * 0.5)) / vec2(h * 0.28, h * 0.5);
    col = mix(col, vec3(0.1, 0.11, 0.12), smoothstep(1.0, 0.7, length(d)) * 0.85);
  }
  return col;
}

void main() {
  vec2 uv = vUv;
  float ar = 16.0 / 9.0;

  // thin edge like a negative in a sleeve, with corner ticks
  vec2 px = vec2(uv.x * ar, uv.y);
  float edgeDist = min(min(px.x, ar - px.x), min(px.y, 1.0 - px.y));
  float edge = smoothstep(0.012, 0.004, edgeDist);
  vec2 cd = min(vec2(px.x, px.y), vec2(ar - px.x, 1.0 - px.y));
  float tick = step(cd.x, 0.07) * step(cd.y, 0.07) * smoothstep(0.022, 0.012, min(cd.x, cd.y));

  // the picture, slightly inset
  vec2 inner = (uv - 0.5) * 1.02 + 0.5;
  float col_i = mod(vCell, 3.0);
  float row_i = floor(vCell / 3.0);
  vec2 wob = vec2(sin(uv.y * 40.0 + uTime * 6.0), cos(uv.x * 30.0 + uTime * 5.0)) * 0.0025 * vHover;
  vec2 auv = vec2((col_i + clamp(inner.x + wob.x, 0.002, 0.998)) / 3.0, (2.0 - row_i + clamp(inner.y + wob.y, 0.002, 0.998)) / 3.0);
  vec3 video = uHasVideo > 0.5 ? texture2D(uVideo, auv).rgb : synthetic(inner, vCell);

  // NVR look: a little desaturated, crushed, grainy, soft scanlines
  float luma = dot(video, vec3(0.299, 0.587, 0.114));
  vec3 col = mix(vec3(luma), video, 0.55);
  col = (col - 0.5) * 1.12 + 0.48;
  float grain = hash(floor(uv * vec2(480.0, 270.0)) + floor(uTime * 12.0)) - 0.5;
  col += grain * 0.07;
  col *= 0.94 + 0.06 * sin(uv.y * 270.0 * 3.14159);
  vec2 vc = uv - 0.5;
  col *= 1.0 - dot(vc, vc) * 0.9;
  col *= 1.0 + vHover * 0.18;

  // OSD label (camera id, place, clock) burned bottom-left
  float lscale = mix(0.5, 0.2, vSolo * uSoloAmt);
  vec2 lsize = vec2(lscale, lscale * ar / 6.095);
  vec2 luv = (uv - vec2(0.03, 0.045)) / lsize;
  if (luv.x > 0.0 && luv.x < 1.0 && luv.y > 0.0 && luv.y < 1.0) {
    float lx = mod(vLabel, uLabelGrid.x);
    float ly = floor(vLabel / uLabelGrid.x);
    vec2 tuv = vec2((lx + luv.x) / uLabelGrid.x, 1.0 - (ly + 1.0 - luv.y) / uLabelGrid.y);
    vec4 l = texture2D(uLabels, tuv);
    col = mix(col, l.rgb, l.a);
  }
  col = mix(col, vec3(0.86, 0.89, 0.91), edge * 0.55 + tick * 0.8);

  // scanline collapse glow
  col += vec3(0.9, 0.95, 1.0) * uCollapse * vSolo * 1.6;
  gl_FragColor = vec4(col, uAlpha * vAlpha);
}
`;
