// The CCTV lens. Applied to the main view while the viewer is "inside" a
// camera, and to every feed snapshot, so the same twin reads as both a model
// and as footage. Everything here is a single full-screen pass.

export const fullscreenVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = vec4(position.xy, 0.0, 1.0);
}
`;

export const cctvFragment = /* glsl */ `
precision highp float;
uniform sampler2D tScene;
uniform vec2 uRes;
uniform float uAmount;
uniform float uIris;
uniform float uFade;
uniform float uTime;
uniform float uFrame;
uniform float uBarrel;

varying vec2 vUv;

float hash(vec2 p) { return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453); }

vec2 barrel(vec2 uv, float k) {
  vec2 c = uv - 0.5;
  return 0.5 + c * (1.0 + k * dot(c, c));
}

void main() {
  float k = uBarrel * uAmount;
  vec2 uv = barrel(vUv, k);
  vec3 clean = texture2D(tScene, vUv).rgb;
  vec3 col = clean;

  if (uAmount > 0.001) {
    // 4:2:0 chroma: luma from the sharp sample, colour from an 8 px macroblock
    vec2 block = (floor(uv * uRes / 8.0) + 0.5) * 8.0 / uRes;
    vec3 sharp = texture2D(tScene, uv).rgb;
    vec3 chromaSrc = texture2D(tScene, block).rgb * 0.6 + texture2D(tScene, uv + vec2(2.5, 0.0) / uRes).rgb * 0.4;
    float Y = dot(sharp, vec3(0.299, 0.587, 0.114));
    float Yc = dot(chromaSrc, vec3(0.299, 0.587, 0.114));
    vec3 cctv = vec3(Y) + (chromaSrc - Yc);
    // NVR grade: most colour drained, but a red object keeps its red
    float redness = smoothstep(0.08, 0.26, cctv.r - max(cctv.g, cctv.b));
    cctv = mix(vec3(Y), cctv, mix(0.32, 1.0, redness));
    cctv = (cctv - 0.5) * 1.1 + 0.5;
    cctv *= vec3(0.97, 1.0, 0.98);
    // block quantisation breathes a little, like a stream under load
    float q = 24.0 + 8.0 * sin(uFrame * 0.7);
    cctv = mix(cctv, floor(cctv * q + 0.5) / q, 0.35);
    // sensor noise held per recorded frame
    float n = hash(floor(uv * uRes / 1.5) + uFrame) - 0.5;
    cctv += n * 0.075;
    cctv *= 0.95 + 0.05 * sin(uv.y * uRes.y * 1.6);
    vec2 vc = uv - 0.5;
    cctv *= 1.0 - dot(vc, vc) * 1.05;
    // outside the barrel image is the lens housing
    float inside = step(0.0, uv.x) * step(uv.x, 1.0) * step(0.0, uv.y) * step(uv.y, 1.0);
    cctv *= inside;
    col = mix(clean, cctv, uAmount);
  }

  // iris: the collapsed point opens into the frame
  vec2 p = (vUv - 0.5) * vec2(uRes.x / uRes.y, 1.0);
  float maxR = length(vec2(uRes.x / uRes.y, 1.0)) * 0.5;
  float r = uIris * maxR;
  float d = length(p);
  float mask = smoothstep(r + 0.002, r - 0.002, d);
  float ring = smoothstep(0.006, 0.0, abs(d - r)) * step(0.001, uIris) * step(uIris, 0.999);
  col = col * mask + vec3(0.86, 0.89, 0.91) * ring * 0.8;
  gl_FragColor = vec4(col * uFade, 1.0);
}
`;
