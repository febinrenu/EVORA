// Materials for the digital twin. One "massing model" shader serves every
// solid: at uLook 0 it is a night architectural model (dark volumes, a ground
// grid), at uLook 1 it is the flat daylight a CCTV camera actually records.

export const massVertex = /* glsl */ `
varying vec3 vNormal;
varying vec3 vWorld;
varying float vDepth;
varying vec3 vTint;
void main() {
  vec4 local = vec4(position, 1.0);
#ifdef USE_COLOR
  vTint = color;
#else
  vTint = vec3(1.0);
#endif
#ifdef USE_INSTANCING
  local = instanceMatrix * local;
  vNormal = normalize(mat3(modelMatrix) * mat3(instanceMatrix) * normal);
#else
  vNormal = normalize(mat3(modelMatrix) * normal);
#endif
  vec4 world = modelMatrix * local;
  vWorld = world.xyz;
  vec4 mv = viewMatrix * world;
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

export const massFragment = /* glsl */ `
precision highp float;
uniform vec3 uColor;
uniform float uLook;
uniform float uGrid;
uniform float uAlpha;
uniform float uEmissive;
uniform vec3 uFogNight;
uniform vec3 uFogDay;
varying vec3 vNormal;
varying vec3 vWorld;
varying float vDepth;
varying vec3 vTint;

void main() {
  // baked batches carry their albedo per vertex; single meshes use uColor
  vec3 albedo = uColor * vTint;
  vec3 n = normalize(vNormal);
  vec3 sun = normalize(vec3(0.45, 0.85, 0.25));
  float diff = max(dot(n, sun), 0.0);
  float hemi = 0.5 + 0.5 * n.y;
  // cheap ambient occlusion where walls meet the ground
  float ao = mix(0.62, 1.0, smoothstep(0.0, 2.2, vWorld.y));

  vec3 day = albedo * (0.38 + 0.42 * hemi + 0.6 * diff) * ao;
  vec3 night = albedo * (0.085 + 0.1 * hemi + 0.06 * diff) * ao + vec3(0.014, 0.02, 0.027);

  if (uGrid > 0.5) {
    vec2 g = abs(fract(vWorld.xz / 5.0) - 0.5);
    float line = smoothstep(0.485, 0.5, max(g.x, g.y));
    vec2 G = abs(fract(vWorld.xz / 25.0) - 0.5);
    float major = smoothstep(0.494, 0.5, max(G.x, G.y));
    night += vec3(0.32, 0.5, 0.66) * (line * 0.05 + major * 0.11) * smoothstep(240.0, 30.0, vDepth);
  }
  vec3 col = mix(night, day, uLook) + albedo * uEmissive * (1.0 - 0.8 * uLook);
  float fogN = smoothstep(70.0, 300.0, vDepth);
  float fogD = smoothstep(120.0, 420.0, vDepth);
  col = mix(col, mix(uFogNight, uFogDay, uLook), mix(fogN, fogD, uLook));
  gl_FragColor = vec4(col, uAlpha);
}
`;

/** Model edges: bright in the night model, nearly invisible in daylight. */
export const edgeVertex = /* glsl */ `
varying float vDepth;
void main() {
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vDepth = -mv.z;
  gl_Position = projectionMatrix * mv;
}
`;

export const edgeFragment = /* glsl */ `
precision highp float;
uniform vec3 uColor;
uniform float uLook;
uniform float uAlpha;
varying float vDepth;
void main() {
  float a = mix(0.55, 0.05, uLook) * smoothstep(380.0, 30.0, vDepth) * uAlpha;
  gl_FragColor = vec4(uColor, a);
}
`;

/** Memory beacons: a soft vertical column and a pulsing ground ring in marker yellow. */
export const beaconVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const beaconFragment = /* glsl */ `
precision highp float;
uniform vec3 uColor;
uniform float uAlpha;
uniform float uTime;
uniform float uPulse;
uniform float uRing;
varying vec2 vUv;
void main() {
  float a;
  if (uRing > 0.5) {
    // flat disc: a thin band near the rim, scaled outward by the CPU as a ripple
    float d = length(vUv - 0.5) * 2.0;
    float band = smoothstep(0.78, 0.92, d) * smoothstep(1.0, 0.94, d);
    a = band * uAlpha * (1.0 - uPulse);
  } else {
    float up = 1.0 - vUv.y;
    float across = 1.0 - abs(vUv.x - 0.5) * 2.0;
    a = pow(up, 2.2) * pow(max(across, 0.0), 0.7) * 0.85 * uAlpha;
    a += smoothstep(0.02, 0.0, vUv.y) * 0.6 * uAlpha;
  }
  gl_FragColor = vec4(uColor * a, a);
}
`;

/** Recon evidence cards: a feed texture inside a thin sleeve edge. */
export const cardVertex = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const cardFragment = /* glsl */ `
precision highp float;
uniform sampler2D uMap;
uniform float uAlpha;
uniform float uTime;
uniform vec3 uEdge;
varying vec2 vUv;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453); }
void main() {
  float ar = 16.0 / 9.0;
  vec2 px = vec2(vUv.x * ar, vUv.y);
  float e = min(min(px.x, ar - px.x), min(px.y, 1.0 - px.y));
  float edge = smoothstep(0.016, 0.006, e);
  vec3 col = texture2D(uMap, vUv).rgb;
  col += (hash(floor(vUv * vec2(480.0, 270.0)) + floor(uTime * 12.0)) - 0.5) * 0.06;
  col = mix(col, uEdge, edge);
  gl_FragColor = vec4(col, uAlpha);
}
`;
