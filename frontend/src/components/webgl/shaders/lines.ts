// Hairline arcs between cameras. Drawn on by scroll; a faint pulse travels each
// link so the network reads as live without anything flashing.
export const linkVertex = /* glsl */ `
attribute float aT;
attribute float aLink;
varying float vT;
varying float vLink;
varying float vFog;
void main() {
  vT = aT;
  vLink = aLink;
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vFog = smoothstep(260.0, 60.0, -mv.z);
  gl_Position = projectionMatrix * mv;
}
`;

export const linkFragment = /* glsl */ `
precision highp float;
uniform float uDraw;
uniform float uAlpha;
uniform float uTime;
uniform vec3 uColor;
varying float vT;
varying float vLink;
varying float vFog;
void main() {
  float local = uDraw * 1.7 - vLink * 0.7;
  if (vT > local) discard;
  float head = smoothstep(0.08, 0.0, local - vT) * step(local, 1.0);
  float pulse = smoothstep(0.06, 0.0, abs(fract(vT - uTime * (0.05 + vLink * 0.06) - vLink) - 0.5) - 0.44);
  float a = (0.28 + 0.5 * pulse + 0.9 * head) * uAlpha * vFog;
  gl_FragColor = vec4(uColor * a, a);
}
`;

/** Ribbon used for the red trace: bright head, softer tail, hidden from inside CCTV views. */
export const ribbonVertex = /* glsl */ `
attribute float aT;
attribute float aSide;
varying float vT;
varying float vSide;
void main() {
  vT = aT;
  vSide = aSide;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

export const ribbonFragment = /* glsl */ `
precision highp float;
uniform float uDraw;
uniform float uAlpha;
uniform float uTime;
uniform vec3 uColor;
varying float vT;
varying float vSide;
void main() {
  if (vT > uDraw) discard;
  float across = 1.0 - abs(vSide);
  float core = smoothstep(0.0, 0.7, across);
  float head = smoothstep(0.035, 0.0, uDraw - vT);
  float flow = 0.75 + 0.25 * sin(vT * 160.0 - uTime * 4.0);
  float a = (core * 0.75 * flow + head * 1.4) * uAlpha;
  gl_FragColor = vec4(uColor * a, a);
}
`;
