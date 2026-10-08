// One vertex shader drives every particle scene. Positions are blended from
// morph targets; text, funnel and timeline targets live in camera space so
// they stay composed on screen while the camera flies.

export const particleVertex = /* glsl */ `
precision highp float;
precision highp sampler2D;

attribute vec4 aSeed;
attribute vec4 aMeta;      // time01, camera, class, level
attribute vec3 aNetwork;
attribute float aTextIdx;

uniform float uTime;
uniform mat4 uCamWorld;
uniform float uViewH;      // visible height at uHudDepth
uniform float uAspect;
uniform float uHudDepth;
uniform float uPixelRatio;
uniform float uSize;

uniform sampler2D uText;
uniform float uTextW;
uniform float uTextCount;
uniform float uTextA;
uniform float uTextB;
uniform float uTextMix;
uniform float uWText;

uniform vec3 uW1;          // lattice, network, funnel
uniform vec2 uW2;          // timeline, recon
uniform float uReveal;
uniform float uBurst;
uniform float uSweep;
uniform float uWipe;
uniform vec2 uTitleBox;
uniform float uDrift;
uniform float uHighlightClass;
uniform float uHighlight;
uniform float uStage;
uniform float uRedReveal;
uniform float uFocus;      // timeline focus, seconds since 08:00
uniform float uSpan;       // timeline visible span, seconds
uniform float uDaySpan;
uniform vec3 uReconCenter;
uniform float uReconRadius;
uniform vec2 uPointer;
uniform float uRepel;
uniform float uAlpha;

varying vec3 vColor;
varying float vAlpha;
varying float vCore;

const float TAU = 6.28318530718;

vec3 toWorld(vec3 v) { return (uCamWorld * vec4(v, 1.0)).xyz; }

// Staggered weight: each particle starts its move at a seed-dependent moment,
// so morphs read as a flow instead of a cross-fade.
float stag(float w, float s) {
  float k = 0.6;
  float x = clamp(w * (1.0 + k) - s * k, 0.0, 1.0);
  return x * x * (3.0 - 2.0 * x);
}

vec4 textAt(float row) {
  float idx = row * uTextCount + aTextIdx;
  ivec2 uv = ivec2(int(mod(idx, uTextW)), int(floor(idx / uTextW)));
  return texelFetch(uText, uv, 0);
}

vec3 flow(vec3 p, float s) {
  float t = uTime;
  vec3 d = vec3(
    sin(t * 0.21 + p.y * 0.11 + s * 6.0),
    cos(t * 0.17 + p.x * 0.05 + s * 4.0) * 0.6,
    sin(t * 0.19 + p.x * 0.07 + p.y * 0.05)
  );
#ifdef CURL
  d += 0.5 * vec3(
    sin(t * 0.43 + p.z * 0.31 + p.x * 0.13),
    sin(t * 0.37 + p.x * 0.23),
    cos(t * 0.41 + p.y * 0.29 + s * 9.0)
  );
#endif
  return d;
}

void main() {
  float s = aSeed.x;
  float t01 = aMeta.x;
  float cls = aMeta.z;
  float level = aMeta.w;
  bool isText = aTextIdx > -0.5;
  float viewW = uViewH * uAspect;

  // --- lattice: the structured index, alive with slow flow
  vec3 pL = position + flow(position, s) * (0.55 * uDrift);
  vec3 burstDir = normalize(aSeed.yzw - 0.5 + 1e-4);

  // --- network: an orbiting shell around the camera node
  vec3 dir = normalize(aSeed.wzy - 0.5 + 1e-4);
  float r = 1.4 + 3.2 * aSeed.y * aSeed.y;
  float ang = uTime * (0.12 + 0.25 * aSeed.z) + s * TAU;
  vec3 orbit = vec3(cos(ang) * dir.x - sin(ang) * dir.z, dir.y * 0.45, sin(ang) * dir.x + cos(ang) * dir.z);
  vec3 pN = aNetwork + orbit * r * vec3(1.6, 1.0, 1.6);

  // --- funnel: survivors converge in front of the camera, the rest fall away
  float fail = clamp(uStage - level, 0.0, 1.0);
  float stageR = mix(34.0, 0.6, pow(clamp(uStage / 4.0, 0.0, 1.0), 0.7));
  float fa = s * TAU * 9.0 + uTime * (0.25 + aSeed.z * 0.2);
  float fr = stageR * sqrt(aSeed.y);
  vec3 fv = vec3(cos(fa) * fr, sin(fa) * fr * 0.62, -uHudDepth - 8.0 + (aSeed.w - 0.5) * 14.0);
  fv.y -= fail * fail * (26.0 + 40.0 * aSeed.z);
  fv.x += fail * (aSeed.w - 0.5) * 18.0;
  vec3 pF = toWorld(fv);

  // --- timeline: x is time again, at any zoom
  float tSec = t01 * uDaySpan;
  float tx = (tSec - uFocus) / uSpan * viewW;
  float lane = (1.5 - cls) * 0.115 * uViewH;
  vec3 tv = vec3(tx, lane + (aSeed.z - 0.5) * 0.07 * uViewH, -uHudDepth + (aSeed.w - 0.5) * 3.0);
  vec3 pT = toWorld(tv);

  // --- reconstruction: a slow disc of every event around the site
  float ra = s * TAU + uTime * 0.03 * (0.4 + aSeed.y);
  float rr = uReconRadius * (0.62 + 0.75 * aSeed.z);
  vec3 pR = uReconCenter + vec3(cos(ra) * rr, (aSeed.w - 0.5) * 10.0 + sin(ra * 3.0 + s * 4.0) * 3.0 + 26.0, sin(ra) * rr * 0.78);

  float w1 = stag(uW1.x, s), w2 = stag(uW1.y, s), w3 = stag(uW1.z, s), w4 = stag(uW2.x, s), w5 = stag(uW2.y, s);
  float sum = w1 + w2 + w3 + w4 + w5;
  vec3 pos = sum > 1e-3 ? (pL * w1 + pN * w2 + pF * w3 + pT * w4 + pR * w5) / sum : pL;
  pos += burstDir * uBurst * (4.0 + 30.0 * aSeed.x * aSeed.x);

  float textW = 0.0;
  float wipeShow = 1.0;
  if (isText && uWText > 0.0) {
    vec4 ta = textAt(uTextA);
    vec4 tb = textAt(uTextB);
    float m = stag(uTextMix, aSeed.z);
    vec2 sp = mix(ta.xy, tb.xy, m);
    // particles swirl between glyphs instead of sliding straight across
    float arc = sin(m * 3.14159) * (0.04 + 0.08 * aSeed.y);
    sp += vec2(cos(s * TAU), sin(s * TAU)) * arc;
    textW = stag(uWText, s);
    if (uSweep > 0.5) {
      // The opening: the DOM title is wiped left to right by a CSS mask with the
      // same numbers as here, so a mote shows only where its solid letter has
      // just gone, and lets go from that edge with a little ragged delay.
      float ut = (sp.x + 0.5 - uTitleBox.x) / max(uTitleBox.y - uTitleBox.x, 1e-3);
      float a = uWipe * 1.35 - 0.25;
      float b = uWipe * 1.35;
      wipeShow = 1.0 - clamp((ut - a) / (b - a), 0.0, 1.0);
      // behind the edge the letter holds as dust for a moment, then drifts off
      float rel = clamp((a - ut) / 0.9 - 0.25 * aSeed.w, 0.0, 1.0);
      textW = min(textW, 1.0 - rel * rel * (3.0 - 2.0 * rel));
    }
    // while it loosens, each mote is carried on a soft wind with a slow curl
    float loose = (1.0 - textW) * textW * 4.0 * uSweep;
    vec2 wind = vec2(0.10 + 0.16 * aSeed.y, 0.035 + 0.07 * (aSeed.z - 0.35));
    wind += 0.018 * vec2(sin(s * 41.0 + uTime * 0.7), cos(s * 29.0 + uTime * 0.55));
    vec3 v = vec3((sp.x + wind.x * loose) * viewW, (sp.y + wind.y * loose) * uViewH, -uHudDepth + (ta.z - 0.5) * 0.6);
    pos = mix(pos, toWorld(v), textW);
  }

  vec4 mv = viewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mv;

  // pointer: a gentle lens that parts the field around the cursor
  vec2 ndc = gl_Position.xy / gl_Position.w;
  vec2 d = ndc - uPointer;
  d.x *= uAspect;
  float dist = length(d);
  float push = uRepel * smoothstep(0.22, 0.0, dist) * 0.05;
  ndc += normalize(d + 1e-5) * push * vec2(1.0 / uAspect, 1.0);
  gl_Position.xy = ndc * gl_Position.w;

  // --- appearance
  float survivor = 1.0 - fail;
  float depthScale = clamp(uHudDepth / max(-mv.z, 1.0), 0.18, 2.0);
  // in a glyph every mote is the same size, so the type reads solid, not grainy
  float size = uSize * mix(0.75 + aSeed.y * 1.1, 1.3, textW * textW) * depthScale;
  size *= mix(1.0, 1.0 + max(uStage - 1.0, 0.0) * 0.9 * survivor, w3);
  size *= mix(1.0, 1.25, textW);
  float hl = uHighlightClass < -0.5 ? 0.0 : (abs(cls - uHighlightClass) < 0.5 ? 1.0 : -1.0);
  size *= 1.0 + max(hl, 0.0) * uHighlight * 0.6;
  gl_PointSize = clamp(size * uPixelRatio, 0.6, 22.0);

  vec3 tint = cls < 0.5 ? vec3(0.87, 0.9, 0.92) : cls < 1.5 ? vec3(0.62, 0.74, 0.86) : cls < 2.5 ? vec3(0.8, 0.77, 0.71) : vec3(0.5, 0.54, 0.6);
  vec3 col = mix(vec3(0.74, 0.78, 0.81), tint, 0.55);
  col = mix(col, vec3(0.92, 0.95, 0.97), textW);
  float bright = 1.0 + (hl > 0.0 ? uHighlight * 1.1 : hl < 0.0 ? -uHighlight * 0.78 : 0.0);
  // the first red on the whole page: red vehicles once the colour filter runs
  float isRed = step(1.5, level) * step(0.5, cls) * step(cls, 1.5);
  float redOn = isRed * uRedReveal * max(w3, w4);
  col = mix(col, vec3(0.96, 0.33, 0.39), redOn);
  bright += redOn * 0.6 + max(uStage - 2.0, 0.0) * survivor * w3 * 0.8;

  float reveal = isText ? 1.0 : smoothstep(s - 0.06, s, uReveal);
  float alpha = uAlpha * reveal * mix(1.0, survivor, w3);
  alpha *= mix(1.0, 1.0 - smoothstep(0.62, 0.8, abs(tx) / viewW), w4);
  alpha *= mix(0.55 + 0.45 * aSeed.w, 1.0, textW * textW);
  alpha *= wipeShow;
  // particles brushing past the lens fade instead of flaring across the type
  alpha *= mix(smoothstep(3.0, 22.0, -mv.z), 1.0, textW);

  vColor = col * bright;
  vAlpha = alpha;
  vCore = redOn + w3 * step(2.5, level) * smoothstep(2.5, 3.5, uStage);
}
`;

export const particleFragment = /* glsl */ `
precision highp float;
varying vec3 vColor;
varying float vAlpha;
varying float vCore;

void main() {
  vec2 c = gl_PointCoord - 0.5;
  float d = length(c);
  if (d > 0.5) discard;
  float soft = pow(1.0 - d * 2.0, 1.6);
  // survivors and red points get a hot core, everything else stays a soft mote
  float core = smoothstep(0.22, 0.0, d) * vCore;
  float a = (soft + core) * vAlpha;
  gl_FragColor = vec4(vColor * a, a);
}
`;
