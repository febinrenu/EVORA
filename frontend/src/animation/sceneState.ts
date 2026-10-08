// The one mutable object the scroll timeline writes and the renderer reads.
// It is deliberately plain (no React state, no proxies): GSAP tweens its
// numbers directly and the engine reads them once per frame.

export interface Vec3Like {
  x: number;
  y: number;
  z: number;
}

export const S = {
  /** seconds since boot, advanced by the shared ticker */
  time: 0,
  /** master scroll progress 0..1 */
  progress: 0,
  /** index into ACTS of the act under the viewport centre */
  act: 0,
  /** pointer in normalised device coordinates (-1..1), smoothed */
  pointer: { x: 0, y: 0, active: 0 },

  /** main camera rig */
  cam: {
    pos: { x: 0, y: 0, z: 60 } as Vec3Like,
    look: { x: 0, y: 0, z: 0 } as Vec3Like,
    fov: 42,
    roll: 0,
    /** 0 = use pos/look, 1 = chase the red sedan along its route */
    follow: 0,
    /** small hand-held drift on top of the rig, 0..1 */
    sway: 1,
  },

  universe: {
    alpha: 0,
    /** morph weights; the timeline keeps them summing to ~1 */
    wText: 1,
    wLattice: 0,
    wNetwork: 0,
    wFunnel: 0,
    wTimeline: 0,
    wRecon: 0,
    /** text rows: 0 title, 1..5 count beats */
    textA: 0,
    textB: 0,
    textMix: 0,
    /** how much of the non-text field is visible (particles multiplying) */
    reveal: 0,
    /** scatter impulse used while text breaks apart */
    burst: 0,
    /** text lets go left to right on a wind (the opening title) instead of in random order */
    sweep: 0,
    /** the wipe across the DOM title, 0..1, the same value as its CSS --sweep */
    wipe: 0,
    /** the DOM title's left and right edge as viewport fractions (measured with the glyphs) */
    titleL: 0.2,
    titleR: 0.8,
    drift: 1,
    /** class highlighted by the current number, -1 none */
    highlightClass: -1,
    highlight: 0,
    /** funnel stage, continuous 0..4 */
    stage: 0,
    /** red appears only once the colour filter runs */
    redReveal: 0,
    /** timeline zoom 0 (whole day) .. 1 (seconds) */
    zoom: 0,
    size: 1,
    /** pointer repulsion strength */
    repel: 1,
  },

  sleeves: {
    alpha: 0,
    /** layout weights: network -> grid -> diagonal -> curved wall -> solo */
    wNetwork: 1,
    wGrid: 0,
    wDiagonal: 0,
    wCurve: 0,
    wSolo: 0,
    /** extra blackout on the solo feed as it collapses to a scanline */
    collapse: 0,
  },

  links: { alpha: 0, draw: 0 },

  twin: {
    alpha: 0,
    /** 0 = night blueprint model, 1 = daylight as the cameras see it */
    look: 0,
    /** CCTV post-processing amount on the main view */
    cctv: 0,
    /** circular reveal of the main view, 0 closed .. 1 open */
    iris: 1,
    /** route progress of the red sedan, 0..1 */
    car: 0,
    person: 0,
    /** red trace ribbon draw progress */
    trace: 0,
    traceAlpha: 0,
    /** yellow memory beacons */
    beacons: 0,
    /** the newly learned main-gate landmark */
    gate: 0,
    /** ghosted cars at each hop during the reconstruction */
    ghosts: 0,
    /** recon evidence sleeves */
    recon: 0,
  },

  /** DOM-driven acts read these to decide what to do */
  evidence: { box: 0 },
  memory: { auto: 0, draw: 0 },
  lights: 0,
  /** global fade to black over the canvas */
  blackout: 0,
};

export type SceneState = typeof S;

/** Bumped whenever the renderer must draw even if nothing animates. */
export const dirty = { value: true };
