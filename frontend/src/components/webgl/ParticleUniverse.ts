// <ParticleUniverse />: the event index as one Points mesh. Every particle
// scene (title, lattice, counts, network, funnel, timeline, reconstruction) is
// a uniform change on this single draw call.
import {
  BufferAttribute,
  BufferGeometry,
  CustomBlending,
  Matrix4,
  OneFactor,
  Points,
  ShaderMaterial,
  Vector2,
  Vector3,
  type PerspectiveCamera,
} from "three";
import { S } from "@/animation/sceneState";
import { CAMERA } from "@/lib/config/animation";
import { DAY } from "@/lib/data/story";
import { buildUniverse } from "@/lib/data/universe";
import { timeWindow } from "@/lib/data/timeWindow";
import type { NetworkNode } from "@/lib/data/site";
import { particleFragment, particleVertex } from "./shaders/particles";
import { TextTargets, type TextRowSpec } from "./TextTargets";

export class ParticleUniverse {
  readonly points: Points;
  readonly text: TextTargets;
  private readonly material: ShaderMaterial;
  private readonly geometry: BufferGeometry;
  private readonly count: number;
  private share = 1;

  constructor(count: number, textCount: number, nodes: NetworkNode[], fontFamily: string, curl: boolean) {
    const data = buildUniverse(count, textCount, nodes);
    this.count = count;
    this.text = new TextTargets(textCount, fontFamily);

    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(data.position, 3));
    g.setAttribute("aSeed", new BufferAttribute(data.seed, 4));
    g.setAttribute("aMeta", new BufferAttribute(data.meta, 4));
    g.setAttribute("aNetwork", new BufferAttribute(data.network, 3));
    g.setAttribute("aTextIdx", new BufferAttribute(data.textIndex, 1));
    this.geometry = g;

    this.material = new ShaderMaterial({
      vertexShader: particleVertex,
      fragmentShader: particleFragment,
      defines: curl ? { CURL: "" } : {},
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: CustomBlending,
      blendSrc: OneFactor,
      blendDst: OneFactor,
      uniforms: {
        uTime: { value: 0 },
        uCamWorld: { value: new Matrix4() },
        uViewH: { value: 1 },
        uAspect: { value: 1 },
        uHudDepth: { value: CAMERA.hudDepth },
        uPixelRatio: { value: 1 },
        uSize: { value: 1.6 },
        uText: { value: this.text.texture },
        uTextW: { value: this.text.texWidth },
        uTextCount: { value: textCount },
        uTextA: { value: 0 },
        uTextB: { value: 0 },
        uTextMix: { value: 0 },
        uWText: { value: 1 },
        uW1: { value: new Vector3(0, 0, 0) },
        uW2: { value: new Vector2(0, 0) },
        uReveal: { value: 0 },
        uBurst: { value: 0 },
        uSweep: { value: 0 },
        uWipe: { value: 0 },
        uTitleBox: { value: new Vector2(0.2, 0.8) },
        uDrift: { value: 1 },
        uHighlightClass: { value: -1 },
        uHighlight: { value: 0 },
        uStage: { value: 0 },
        uRedReveal: { value: 0 },
        uFocus: { value: DAY.focusT },
        uSpan: { value: DAY.span },
        uDaySpan: { value: DAY.span },
        uReconCenter: { value: new Vector3(6, 0, -8) },
        uReconRadius: { value: 92 },
        uPointer: { value: new Vector2(9, 9) },
        uRepel: { value: 1 },
        uAlpha: { value: 0 },
      },
    });

    this.points = new Points(g, this.material);
    this.points.frustumCulled = false;
  }

  buildText(rows: TextRowSpec[], w: number, h: number): void {
    this.text.build(rows, w, h);
  }

  /** Governor hook: draw a uniform subset (particles are stored in random order). */
  setShare(share: number): void {
    this.share = share;
    this.geometry.setDrawRange(0, Math.floor(this.count * share));
  }

  get drawShare(): number {
    return this.share;
  }

  update(camera: PerspectiveCamera, pixelRatio: number): void {
    const u = this.material.uniforms;
    const U = S.universe;
    this.points.visible = U.alpha > 0.002;
    if (!this.points.visible) return;
    u.uTime.value = S.time;
    u.uCamWorld.value.copy(camera.matrixWorld);
    u.uViewH.value = 2 * CAMERA.hudDepth * Math.tan((camera.fov * Math.PI) / 360);
    u.uAspect.value = camera.aspect;
    u.uPixelRatio.value = pixelRatio;
    u.uSize.value = 1.55 * U.size;
    u.uTextA.value = U.textA;
    u.uTextB.value = U.textB;
    u.uTextMix.value = U.textMix;
    u.uWText.value = U.wText;
    u.uW1.value.set(U.wLattice, U.wNetwork, U.wFunnel);
    u.uW2.value.set(U.wTimeline, U.wRecon);
    u.uReveal.value = U.reveal;
    u.uBurst.value = U.burst;
    u.uSweep.value = U.sweep;
    u.uWipe.value = U.wipe;
    u.uTitleBox.value.set(U.titleL, U.titleR);
    u.uDrift.value = U.drift;
    u.uHighlightClass.value = U.highlightClass;
    u.uHighlight.value = U.highlight;
    u.uStage.value = U.stage;
    u.uRedReveal.value = U.redReveal;
    const win = timeWindow(U.zoom);
    u.uSpan.value = win.span;
    u.uFocus.value = win.focus;
    u.uPointer.value.set(S.pointer.x, S.pointer.y);
    u.uRepel.value = U.repel * S.pointer.active;
    u.uAlpha.value = U.alpha;
  }

  dispose(): void {
    this.geometry.dispose();
    this.material.dispose();
    this.text.dispose();
  }
}
