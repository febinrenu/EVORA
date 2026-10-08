// The one renderer. A single fixed canvas, a single camera and a single frame
// function driven by the shared GSAP ticker. Scenes are layers that switch
// themselves off when the scroll state says they are invisible, and the frame
// is skipped entirely when nothing is on screen.
import {
  Color,
  HalfFloatType,
  LinearFilter,
  Mesh,
  OrthographicCamera,
  PerspectiveCamera,
  PlaneGeometry,
  Scene,
  ShaderMaterial,
  SRGBColorSpace,
  Vector2,
  Vector3,
  WebGLRenderer,
  WebGLRenderTarget,
  type Object3D,
  type Texture,
} from "three";
import { S } from "@/animation/sceneState";
import { CAMERA, CCTV } from "@/lib/config/animation";
import type { TierSpec } from "@/lib/config/tiers";
import { buildNetwork, twinCam, type CameraPose, type NetworkNode } from "@/lib/data/site";
import { Governor } from "@/lib/perf/tier";
import { CameraNetwork } from "./CameraNetwork";
import { DigitalTwin, FOG_DAY, FOG_NIGHT } from "./DigitalTwin";
import { ParticleUniverse } from "./ParticleUniverse";
import { cctvFragment, fullscreenVertex } from "./shaders/cctv";
import type { TextRowSpec } from "./TextTargets";
import { VideoAtlas } from "./textures/VideoAtlas";

export interface EngineOptions {
  canvas: HTMLCanvasElement;
  spec: TierSpec;
  mobile: boolean;
  fonts: { display: string; mono: string };
}

export type FrameHook = (engine: Engine) => void;

const VOID = new Color("#06080A");
const MIN_FRAME_MS = 1000 / 250;

export class Engine {
  readonly renderer: WebGLRenderer;
  readonly camera: PerspectiveCamera;
  readonly universeScene = new Scene();
  readonly nodes: NetworkNode[];
  readonly universe: ParticleUniverse;
  readonly network: CameraNetwork;
  readonly atlas = new VideoAtlas();
  twin: DigitalTwin | null = null;

  private readonly post: { scene: Scene; camera: OrthographicCamera; mat: ShaderMaterial; quad: Mesh };
  private sceneRT: WebGLRenderTarget;
  private readonly hooks = new Set<FrameHook>();
  private readonly governor: Governor;
  private dpr: number;
  private width = 1;
  private height = 1;
  private lastFrame = 0;
  private lastCctvRender = -1;
  private drewBlank = false;
  private readonly chasePos = new Vector3();
  private readonly chaseLook = new Vector3();
  private readonly look = new Vector3();
  private readonly clear = new Color();
  private readonly v = new Vector3();
  private disposed = false;
  /** DPR change requested by the governor, applied at the next all-black frame */
  private pendingDpr: number | null = null;
  /** `?gov=0` pins quality, for profiling */
  private readonly governed = new URLSearchParams(window.location.search).get("gov") !== "0";

  constructor(private readonly opts: EngineOptions) {
    const { canvas, spec } = opts;
    this.renderer = new WebGLRenderer({
      canvas,
      antialias: spec.antialias,
      alpha: false,
      stencil: false,
      depth: true,
      powerPreference: "high-performance",
    });
    this.renderer.outputColorSpace = SRGBColorSpace;
    this.renderer.setClearColor(VOID, 1);
    this.renderer.autoClear = false;
    this.dpr = Math.min(window.devicePixelRatio || 1, spec.dprCap);
    this.renderer.setPixelRatio(this.dpr);

    this.camera = new PerspectiveCamera(CAMERA.fov, 1, CAMERA.near, CAMERA.far);
    this.nodes = buildNetwork(24);
    this.universe = new ParticleUniverse(spec.particles, spec.textParticles, this.nodes, opts.fonts.display, spec.curl);
    this.universeScene.add(this.universe.points);
    this.network = new CameraNetwork(this.nodes, this.atlas, opts.fonts.mono, opts.mobile);
    this.network.addTo(this.universeScene);

    this.sceneRT = this.makeRT(2, 2);
    const mat = new ShaderMaterial({
      vertexShader: fullscreenVertex,
      fragmentShader: cctvFragment,
      depthTest: false,
      depthWrite: false,
      uniforms: {
        tScene: { value: this.sceneRT.texture },
        uRes: { value: new Vector2(1, 1) },
        uAmount: { value: 0 },
        uIris: { value: 1 },
        uFade: { value: 1 },
        uTime: { value: 0 },
        uFrame: { value: 0 },
        uBarrel: { value: 0.16 },
      },
    });
    const quad = new Mesh(new PlaneGeometry(2, 2), mat);
    quad.frustumCulled = false;
    const postScene = new Scene();
    postScene.add(quad);
    this.post = { scene: postScene, camera: new OrthographicCamera(-1, 1, 1, -1, 0, 1), mat, quad };

    this.governor = new Governor(
      spec.dprCap,
      (dpr) => (this.pendingDpr = dpr),
      (share) => this.universe.setShare(share),
    );
    void this.atlas.load();
  }

  private makeRT(w: number, h: number): WebGLRenderTarget {
    return new WebGLRenderTarget(w, h, { type: HalfFloatType, minFilter: LinearFilter, magFilter: LinearFilter, depthBuffer: true });
  }

  ensureTwin(): DigitalTwin {
    this.twin ??= new DigitalTwin();
    return this.twin;
  }

  /**
   * Compile every program the story will use before the visitor can scroll,
   * in parallel where KHR_parallel_shader_compile exists. The twin renders both
   * into the half-float scene target and straight to the screen, which are
   * different program variants, so both are prepared.
   */
  async precompile(): Promise<void> {
    const twin = this.ensureTwin();
    const r = this.renderer;
    r.setRenderTarget(this.sceneRT);
    const intoTarget = r.compileAsync(twin.scene, this.camera);
    r.setRenderTarget(null);
    const toScreen = r.compileAsync(twin.scene, this.camera);
    const universe = r.compileAsync(this.universeScene, this.camera);
    const post = r.compileAsync(this.post.scene, this.post.camera);
    await Promise.all([intoTarget, toScreen, universe, post]);
    this.warmDraw(twin);
    this.governor.warmup();
  }

  /**
   * ANGLE (D3D11) finishes native shader compilation lazily, at the first
   * draw for each output target. Draw every layer once, invisibly (alphas are
   * zero at boot and the frame is cleared before it is presented), so those
   * stalls happen during calibration instead of mid-scroll.
   */
  private warmDraw(twin: DigitalTwin): void {
    const touched: { o: Object3D; visible: boolean; culled: boolean }[] = [];
    const expose = (root: Object3D) =>
      root.traverse((o) => {
        touched.push({ o, visible: o.visible, culled: o.frustumCulled });
        o.visible = true;
        o.frustumCulled = false;
      });
    expose(this.universeScene);
    expose(twin.scene);
    const r = this.renderer;
    r.setRenderTarget(this.sceneRT);
    r.clear();
    r.render(twin.scene, this.camera);
    r.setRenderTarget(null);
    r.clear();
    r.render(twin.scene, this.camera);
    r.render(this.post.scene, this.post.camera);
    r.render(this.universeScene, this.camera);
    r.clear();
    for (const t of touched) {
      t.o.visible = t.visible;
      t.o.frustumCulled = t.culled;
    }
  }

  addHook(h: FrameHook): () => void {
    this.hooks.add(h);
    return () => this.hooks.delete(h);
  }

  buildText(rows: TextRowSpec[]): void {
    this.universe.buildText(rows, this.width, this.height);
  }

  setSize(w: number, h: number): void {
    this.width = w;
    this.height = h;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.resizeRT();
  }

  private resizeRT(): void {
    // the CCTV view never needs more than ~1.25x: the lens degrades it anyway
    const s = Math.min(this.dpr, 1.25);
    this.sceneRT.setSize(Math.max(2, Math.floor(this.width * s)), Math.max(2, Math.floor(this.height * s)));
    (this.post.mat.uniforms.uRes.value as Vector2).set(this.width * s, this.height * s);
  }

  setDpr(dpr: number): void {
    this.dpr = dpr;
    this.renderer.setPixelRatio(dpr);
    this.renderer.setSize(this.width, this.height, false);
    this.resizeRT();
  }

  get pixelRatio(): number {
    return this.dpr;
  }

  get viewport(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  /** World point to CSS pixels; z > 1 means behind the camera. */
  project(p: Vector3, out: { x: number; y: number; z: number }): void {
    this.v.copy(p).project(this.camera);
    out.x = (this.v.x * 0.5 + 0.5) * this.width;
    out.y = (-this.v.y * 0.5 + 0.5) * this.height;
    out.z = this.v.z;
  }

  private updateCamera(): void {
    const C = S.cam;
    const follow = C.follow;
    this.camera.position.set(C.pos.x, C.pos.y, C.pos.z);
    this.look.set(C.look.x, C.look.y, C.look.z);
    if (follow > 0 && this.twin) {
      this.twin.chasePose(S.twin.car, this.chasePos, this.chaseLook);
      this.camera.position.lerp(this.chasePos, follow);
      this.look.lerp(this.chaseLook, follow);
    }
    // a breath of hand-held drift plus pointer parallax; never while inside a CCTV view
    const sway = C.sway * (1 - S.twin.cctv);
    const t = S.time;
    this.camera.position.x += (Math.sin(t * 0.31) * 0.35 + S.pointer.x * 1.6) * sway;
    this.camera.position.y += (Math.sin(t * 0.23 + 1.3) * 0.25 + S.pointer.y * 0.9) * sway;
    this.camera.lookAt(this.look);
    if (C.roll) this.camera.rotateZ(C.roll);
    if (this.camera.fov !== C.fov) {
      this.camera.fov = C.fov;
      this.camera.updateProjectionMatrix();
    }
    this.camera.updateMatrixWorld();
  }

  /** One frame. Called by the shared ticker with the frame delta in ms. */
  frame(dtMs: number): void {
    if (this.disposed) return;
    // Never queue GPU work faster than a 240 Hz display can show it. Unthrottled
    // contexts (some embedded or headless browsers) otherwise flood the command
    // queue until a synchronous call stalls for seconds.
    const now = performance.now();
    if (now - this.lastFrame < MIN_FRAME_MS) return;
    this.lastFrame = now;
    if (this.governed) this.governor.sample(dtMs);
    this.updateCamera();
    const T = S.twin;
    const twinOn = T.alpha > 0.002 && this.twin !== null;
    const universeOn = S.universe.alpha > 0.002 || S.sleeves.alpha > 0.002 || S.links.alpha > 0.002;
    const visible = (twinOn || universeOn) && S.blackout < 0.999;

    if (!visible) {
      // resizing the drawing buffer stalls the GPU: only do it while nothing is on screen
      if (this.pendingDpr !== null) {
        this.setDpr(this.pendingDpr);
        this.pendingDpr = null;
        this.drewBlank = false;
      }
      // draw black once, then stop touching the GPU until something returns
      if (!this.drewBlank) {
        this.renderer.setRenderTarget(null);
        this.renderer.setClearColor(VOID, 1);
        this.renderer.clear();
        this.drewBlank = true;
      }
      for (const h of this.hooks) h(this);
      return;
    }
    this.drewBlank = false;
    const r = this.renderer;
    r.setRenderTarget(null);

    if (twinOn && this.twin) {
      const look = T.look;
      this.twin.update(this.camera, { look, overlays: true });
      const direct = T.cctv < 0.002 && T.iris > 0.998 && T.alpha > 0.998;
      this.clear.copy(FOG_NIGHT).lerp(FOG_DAY, look);
      if (direct) {
        r.setClearColor(this.clear, 1);
        r.clear();
        r.render(this.twin.scene, this.camera);
      } else {
        // a camera records at NVR frame rate: hold frames while fully inside a CCTV view
        const holding = T.cctv > 0.985 && S.time - this.lastCctvRender < 1 / CCTV.fps;
        if (!holding) {
          r.setRenderTarget(this.sceneRT);
          r.setClearColor(this.clear, 1);
          r.clear();
          r.render(this.twin.scene, this.camera);
          this.lastCctvRender = S.time;
        }
        const u = this.post.mat.uniforms;
        u.tScene.value = this.sceneRT.texture;
        u.uAmount.value = T.cctv;
        u.uIris.value = T.iris;
        u.uFade.value = T.alpha;
        u.uTime.value = S.time;
        u.uFrame.value = Math.floor(S.time * CCTV.fps);
        // the post pass is opaque and covers the screen: no clear needed
        r.setRenderTarget(null);
        r.render(this.post.scene, this.post.camera);
      }
    } else {
      r.setClearColor(VOID, 1);
      r.clear();
    }

    if (universeOn) {
      this.universe.update(this.camera, this.dpr);
      this.network.update(this.camera);
      r.clearDepth();
      r.render(this.universeScene, this.camera);
    } else {
      this.atlas.setPlaying(false);
    }
    for (const h of this.hooks) h(this);
  }

  /**
   * Render what a twin camera sees, through the CCTV lens, into a small
   * texture. Used for evidence cards and clarify thumbnails. `carU` places the
   * sedan for that moment; null hides it from the frame.
   */
  snapshot(pose: CameraPose, carU: number | null, target?: WebGLRenderTarget): WebGLRenderTarget {
    const twin = this.ensureTwin();
    const [w, h] = CCTV.feedSize;
    const out = target ?? new WebGLRenderTarget(w, h, { minFilter: LinearFilter, magFilter: LinearFilter });
    out.texture.colorSpace = SRGBColorSpace;
    const scratch = new WebGLRenderTarget(w, h, { type: HalfFloatType, depthBuffer: true });
    const cam = new PerspectiveCamera(48, w / h, 0.1, 600);
    cam.position.set(...pose.pos);
    cam.lookAt(new Vector3(...pose.look));
    cam.updateMatrixWorld();

    const savedCar = S.twin.car;
    const carVisible = twin.car.visible;
    twin.car.visible = carU !== null;
    S.twin.car = carU ?? savedCar;
    twin.update(cam, { look: 1, overlays: false });
    const r = this.renderer;
    r.setRenderTarget(scratch);
    r.setClearColor(FOG_DAY, 1);
    r.clear();
    r.render(twin.scene, cam);
    const u = this.post.mat.uniforms;
    const prev = { tex: u.tScene.value as Texture, amount: u.uAmount.value as number, iris: u.uIris.value as number, fade: u.uFade.value as number };
    const res = (u.uRes.value as Vector2).clone();
    u.tScene.value = scratch.texture;
    u.uAmount.value = 1;
    u.uIris.value = 1;
    u.uFade.value = 1;
    (u.uRes.value as Vector2).set(w, h);
    r.setRenderTarget(out);
    r.render(this.post.scene, this.post.camera);
    u.tScene.value = prev.tex;
    u.uAmount.value = prev.amount;
    u.uIris.value = prev.iris;
    u.uFade.value = prev.fade;
    (u.uRes.value as Vector2).copy(res);
    r.setRenderTarget(null);
    r.setClearColor(VOID, 1);
    twin.car.visible = carVisible;
    S.twin.car = savedCar;
    scratch.dispose();
    return out;
  }

  /** Copy a render target into a 2D canvas for DOM use (thumbnails), without stalling the GPU. */
  async readToCanvas(rt: WebGLRenderTarget, canvas: HTMLCanvasElement): Promise<void> {
    const { width, height } = rt;
    const px = new Uint8Array(width * height * 4);
    await this.renderer.readRenderTargetPixelsAsync(rt, 0, 0, width, height, px);
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const img = ctx.createImageData(width, height);
    // WebGL rows are bottom-up
    for (let y = 0; y < height; y++) img.data.set(px.subarray((height - 1 - y) * width * 4, (height - y) * width * 4), y * width * 4);
    ctx.putImageData(img, 0, 0);
  }

  poseOf(id: string): CameraPose {
    return twinCam(id);
  }

  warmup(): void {
    this.governor.warmup();
  }

  dispose(): void {
    this.disposed = true;
    this.universe.dispose();
    this.network.dispose();
    this.atlas.dispose();
    this.twin?.dispose();
    this.sceneRT.dispose();
    this.post.mat.dispose();
    this.post.quad.geometry.dispose();
    this.renderer.dispose();
    this.hooks.clear();
  }
}
