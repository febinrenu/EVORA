// <CameraNetwork />, <CameraNode /> and <VideoWall />: the 24 camera sleeves
// and their links. The same instanced sleeves fly from the network into the
// wall (grid -> diagonal -> curved -> solo), so the wall is not a new object,
// it is the network turning to face you.
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  CustomBlending,
  DoubleSide,
  DynamicDrawUsage,
  InstancedBufferAttribute,
  InstancedMesh,
  LineSegments,
  Matrix4,
  NormalBlending,
  OneFactor,
  PlaneGeometry,
  Quaternion,
  ShaderMaterial,
  Vector2,
  Vector3,
  type PerspectiveCamera,
  type Scene,
} from "three";
import { S } from "@/animation/sceneState";
import { CAMERA } from "@/lib/config/animation";
import { buildLinks, isTwinCam, type NetworkNode } from "@/lib/data/site";
import { mulberry32 } from "@/lib/math";
import { linkFragment, linkVertex } from "./shaders/lines";
import { sleeveFragment, sleeveVertex } from "./shaders/sleeve";
import { LABEL, LabelAtlas } from "./textures/LabelAtlas";
import type { VideoAtlas } from "./textures/VideoAtlas";

const SLEEVE_W = 7.2;
const CYAN = new Color("#86B6E0");
const AXIS_Y = new Vector3(0, 1, 0);
const AXIS_Z = new Vector3(0, 0, 1);
/** On phones the wall is 2x2 plus the solo feed (wall index 4). */
const MOBILE_SLOTS: Record<number, number> = { 0: 0, 1: 1, 3: 2, 4: 3 };

export class CameraNetwork {
  readonly sleeves: InstancedMesh;
  readonly links: LineSegments;
  readonly labels: LabelAtlas;
  /** node indices that make up the 3x3 wall; wall[4] goes solo */
  readonly wall: number[];
  private readonly sleeveMat: ShaderMaterial;
  private readonly linkMat: ShaderMaterial;
  private readonly alpha: InstancedBufferAttribute;
  private readonly hover: InstancedBufferAttribute;
  private readonly hoverState: Float32Array;
  // scratch objects: update() runs every frame and must not allocate
  private readonly m = new Matrix4();
  private readonly q = new Quaternion();
  private readonly qTmp = new Quaternion();
  private readonly qYaw = new Quaternion();
  private readonly qRoll = new Quaternion();
  private readonly camQ = new Quaternion();
  private readonly p = new Vector3();
  private readonly pTmp = new Vector3();
  private readonly acc = new Vector3();
  private readonly s = new Vector3();
  private readonly v2 = new Vector2();
  private wsum = 0;

  constructor(
    private readonly nodes: NetworkNode[],
    private readonly atlas: VideoAtlas,
    mono: string,
    private readonly mobile: boolean,
  ) {
    const n = nodes.length;
    this.labels = new LabelAtlas(nodes, mono);
    this.wall = nodes
      .map((node, i) => ({ node, i }))
      .filter(({ node }) => !isTwinCam(node.id))
      .slice(0, 9)
      .map(({ i }) => i);

    const geo = new PlaneGeometry(1, 1);
    const cells = new Float32Array(n);
    const labels = new Float32Array(n);
    const solo = new Float32Array(n);
    nodes.forEach((node, i) => {
      cells[i] = node.cell;
      labels[i] = i;
    });
    solo[this.wall[4]] = 1;
    geo.setAttribute("aCell", new InstancedBufferAttribute(cells, 1));
    geo.setAttribute("aLabel", new InstancedBufferAttribute(labels, 1));
    geo.setAttribute("aSolo", new InstancedBufferAttribute(solo, 1));
    this.alpha = new InstancedBufferAttribute(new Float32Array(n), 1);
    this.alpha.setUsage(DynamicDrawUsage);
    this.hover = new InstancedBufferAttribute(new Float32Array(n), 1);
    this.hover.setUsage(DynamicDrawUsage);
    this.hoverState = new Float32Array(n);
    geo.setAttribute("aAlpha", this.alpha);
    geo.setAttribute("aHover", this.hover);

    this.sleeveMat = new ShaderMaterial({
      vertexShader: sleeveVertex,
      fragmentShader: sleeveFragment,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      side: DoubleSide,
      blending: NormalBlending,
      uniforms: {
        uVideo: { value: atlas.texture },
        uLabels: { value: this.labels.texture },
        uHasVideo: { value: 0 },
        uTime: { value: 0 },
        uAlpha: { value: 0 },
        uCollapse: { value: 0 },
        uLabelGrid: { value: new Vector2(LABEL.cols, LABEL.rows) },
        uSoloAmt: { value: 0 },
      },
    });
    this.sleeves = new InstancedMesh(geo, this.sleeveMat, n);
    this.sleeves.instanceMatrix.setUsage(DynamicDrawUsage);
    this.sleeves.frustumCulled = false;
    this.sleeves.renderOrder = 2;

    this.linkMat = new ShaderMaterial({
      vertexShader: linkVertex,
      fragmentShader: linkFragment,
      transparent: true,
      depthWrite: false,
      depthTest: false,
      blending: CustomBlending,
      blendSrc: OneFactor,
      blendDst: OneFactor,
      uniforms: { uDraw: { value: 0 }, uAlpha: { value: 0 }, uTime: { value: 0 }, uColor: { value: CYAN } },
    });
    this.links = new LineSegments(this.buildLinkGeometry(), this.linkMat);
    this.links.frustumCulled = false;
    this.links.renderOrder = 1;
  }

  /** Quadratic arcs between linked nodes, each split into short segments carrying their position along the link. */
  private buildLinkGeometry(): BufferGeometry {
    const links = buildLinks(this.nodes);
    const SEG = 28;
    const verts = links.length * SEG * 2;
    const pos = new Float32Array(verts * 3);
    const t = new Float32Array(verts);
    const id = new Float32Array(verts);
    const rand = mulberry32(77);
    const a = new Vector3();
    const b = new Vector3();
    const c = new Vector3();
    const point = (u: number, out: Vector3) => {
      const k = 1 - u;
      return out.set(
        k * k * a.x + 2 * k * u * c.x + u * u * b.x,
        k * k * a.y + 2 * k * u * c.y + u * u * b.y,
        k * k * a.z + 2 * k * u * c.z + u * u * b.z,
      );
    };
    const p0 = new Vector3();
    const p1 = new Vector3();
    let o = 0;
    for (const [i, j] of links) {
      a.fromArray(this.nodes[i].pos);
      b.fromArray(this.nodes[j].pos);
      c.addVectors(a, b).multiplyScalar(0.5);
      c.y += a.distanceTo(b) * 0.28;
      const r = rand();
      for (let s = 0; s < SEG; s++) {
        point(s / SEG, p0);
        point((s + 1) / SEG, p1);
        pos.set([p0.x, p0.y, p0.z, p1.x, p1.y, p1.z], o * 3);
        t[o] = s / SEG;
        t[o + 1] = (s + 1) / SEG;
        id[o] = id[o + 1] = r;
        o += 2;
      }
    }
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(pos, 3));
    g.setAttribute("aT", new BufferAttribute(t, 1));
    g.setAttribute("aLink", new BufferAttribute(id, 1));
    return g;
  }

  addTo(scene: Scene): void {
    scene.add(this.links, this.sleeves);
  }

  /** Lay out every sleeve for the current blend of network, grid, diagonal, curve and solo. */
  update(camera: PerspectiveCamera): void {
    const L = S.sleeves;
    this.links.visible = S.links.alpha > 0.002;
    this.sleeves.visible = L.alpha > 0.002;
    this.atlas.setPlaying(this.sleeves.visible);
    if (this.links.visible) {
      const lu = this.linkMat.uniforms;
      lu.uDraw.value = S.links.draw;
      lu.uAlpha.value = S.links.alpha;
      lu.uTime.value = S.time;
    }
    if (!this.sleeves.visible) return;

    const u = this.sleeveMat.uniforms;
    u.uTime.value = S.time;
    u.uAlpha.value = L.alpha;
    u.uCollapse.value = L.collapse;
    u.uSoloAmt.value = L.wSolo;
    u.uHasVideo.value = this.atlas.ready ? 1 : 0;
    u.uVideo.value = this.atlas.texture;
    this.labels.tick(9 * 3600 + 14 * 60 + S.time);

    const viewH = 2 * CAMERA.hudDepth * Math.tan((camera.fov * Math.PI) / 360);
    const viewW = viewH * camera.aspect;
    camera.getWorldQuaternion(this.camQ);
    const cols = this.mobile ? 2 : 3;
    const rows = this.mobile ? 2 : 3;
    const cellW = viewW * (this.mobile ? 0.44 : 0.3);
    const cellH = cellW * (9 / 16);
    const D = CAMERA.hudDepth;
    const wallAmount = 1 - L.wNetwork;
    const soloCover = Math.max(viewW, viewH * (16 / 9)) * 1.01;
    const alphas = this.alpha.array as Float32Array;

    for (let i = 0; i < this.nodes.length; i++) {
      const k = this.wall.indexOf(i);
      const slot = k < 0 ? -1 : this.mobile ? (MOBILE_SLOTS[k] ?? -1) : k;
      const inWall = slot >= 0;
      const hover = 1 + this.hoverState[i] * 0.12;
      this.p.fromArray(this.nodes[i].pos);
      this.q.copy(this.camQ);
      let sx = SLEEVE_W * hover;
      let alpha = inWall ? 1 : 1 - wallAmount;
      // a sleeve the camera is about to pass through fades instead of filling the frame
      const near = Math.min(1, Math.max(0, (this.p.distanceTo(camera.position) - 6) / 12));
      alpha *= near + (1 - near) * wallAmount;

      if (inWall && wallAmount > 0) {
        const cx = (slot % cols) - (cols - 1) / 2;
        const cy = (rows - 1) / 2 - Math.floor(slot / cols);
        const isSolo = k === 4;
        this.acc.set(0, 0, 0);
        this.q.set(0, 0, 0, 0);
        this.wsum = 0;
        this.blendWorld(L.wNetwork, this.p, this.camQ);
        this.blendLocal(camera, L.wGrid, cx * cellW * 1.06, cy * cellH * 1.06, -D, 0, 0);
        // diagonal: sheared and receding to the right
        this.blendLocal(camera, L.wDiagonal, cx * cellW + cy * cellW * 0.32, cy * cellH - cx * cellH * 0.18, -D - (cx - cy) * 9, -0.32, 0.08);
        // curved wall wrapping around the viewer
        const a = cx * 0.46;
        this.blendLocal(camera, L.wCurve, Math.sin(a) * D * 0.9, cy * cellH * 1.08, -Math.cos(a) * D * 0.9 + 2, -a, 0);
        // solo: the centre feed fills the frame, the others are pushed past the viewer
        if (isSolo) this.blendLocal(camera, L.wSolo, 0, 0, -D, 0, 0);
        else this.blendLocal(camera, L.wSolo, cx * cellW * 3.2, cy * cellH * 3.2, -D + 18, 0, 0);
        this.p.copy(this.acc).multiplyScalar(1 / Math.max(this.wsum, 1e-6));
        this.q.normalize();
        // network sleeves are world-sized, wall sleeves are screen-sized
        const targetW = isSolo ? cellW + (soloCover - cellW) * L.wSolo : cellW;
        sx = SLEEVE_W + (targetW - SLEEVE_W) * Math.min(1, wallAmount * 1.2);
        if (!isSolo) alpha *= 1 - L.wSolo;
      }
      this.s.set(sx, sx * (9 / 16), 1);
      this.m.compose(this.p, this.q, this.s);
      this.sleeves.setMatrixAt(i, this.m);
      alphas[i] = alpha;
    }
    this.sleeves.instanceMatrix.needsUpdate = true;
    this.alpha.needsUpdate = true;
    this.updateHover(camera);
  }

  private blendLocal(camera: PerspectiveCamera, w: number, x: number, y: number, z: number, yaw: number, roll: number): void {
    if (w <= 0) return;
    this.pTmp.set(x, y, z).applyMatrix4(camera.matrixWorld);
    this.qYaw.setFromAxisAngle(AXIS_Y, yaw);
    this.qRoll.setFromAxisAngle(AXIS_Z, roll);
    this.qTmp.copy(this.camQ).multiply(this.qYaw).multiply(this.qRoll);
    this.blendWorld(w, this.pTmp, this.qTmp);
  }

  private blendWorld(w: number, pos: Vector3, rot: Quaternion): void {
    if (w <= 0) return;
    this.acc.addScaledVector(pos, w);
    // nlerp: keep every quaternion in the same hemisphere before summing
    const sign = this.q.x * rot.x + this.q.y * rot.y + this.q.z * rot.z + this.q.w * rot.w < 0 ? -w : w;
    this.q.set(this.q.x + rot.x * sign, this.q.y + rot.y * sign, this.q.z + rot.z * sign, this.q.w + rot.w * sign);
    this.wsum += w;
  }

  /** Nearest sleeve to the pointer lifts slightly; computed on 24 projected centres, no raycasting. */
  private updateHover(camera: PerspectiveCamera): void {
    const arr = this.hover.array as Float32Array;
    const active = S.sleeves.wNetwork > 0.6 && S.pointer.active > 0.5;
    let best = -1;
    let bestD = 0.12;
    if (active) {
      for (let i = 0; i < this.nodes.length; i++) {
        this.pTmp.fromArray(this.nodes[i].pos).project(camera);
        if (this.pTmp.z > 1) continue;
        this.v2.set((this.pTmp.x - S.pointer.x) * camera.aspect, this.pTmp.y - S.pointer.y);
        const d = this.v2.length();
        if (d < bestD) {
          bestD = d;
          best = i;
        }
      }
    }
    for (let i = 0; i < arr.length; i++) {
      this.hoverState[i] += ((i === best ? 1 : 0) - this.hoverState[i]) * 0.12;
      arr[i] = this.hoverState[i];
    }
    this.hover.needsUpdate = true;
  }

  dispose(): void {
    this.sleeves.geometry.dispose();
    this.sleeveMat.dispose();
    this.links.geometry.dispose();
    this.linkMat.dispose();
    this.labels.dispose();
  }
}
