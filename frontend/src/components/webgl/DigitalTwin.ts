// The digital twin: a procedural campus the size of the story. It is rendered
// two ways from the same scene graph: as a night massing model (memory map,
// reconstruction) and, through the CCTV lens, as the footage each camera
// recorded (evidence, trace). <EntityPath />, <MemoryMap /> and
// <ReconstructionScene /> are layers of this one scene.
import {
  BoxGeometry,
  BufferAttribute,
  BufferGeometry,
  CapsuleGeometry,
  Color,
  CustomBlending,
  CylinderGeometry,
  DoubleSide,
  EdgesGeometry,
  Group,
  IcosahedronGeometry,
  InstancedMesh,
  LineSegments,
  Matrix4,
  Mesh,
  OctahedronGeometry,
  OneFactor,
  PlaneGeometry,
  Quaternion,
  Scene,
  ShaderMaterial,
  SphereGeometry,
  Vector3,
  type BufferGeometry as Geo,
  type Material,
  type Object3D,
  type PerspectiveCamera,
  type Texture,
} from "three";
import { S } from "@/animation/sceneState";
import { CAR_ROUTE, HOP_U, PERSON_ROUTE, TWIN_CAMS, twinCam } from "@/lib/data/site";
import { ENTITIES, PLACES, RECON_CAMS } from "@/lib/data/story";
import { bake } from "./bake";
import { ribbonGeometry } from "./geometry";
import { ribbonFragment, ribbonVertex } from "./shaders/lines";
import { beaconFragment, beaconVertex, cardFragment, cardVertex, edgeFragment, edgeVertex, massFragment, massVertex } from "./shaders/twin";

const BONE = new Color("#DCE3E7");
const CYAN = new Color("#86B6E0");
const RED = new Color("#F05A66");
const MARKER = new Color("#E9B308");
export const FOG_NIGHT = new Color("#06080A");
export const FOG_DAY = new Color("#8E9599");

interface Shared {
  uLook: { value: number };
}

export class DigitalTwin {
  readonly scene = new Scene();
  readonly car = new Group();
  private readonly shared: Shared = { uLook: { value: 0 } };
  private readonly disposables: (Geo | Material)[] = [];
  private readonly person = new Group();
  private ghost: Mesh | null = null;
  private ghostMat: ShaderMaterial | null = null;
  private readonly ribbonMat: ShaderMaterial;
  private readonly beaconMats: ShaderMaterial[] = [];
  private readonly rings: Mesh[] = [];
  private readonly frustumMat: ShaderMaterial;
  private readonly markers = new Group();
  private readonly markerMat: ShaderMaterial;
  private readonly cards: Mesh[] = [];
  private readonly cardMats: ShaderMaterial[] = [];
  private readonly leaderMat: ShaderMaterial;
  private readonly overlays = new Group();
  private readonly tmp = new Vector3();
  private readonly tan = new Vector3();

  constructor() {
    this.buildGround();
    this.buildBuildings();
    this.buildGate();
    this.buildParking();
    this.buildTrees();
    this.buildPoles();
    this.bakeSite();
    this.buildCar(this.car, this.mass("#B3121F"), this.mass("#14181B"), this.mass("#0B0D0F"));
    this.bakeGroup(this.car, this.mass("#FFFFFF", { emissive: 0.55, vertex: true }));
    this.scene.add(this.car);
    this.buildPerson();
    this.bakeGroup(this.person, this.mass("#FFFFFF", { vertex: true }));
    this.buildGhosts();

    this.ribbonMat = this.additive(ribbonVertex, ribbonFragment, { uDraw: { value: 0 }, uAlpha: { value: 0 }, uTime: { value: 0 }, uColor: { value: RED } });
    // the trace starts where EVORA first saw the car: the main gate
    const ribbon = new Mesh(ribbonGeometry(CAR_ROUTE, Math.max(0, HOP_U.CAM_04 - 0.03), 1, 0.55, 0.14, 500), this.ribbonMat);
    ribbon.renderOrder = 5;
    this.overlays.add(ribbon);
    this.disposables.push(ribbon.geometry);

    this.frustumMat = this.lineMat(CYAN);
    this.overlays.add(this.buildFrustums());
    this.buildBeacons();
    this.markerMat = this.additive(beaconVertex, /* glsl */ `
      precision highp float; uniform vec3 uColor; uniform float uAlpha; varying vec2 vUv;
      void main(){ gl_FragColor = vec4(uColor * uAlpha, uAlpha); }`, { uColor: { value: BONE }, uAlpha: { value: 0 } });
    this.buildMarkers();
    this.leaderMat = this.lineMat(BONE);
    this.buildCards();
    this.scene.add(this.overlays);
  }

  // ---------- material helpers
  private mass(hex: string, opts: { grid?: boolean; emissive?: number; vertex?: boolean } = {}): ShaderMaterial {
    const m = new ShaderMaterial({
      vertexShader: massVertex,
      fragmentShader: massFragment,
      vertexColors: opts.vertex ?? false,
      uniforms: {
        uColor: { value: new Color(hex) },
        uLook: this.shared.uLook,
        uGrid: { value: opts.grid ? 1 : 0 },
        uAlpha: { value: 1 },
        uEmissive: { value: opts.emissive ?? 0 },
        uFogNight: { value: FOG_NIGHT },
        uFogDay: { value: FOG_DAY },
      },
    });
    this.disposables.push(m);
    return m;
  }

  private additive(vs: string, fs: string, uniforms: Record<string, { value: unknown }>): ShaderMaterial {
    const m = new ShaderMaterial({
      vertexShader: vs,
      fragmentShader: fs,
      uniforms,
      transparent: true,
      depthWrite: false,
      side: DoubleSide,
      blending: CustomBlending,
      blendSrc: OneFactor,
      blendDst: OneFactor,
    });
    this.disposables.push(m);
    return m;
  }

  private lineMat(color: Color): ShaderMaterial {
    return this.additive(edgeVertex, /* glsl */ `
      precision highp float; uniform vec3 uColor; uniform float uAlpha; varying float vDepth;
      void main(){ float a = uAlpha * smoothstep(360.0, 30.0, vDepth); gl_FragColor = vec4(uColor * a, a); }`, {
      uColor: { value: color },
      uAlpha: { value: 0 },
    });
  }

  private block(geo: Geo, mat: Material, x: number, y: number, z: number, edges = true, parent: Object3D = this.scene): Mesh {
    const mesh = new Mesh(geo, mat);
    mesh.position.set(x, y, z);
    parent.add(mesh);
    this.disposables.push(geo);
    if (edges) {
      const eg = new EdgesGeometry(geo, 20);
      mesh.add(new LineSegments(eg, this.edgeMat));
      this.disposables.push(eg);
    }
    return mesh;
  }

  /** One edge material for the whole model: bright at night, nearly gone in daylight. */
  private readonly edgeMat = new ShaderMaterial({
    vertexShader: edgeVertex,
    fragmentShader: edgeFragment,
    uniforms: { uColor: { value: CYAN }, uLook: this.shared.uLook, uAlpha: { value: 1 } },
    transparent: true,
    depthWrite: false,
  });

  /** Merge everything built so far (except the gridded ground) into two draw calls. */
  private bakeSite(): void {
    const statics = this.scene.children.filter((o) => !o.userData.keep);
    const { solid, edges, spent } = bake(statics, this.mass("#FFFFFF", { vertex: true }), this.edgeMat);
    if (solid) this.scene.add(solid);
    if (edges) this.scene.add(edges);
    this.retire(spent);
  }

  /** Merge a moving group's parts into one mesh that keeps the group's transform. */
  private bakeGroup(group: Group, material: ShaderMaterial): void {
    const parts = [...group.children];
    const saved = group.matrix.clone();
    group.position.set(0, 0, 0);
    group.rotation.set(0, 0, 0);
    group.updateMatrixWorld(true);
    const { solid, spent } = bake(parts, material, this.edgeMat);
    if (solid) group.add(solid);
    saved.decompose(group.position, group.quaternion, group.scale);
    this.retire(spent);
  }

  private retire(spent: BufferGeometry[]): void {
    for (const g of spent) {
      g.dispose();
      const i = this.disposables.indexOf(g);
      if (i >= 0) this.disposables.splice(i, 1);
    }
  }

  // ---------- site
  private buildGround(): void {
    const g = new PlaneGeometry(520, 380);
    g.rotateX(-Math.PI / 2);
    this.block(g, this.mass("#5E6467", { grid: true }), 6, 0, -10, false).userData.keep = true;
    const asphalt = this.mass("#2A2F33");
    this.block(new BoxGeometry(320, 0.06, 12), asphalt, 0, 0.03, 45, false);
    const drive = new Mesh(ribbonGeometry(CAR_ROUTE, 0.17, 1, 8, 0.05, 220), asphalt);
    this.scene.add(drive);
    this.disposables.push(drive.geometry);
    // dashed centre line on the public road
    const dash = this.mass("#C9CDCF");
    const dashGeo = new BoxGeometry(3, 0.02, 0.22);
    const dashes = new InstancedMesh(dashGeo, dash, 40);
    const m = new Matrix4();
    for (let i = 0; i < 40; i++) dashes.setMatrixAt(i, m.makeTranslation(-156 + i * 8, 0.07, 45));
    this.scene.add(dashes);
    this.disposables.push(dashGeo);
  }

  private buildBuildings(): void {
    const concrete = this.mass("#A9AEB0");
    const warm = this.mass("#B3ADA4");
    const dark = this.mass("#1A1E21");
    const b = (w: number, h: number, d: number, x: number, z: number, mat = concrete) => this.block(new BoxGeometry(w, h, d), mat, x, h / 2, z);
    b(26, 9, 20, -28, -14);
    b(14, 4, 6, -20, -1.5, warm);
    b(36, 13, 22, 46, -62);
    b(18, 8, 30, 74, -20, warm);
    b(30, 16, 26, -74, -42);
    b(22, 10, 18, -58, -78, warm);
    b(26, 7, 16, 8, -86);
    b(3.2, 3, 3.2, -11, 29, warm);
    // rear entrance: door recess and canopy on the south face of the rear block
    this.block(new BoxGeometry(5, 3.8, 0.5), dark, 40, 1.9, -50.8, false);
    this.block(new BoxGeometry(8, 0.35, 3.2), concrete, 40, 4.4, -49.6);
    // loading bay doors on the west face
    for (let i = 0; i < 3; i++) this.block(new BoxGeometry(0.5, 4.2, 5), dark, 64.8, 2.1, -30 + i * 8, false);
    // lobby glazing
    this.block(new BoxGeometry(10, 3.4, 0.4), dark, -24, 1.7, -3.9, false);
  }

  private buildGate(): void {
    const stone = this.mass("#9EA3A5");
    this.block(new BoxGeometry(1.5, 4.4, 1.5), stone, -6.4, 2.2, 34);
    this.block(new BoxGeometry(1.5, 4.4, 1.5), stone, 6.4, 2.2, 34);
    // raised barrier arm, pivoting at the left pillar
    const armGeo = new BoxGeometry(9, 0.22, 0.22);
    armGeo.translate(4.5, 0, 0);
    const arm = this.block(armGeo, this.mass("#D9DCDD"), -5.6, 1.15, 34.9);
    arm.rotation.z = 1.25;
    const fence = this.mass("#4A5054");
    this.block(new BoxGeometry(114, 1.8, 0.16), fence, -64, 0.9, 34);
    this.block(new BoxGeometry(114, 1.8, 0.16), fence, 64, 0.9, 34);
  }

  private buildParking(): void {
    this.block(new BoxGeometry(46, 0.05, 28), this.mass("#3A3F43"), -38, 0.03, 20, false);
    const lineGeo = new BoxGeometry(0.16, 0.02, 5.4);
    const lines = new InstancedMesh(lineGeo, this.mass("#BFC4C6"), 24);
    const m = new Matrix4();
    for (let i = 0; i < 24; i++) lines.setMatrixAt(i, m.makeTranslation(-58 + (i % 12) * 3.6, 0.07, i < 12 ? 11 : 28));
    this.scene.add(lines);
    this.disposables.push(lineGeo);
    const grey = ["#6F7579", "#8C9295", "#505659", "#9AA0A3", "#5B6266"];
    const slots = [0, 1, 3, 4, 6, 9, 10, 13, 15, 17, 20, 22];
    slots.forEach((s, k) => {
      const g = new Group();
      const x = -56.2 + (s % 12) * 3.6;
      const z = s < 12 ? 11 : 28;
      g.position.set(x, 0, z);
      this.buildCar(g, this.mass(grey[k % grey.length]), this.mass("#15191C"), this.mass("#0B0D0F"));
      this.scene.add(g);
    });
    // the black SUV from the memory map, parked in bay 12
    const suv = new Group();
    const [sx, , sz] = ENTITIES.find((e) => e.id === "black-suv")?.pos ?? [-40, 0, 24];
    suv.position.set(sx, 0, sz);
    suv.scale.set(1.08, 1.22, 1.08);
    this.buildCar(suv, this.mass("#121416"), this.mass("#07090A"), this.mass("#050607"));
    this.scene.add(suv);
  }

  private buildTrees(): void {
    const crownGeo = new IcosahedronGeometry(2.4, 0);
    const trunkGeo = new BoxGeometry(0.35, 2.4, 0.35);
    const n = 46;
    const crowns = new InstancedMesh(crownGeo, this.mass("#55605A"), n);
    const trunks = new InstancedMesh(trunkGeo, this.mass("#3B3A37"), n);
    const m = new Matrix4();
    const q = new Quaternion();
    const s = new Vector3();
    const p = new Vector3();
    for (let i = 0; i < n; i++) {
      const side = i % 2 === 0 ? 1 : -1;
      const x = side * (16 + ((i * 37) % 100));
      const z = 30 - ((i * 13) % 9);
      const sc = 0.8 + ((i * 7) % 5) * 0.12;
      p.set(x, 3.8 * sc, z);
      s.set(sc, sc * 1.15, sc);
      q.setFromAxisAngle(new Vector3(0, 1, 0), i);
      crowns.setMatrixAt(i, m.compose(p, q, s));
      trunks.setMatrixAt(i, m.makeTranslation(x, 1.2, z));
    }
    this.scene.add(crowns, trunks);
    this.disposables.push(crownGeo, trunkGeo);
  }

  private buildPoles(): void {
    const pole = this.mass("#3E4448");
    const housing = this.mass("#C9CDCF");
    for (const c of TWIN_CAMS) {
      const [x, y, z] = c.pos;
      this.block(new BoxGeometry(0.22, y, 0.22), pole, x, y / 2, z, false);
      const h = this.block(new BoxGeometry(0.5, 0.42, 1.1), housing, x, y + 0.1, z, false);
      h.lookAt(new Vector3(...c.look));
    }
  }

  /** A sedan from boxes: body, glasshouse, wheels. Parented to `g`, nose along +z. */
  private buildCar(g: Group, body: Material, glass: Material, tyre: Material): void {
    const bodyGeo = new BoxGeometry(1.95, 0.85, 4.6);
    const b = new Mesh(bodyGeo, body);
    b.position.y = 0.72;
    const cabinGeo = new BoxGeometry(1.7, 0.72, 2.45);
    const c = new Mesh(cabinGeo, glass);
    c.position.set(0, 1.48, -0.2);
    const roofGeo = new BoxGeometry(1.62, 0.08, 2.1);
    const r = new Mesh(roofGeo, body);
    r.position.set(0, 1.86, -0.25);
    g.add(b, c, r);
    const wheelGeo = new CylinderGeometry(0.36, 0.36, 0.28, 12);
    wheelGeo.rotateZ(Math.PI / 2);
    for (const [wx, wz] of [
      [-0.92, 1.45],
      [0.92, 1.45],
      [-0.92, -1.45],
      [0.92, -1.45],
    ]) {
      const w = new Mesh(wheelGeo, tyre);
      w.position.set(wx, 0.36, wz);
      g.add(w);
    }
    this.disposables.push(bodyGeo, cabinGeo, roofGeo, wheelGeo);
  }

  private buildPerson(): void {
    const bodyGeo = new CapsuleGeometry(0.3, 1.0, 4, 10);
    const b = new Mesh(bodyGeo, this.mass("#2D4F8C"));
    b.position.y = 0.95;
    const headGeo = new SphereGeometry(0.2, 12, 10);
    const h = new Mesh(headGeo, this.mass("#8E8780"));
    h.position.y = 1.78;
    const packGeo = new BoxGeometry(0.42, 0.5, 0.22);
    const p = new Mesh(packGeo, this.mass("#1B1E21"));
    p.position.set(0, 1.15, -0.3);
    this.person.add(b, h, p);
    this.scene.add(this.person);
    this.disposables.push(bodyGeo, headGeo, packGeo);
  }

  /** Onion-skin copies of the sedan at each sighting, baked into one translucent mesh. */
  private buildGhosts(): void {
    const copies = [HOP_U.CAM_04, HOP_U.CAM_07, HOP_U.CAM_12].map((u) => {
      const g = new Group();
      this.buildCar(g, this.mass("#FFFFFF"), this.mass("#FFFFFF"), this.mass("#FFFFFF"));
      this.placeOnRoute(g, u);
      return g;
    });
    this.ghostMat = this.additive(massVertex, /* glsl */ `
      precision highp float; uniform vec3 uColor; uniform float uAlpha; varying vec3 vNormal;
      void main(){ float f = 0.35 + 0.65 * pow(1.0 - abs(normalize(vNormal).y), 2.0); gl_FragColor = vec4(uColor * f * uAlpha, uAlpha); }`, {
      uColor: { value: RED },
      uAlpha: { value: 0 },
    });
    const { solid, spent } = bake(copies, this.ghostMat, this.edgeMat);
    this.retire(spent);
    if (solid) {
      solid.visible = false;
      this.ghost = solid;
      this.overlays.add(solid);
    }
  }

  private buildFrustums(): LineSegments {
    const pts: number[] = [];
    const f = new Vector3();
    const right = new Vector3();
    const up = new Vector3();
    for (const c of TWIN_CAMS) {
      const o = new Vector3(...c.pos);
      f.set(...c.look).sub(o).normalize();
      right.crossVectors(f, new Vector3(0, 1, 0)).normalize();
      up.crossVectors(right, f).normalize();
      const dist = 16;
      const h = Math.tan((48 * Math.PI) / 360) * dist;
      const w = h * (16 / 9);
      const corners = [
        [-1, -1],
        [1, -1],
        [1, 1],
        [-1, 1],
      ].map(([a, b]) => o.clone().addScaledVector(f, dist).addScaledVector(right, a * w).addScaledVector(up, b * h));
      for (let i = 0; i < 4; i++) {
        pts.push(...o.toArray(), ...corners[i].toArray());
        pts.push(...corners[i].toArray(), ...corners[(i + 1) % 4].toArray());
      }
    }
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(new Float32Array(pts), 3));
    this.disposables.push(g);
    return new LineSegments(g, this.frustumMat);
  }

  private buildBeacons(): void {
    const colGeo = new CylinderGeometry(1.1, 1.1, 46, 24, 1, true);
    colGeo.translate(0, 23, 0);
    const ringGeo = new PlaneGeometry(8, 8);
    ringGeo.rotateX(-Math.PI / 2);
    this.disposables.push(colGeo, ringGeo);
    PLACES.forEach((place, i) => {
      const colMat = this.additive(beaconVertex, beaconFragment, { uColor: { value: MARKER }, uAlpha: { value: 0 }, uTime: { value: 0 }, uPulse: { value: 0 }, uRing: { value: 0 } });
      const ringMat = this.additive(beaconVertex, beaconFragment, { uColor: { value: MARKER }, uAlpha: { value: 0 }, uTime: { value: 0 }, uPulse: { value: i * 0.21 }, uRing: { value: 1 } });
      const col = new Mesh(colGeo, colMat);
      const ring = new Mesh(ringGeo, ringMat);
      col.position.set(place.pos[0], 0, place.pos[2]);
      ring.position.set(place.pos[0], 0.2, place.pos[2]);
      col.userData.place = place.id;
      ring.userData.phase = i * 0.21;
      this.beaconMats.push(colMat, ringMat);
      this.rings.push(ring);
      this.overlays.add(col, ring);
    });
  }

  private buildMarkers(): void {
    const geo = new OctahedronGeometry(0.9, 0);
    this.disposables.push(geo);
    const stem: number[] = [];
    for (const e of ENTITIES) {
      const m = new Mesh(geo, this.markerMat);
      m.position.set(e.pos[0], 9, e.pos[2]);
      this.markers.add(m);
      stem.push(e.pos[0], 0.2, e.pos[2], e.pos[0], 8.1, e.pos[2]);
    }
    const sg = new BufferGeometry();
    sg.setAttribute("position", new BufferAttribute(new Float32Array(stem), 3));
    this.disposables.push(sg);
    this.markers.add(new LineSegments(sg, this.markerMat));
    this.overlays.add(this.markers);
  }

  private buildCards(): void {
    const geo = new PlaneGeometry(16 * 1.5, 9 * 1.5);
    this.disposables.push(geo);
    const leader: number[] = [];
    for (const id of RECON_CAMS) {
      const c = twinCam(id);
      const mat = new ShaderMaterial({
        vertexShader: cardVertex,
        fragmentShader: cardFragment,
        uniforms: { uMap: { value: null }, uAlpha: { value: 0 }, uTime: { value: 0 }, uEdge: { value: BONE } },
        transparent: true,
        depthWrite: false,
        side: DoubleSide,
      });
      this.disposables.push(mat);
      const card = new Mesh(geo, mat);
      card.position.set(c.pos[0], 30, c.pos[2]);
      card.renderOrder = 8;
      this.cards.push(card);
      this.cardMats.push(mat);
      this.overlays.add(card);
      leader.push(c.pos[0], c.pos[1] + 0.6, c.pos[2], c.pos[0], 30 - 6.9, c.pos[2]);
    }
    const lg = new BufferGeometry();
    lg.setAttribute("position", new BufferAttribute(new Float32Array(leader), 3));
    this.disposables.push(lg);
    this.overlays.add(new LineSegments(lg, this.leaderMat));
  }

  /** Feed snapshots for the reconstruction cards. */
  setCardTextures(textures: Texture[]): void {
    textures.forEach((t, i) => {
      if (this.cardMats[i]) this.cardMats[i].uniforms.uMap.value = t;
    });
  }

  // ---------- per-frame
  placeOnRoute(obj: Object3D, u: number, curve = CAR_ROUTE): void {
    const uu = Math.min(0.9999, Math.max(0, u));
    curve.getPointAt(uu, this.tmp);
    curve.getTangentAt(uu, this.tan);
    obj.position.copy(this.tmp);
    obj.rotation.set(0, Math.atan2(this.tan.x, this.tan.z), 0);
  }

  /** Chase camera behind the car along the route, for the cross-camera trace. */
  chasePose(u: number, outPos: Vector3, outLook: Vector3): void {
    const uu = Math.min(0.999, Math.max(0, u));
    CAR_ROUTE.getPointAt(uu, this.tmp);
    CAR_ROUTE.getTangentAt(uu, this.tan);
    outPos.copy(this.tmp).addScaledVector(this.tan, -24);
    outPos.x += -this.tan.z * 7;
    outPos.y += 12;
    outPos.z += this.tan.x * 7;
    outLook.copy(this.tmp).addScaledVector(this.tan, 16).setY(0.8);
  }

  update(camera: PerspectiveCamera, opts: { look: number; overlays: boolean }): void {
    const T = S.twin;
    this.shared.uLook.value = opts.look;
    this.placeOnRoute(this.car, T.car);
    // the person loops slowly whenever they are on screen
    const pu = T.person > 0 ? T.person : (S.time * 0.012) % 1;
    this.placeOnRoute(this.person, pu, PERSON_ROUTE);

    this.overlays.visible = opts.overlays;
    if (!opts.overlays) return;
    const t = S.time;
    this.ribbonMat.uniforms.uDraw.value = T.trace;
    this.ribbonMat.uniforms.uAlpha.value = T.traceAlpha * (1 - S.twin.cctv);
    this.ribbonMat.uniforms.uTime.value = t;
    this.frustumMat.uniforms.uAlpha.value = T.beacons * 0.55;
    this.markerMat.uniforms.uAlpha.value = T.beacons * 0.9;
    this.markers.children.forEach((m, i) => {
      if (m instanceof Mesh) {
        m.rotation.y = t * 0.6 + i;
        m.position.y = 9 + Math.sin(t * 1.3 + i) * 0.4;
      }
    });
    PLACES.forEach((place, i) => {
      const isGate = place.id === "main-gate";
      const a = isGate ? Math.max(T.gate, T.beacons) * (1 + T.gate * 0.6) : T.beacons;
      const col = this.beaconMats[i * 2];
      const ring = this.beaconMats[i * 2 + 1];
      col.uniforms.uAlpha.value = a * 0.55;
      const wave = (t * (isGate ? 0.55 : 0.32) + (this.rings[i].userData.phase as number)) % 1;
      ring.uniforms.uPulse.value = wave;
      ring.uniforms.uAlpha.value = a;
      this.rings[i].scale.setScalar(0.6 + wave * (isGate ? 3.4 : 2.2));
    });
    if (this.ghost && this.ghostMat) {
      this.ghost.visible = T.ghosts > 0.002;
      this.ghostMat.uniforms.uAlpha.value = T.ghosts * 0.28;
    }
    camera.getWorldQuaternion(QTMP);
    this.cards.forEach((c, i) => {
      c.visible = T.recon > 0.002;
      c.quaternion.copy(QTMP);
      c.position.y = 30 + (1 - T.recon) * -8 + Math.sin(t * 0.8 + i) * 0.3;
      this.cardMats[i].uniforms.uAlpha.value = T.recon;
      this.cardMats[i].uniforms.uTime.value = t;
    });
    this.leaderMat.uniforms.uAlpha.value = T.recon * 0.6;
  }

  /** Hide everything that only exists for the viewer (beacons, trace, cards) when rendering a camera's own view. */
  setOverlays(on: boolean): void {
    this.overlays.visible = on;
  }

  dispose(): void {
    for (const d of this.disposables) d.dispose();
    this.disposables.length = 0;
    this.edgeMat.dispose();
    this.scene.traverse((o) => {
      if (o instanceof Mesh || o instanceof LineSegments) o.geometry.dispose();
    });
  }
}

const QTMP = new Quaternion();
