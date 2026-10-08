import { BufferAttribute, BufferGeometry, Vector3, type Curve } from "three";

/**
 * Flat ribbon along a curve (in the XZ plane, lifted to `y`). Carries `aT`
 * (0..1 along the curve) and `aSide` (-1..1 across) for draw-on shaders.
 */
export function ribbonGeometry(curve: Curve<Vector3>, u0: number, u1: number, width: number, y: number, segments: number): BufferGeometry {
  const verts = (segments + 1) * 2;
  const pos = new Float32Array(verts * 3);
  const t = new Float32Array(verts);
  const side = new Float32Array(verts);
  const uv = new Float32Array(verts * 2);
  const p = new Vector3();
  const tan = new Vector3();
  const nrm = new Vector3();
  for (let i = 0; i <= segments; i++) {
    const f = i / segments;
    const u = u0 + (u1 - u0) * f;
    curve.getPointAt(u, p);
    curve.getTangentAt(u, tan);
    nrm.set(-tan.z, 0, tan.x).normalize().multiplyScalar(width / 2);
    pos.set([p.x + nrm.x, y, p.z + nrm.z, p.x - nrm.x, y, p.z - nrm.z], i * 6);
    t[i * 2] = t[i * 2 + 1] = u;
    side[i * 2] = 1;
    side[i * 2 + 1] = -1;
    uv.set([f, 1, f, 0], i * 4);
  }
  const index: number[] = [];
  for (let i = 0; i < segments; i++) {
    const a = i * 2;
    index.push(a, a + 1, a + 2, a + 1, a + 3, a + 2);
  }
  const g = new BufferGeometry();
  g.setAttribute("position", new BufferAttribute(pos, 3));
  g.setAttribute("aT", new BufferAttribute(t, 1));
  g.setAttribute("aSide", new BufferAttribute(side, 1));
  g.setAttribute("uv", new BufferAttribute(uv, 2));
  g.setIndex(index);
  g.computeVertexNormals();
  return g;
}
