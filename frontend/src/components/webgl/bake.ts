// Static batching for the twin. Built as many small meshes for readability,
// then baked into one solid mesh (albedo per vertex) and one edge mesh, so the
// whole campus costs two draw calls instead of ~150.
import {
  BufferAttribute,
  BufferGeometry,
  Color,
  InstancedMesh,
  LineSegments,
  Matrix4,
  Mesh,
  ShaderMaterial,
  type Material,
  type Object3D,
} from "three";
import { mergeGeometries } from "three/examples/jsm/utils/BufferGeometryUtils.js";

const tmp = new Matrix4();

function solidPiece(src: BufferGeometry, world: Matrix4, color: Color): BufferGeometry {
  const g = (src.index ? src.toNonIndexed() : src.clone()) as BufferGeometry;
  for (const name of Object.keys(g.attributes)) if (name !== "position" && name !== "normal") g.deleteAttribute(name);
  if (!g.attributes.normal) g.computeVertexNormals();
  g.applyMatrix4(world);
  const n = g.attributes.position.count;
  const c = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) c.set([color.r, color.g, color.b], i * 3);
  g.setAttribute("color", new BufferAttribute(c, 3));
  return g;
}

function edgePiece(src: BufferGeometry, world: Matrix4): BufferGeometry {
  const g = (src.index ? src.toNonIndexed() : src.clone()) as BufferGeometry;
  for (const name of Object.keys(g.attributes)) if (name !== "position") g.deleteAttribute(name);
  g.applyMatrix4(world);
  return g;
}

const isMass = (m: Material | Material[]): m is ShaderMaterial =>
  !Array.isArray(m) && m instanceof ShaderMaterial && "uColor" in m.uniforms && "uGrid" in m.uniforms && m.uniforms.uGrid.value === 0 && !m.transparent;

const isEdge = (m: Material | Material[]): m is ShaderMaterial => !Array.isArray(m) && m instanceof ShaderMaterial && "uLook" in m.uniforms && !("uGrid" in m.uniforms);

/**
 * Merge every static solid and edge under `objects` into `solidMat` / `edgeMat`
 * batches. Originals are detached; their geometries are returned for disposal.
 */
export function bake(objects: Object3D[], solidMat: ShaderMaterial, edgeMat: ShaderMaterial): { solid: Mesh | null; edges: LineSegments | null; spent: BufferGeometry[] } {
  const solids: BufferGeometry[] = [];
  const edges: BufferGeometry[] = [];
  const spent = new Set<BufferGeometry>();
  for (const root of objects) {
    root.updateMatrixWorld(true);
    root.traverse((o) => {
      if (o instanceof InstancedMesh && isMass(o.material)) {
        const color = o.material.uniforms.uColor.value as Color;
        for (let i = 0; i < o.count; i++) {
          o.getMatrixAt(i, tmp);
          solids.push(solidPiece(o.geometry, tmp.premultiply(o.matrixWorld), color));
        }
        spent.add(o.geometry);
      } else if (o instanceof Mesh && isMass(o.material)) {
        solids.push(solidPiece(o.geometry, o.matrixWorld, o.material.uniforms.uColor.value as Color));
        spent.add(o.geometry);
      } else if (o instanceof LineSegments && isEdge(o.material)) {
        edges.push(edgePiece(o.geometry, o.matrixWorld));
        spent.add(o.geometry);
      }
    });
  }
  for (const root of objects) root.removeFromParent();
  const solidGeo = solids.length ? mergeGeometries(solids) : null;
  const edgeGeo = edges.length ? mergeGeometries(edges) : null;
  solids.forEach((g) => g.dispose());
  edges.forEach((g) => g.dispose());
  return {
    solid: solidGeo ? new Mesh(solidGeo, solidMat) : null,
    edges: edgeGeo ? new LineSegments(edgeGeo, edgeMat) : null,
    spent: [...spent],
  };
}
