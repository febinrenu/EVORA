// The site drawn as a cyanotype plan in world coordinates (x, z), so anything
// placed on it lines up with the 3D twin.
import { CAR_ROUTE } from "@/lib/data/site";

export const PLAN_BOX = { x: -92, z: -100, w: 188, h: 156 } as const;

export const planPct = (x: number, z: number): { left: string; top: string } => ({
  left: `${((x - PLAN_BOX.x) / PLAN_BOX.w) * 100}%`,
  top: `${((z - PLAN_BOX.z) / PLAN_BOX.h) * 100}%`,
});

const BUILDINGS: [number, number, number, number][] = [
  [-28, -14, 26, 20],
  [-20, -1.5, 14, 6],
  [46, -62, 36, 22],
  [74, -20, 18, 30],
  [-74, -42, 30, 26],
  [-58, -78, 22, 18],
  [8, -86, 26, 16],
];

const route = Array.from({ length: 80 }, (_, i) => {
  const p = CAR_ROUTE.getPointAt(0.18 + (i / 79) * 0.82);
  return `${p.x.toFixed(1)},${p.z.toFixed(1)}`;
}).join(" ");

export function SitePlan({ showRoute = false }: { showRoute?: boolean }) {
  return (
    <svg className="siteplan" viewBox={`${PLAN_BOX.x} ${PLAN_BOX.z} ${PLAN_BOX.w} ${PLAN_BOX.h}`} aria-hidden="true">
      <defs>
        <pattern id="plan-grid" width="5" height="5" patternUnits="userSpaceOnUse">
          <path d="M5 0H0V5" fill="none" stroke="currentColor" strokeWidth="0.08" opacity="0.5" />
        </pattern>
      </defs>
      <rect x={PLAN_BOX.x} y={PLAN_BOX.z} width={PLAN_BOX.w} height={PLAN_BOX.h} fill="url(#plan-grid)" opacity="0.5" />
      <rect x={PLAN_BOX.x} y={39} width={PLAN_BOX.w} height={12} className="plan-road" />
      <line x1={PLAN_BOX.x} x2={-6.4} y1={34} y2={34} className="plan-fence" />
      <line x1={6.4} x2={PLAN_BOX.x + PLAN_BOX.w} y1={34} y2={34} className="plan-fence" />
      <rect x={-61} y={6} width={46} height={28} className="plan-lot" />
      {BUILDINGS.map(([x, z, w, d], i) => (
        <rect key={i} x={x - w / 2} y={z - d / 2} width={w} height={d} className="plan-building" />
      ))}
      <polyline points={route} className={showRoute ? "plan-route is-on" : "plan-route"} />
    </svg>
  );
}
