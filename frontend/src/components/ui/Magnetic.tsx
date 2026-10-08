"use client";

// Magnetic hover: the control leans toward the pointer inside a radius. Uses
// gsap.quickTo (one tween per axis, reused) so pointermove never allocates.
import Link from "next/link";
import { useEffect, useRef, type ReactNode } from "react";
import { gsap } from "gsap";
import { DOM } from "@/lib/config/animation";

function useMagnet<T extends HTMLElement>(strength: number = DOM.magnetStrength) {
  const ref = useRef<T>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || window.matchMedia("(pointer: coarse), (prefers-reduced-motion: reduce)").matches) return;
    const xTo = gsap.quickTo(el, "x", { duration: 0.5, ease: "power3.out" });
    const yTo = gsap.quickTo(el, "y", { duration: 0.5, ease: "power3.out" });
    let rect = el.getBoundingClientRect();
    const onEnter = () => {
      rect = el.getBoundingClientRect();
    };
    const onMove = (e: PointerEvent) => {
      const cx = rect.left + rect.width / 2;
      const cy = rect.top + rect.height / 2;
      xTo((e.clientX - cx) * strength);
      yTo((e.clientY - cy) * strength);
    };
    const onLeave = () => {
      xTo(0);
      yTo(0);
    };
    el.addEventListener("pointerenter", onEnter);
    el.addEventListener("pointermove", onMove);
    el.addEventListener("pointerleave", onLeave);
    return () => {
      el.removeEventListener("pointerenter", onEnter);
      el.removeEventListener("pointermove", onMove);
      el.removeEventListener("pointerleave", onLeave);
    };
  }, [strength]);
  return ref;
}

export function MagneticLink({ href, className, children }: { href: string; className?: string; children: ReactNode }) {
  const ref = useMagnet<HTMLAnchorElement>();
  return (
    <Link ref={ref} href={href} className={className} data-cursor="Enter">
      {children}
    </Link>
  );
}

export function MagneticButton({ className, children, onClick, label }: { className?: string; children: ReactNode; onClick: () => void; label?: string }) {
  const ref = useMagnet<HTMLButtonElement>(0.25);
  return (
    <button ref={ref} type="button" className={className} onClick={onClick} aria-label={label}>
      {children}
    </button>
  );
}
