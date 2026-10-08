// Lightweight text splitting rendered at build time (no layout shift, no
// runtime DOM surgery). Assistive tech reads the plain string; the split
// glyphs are presentation only.
import type { ElementType, ReactNode } from "react";

interface SplitProps {
  text: string;
  as?: ElementType;
  className?: string;
  /** extra attributes for the root, e.g. data-el */
  [data: `data-${string}`]: string | boolean | undefined;
}

export function Split({ text, as: Tag = "span", className, ...rest }: SplitProps): ReactNode {
  const words = text.split(" ");
  return (
    <Tag className={className} {...rest}>
      <span className="sr-only">{text}</span>
      <span className="split" aria-hidden="true">
        {words.map((w, wi) => (
          <span className="w" key={wi}>
            {Array.from(w).map((ch, ci) => (
              <span className="chm" key={ci}>
                <span className="ch">{ch}</span>
              </span>
            ))}
            {wi < words.length - 1 ? <span className="sp"> </span> : null}
          </span>
        ))}
      </span>
    </Tag>
  );
}
