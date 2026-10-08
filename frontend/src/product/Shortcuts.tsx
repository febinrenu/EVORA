"use client";

import { useEffect, useRef } from "react";

const KEYS: [string, string][] = [
  ["/", "Ask about the footage"],
  ["Enter", "Send the question"],
  ["↑ ↓", "Earlier questions, in the ask bar"],
  ["[ ]", "Previous or next evidence"],
  ["J K L", "Back, pause, forward in a clip"],
  ["← →", "Step one frame in a paused clip"],
  ["?", "Show or hide this list"],
];

export function Shortcuts({ onClose }: { onClose: () => void }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => ref.current?.focus(), []);
  return (
    <div className="lt-keys" role="dialog" aria-modal="true" aria-label="Keyboard shortcuts" tabIndex={-1} ref={ref} onClick={onClose}>
      <div className="lt-keys-sheet" onClick={(e) => e.stopPropagation()}>
        <h2>Keyboard</h2>
        <dl>
          {KEYS.map(([k, v]) => (
            <div key={k}>
              <dt>{k}</dt>
              <dd>{v}</dd>
            </div>
          ))}
        </dl>
        <button type="button" onClick={onClose}>
          Close
        </button>
      </div>
    </div>
  );
}
