"use client";

// Clarify card (P4.8, first pass): the yellow-tabbed question, a contact sheet
// of camera frames to choose from, or a typed answer ("camera 2", "the lobby
// one"). The paused query continues on the same sheet once answered.
import { useState } from "react";
import type { ClarifyRequest } from "@/lib/api/client";
import { useEvora, type Case } from "./store";
import { Frame } from "./Frame";

export function ClarifyCard({ c, req }: { c: Case; req: ClarifyRequest }) {
  const clarify = useEvora((s) => s.clarify);
  const [typed, setTyped] = useState("");
  const what = req.referent.text;

  return (
    <div className="lt-clarify" role="group" aria-label="One question before answering">
      <span className="lt-tab">Clarify once</span>
      <p className="lt-clarify-q">{req.question}</p>
      {req.options?.length ? (
        <ul className="lt-contact">
          {req.options.map((o) => (
            <li key={o.camera_id}>
              <button type="button" onClick={() => clarify(c.id, { query_id: req.query_id, camera_id: o.camera_id }, `“${what}” is ${o.camera_name}`)}>
                <Frame src={o.thumb_url} alt={`${o.camera_name}, current frame`} />
                <span>{o.camera_name}</span>
              </button>
            </li>
          ))}
        </ul>
      ) : null}
      <form
        className="lt-clarify-typed"
        onSubmit={(e) => {
          e.preventDefault();
          if (typed.trim()) clarify(c.id, { query_id: req.query_id, text: typed.trim() }, `“${what}” is ${typed.trim()}`);
        }}
      >
        <label>
          <span>Or answer in words</span>
          <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="camera 2, the lobby one…" />
        </label>
        <button type="submit" disabled={!typed.trim()}>
          Save place
        </button>
      </form>
    </div>
  );
}
