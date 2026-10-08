"use client";

// Known places ledger (P4.9): every fact EVORA learned, where it came from and
// how often it is used. Rename, drop an alias, confirm a guessed alias, or
// delete the place (asks first). Memory is always yellow.
import { useState } from "react";
import { ApiError, endpoints, type CameraInfo, type MemoryFact } from "@/lib/api/client";
import { clock, day } from "./format";
import { useEvora } from "./store";

const SOURCE: Record<MemoryFact["source"], string> = {
  clarification: "Learned from you",
  statement: "Told by you",
  correction: "Corrected by you",
  import: "Imported",
};

export function KnownPlaces() {
  const memory = useEvora((s) => s.memory);
  const cameras = useEvora((s) => s.cameras);
  const facts = memory.filter((f) => !f.superseded_by);
  return (
    <section className="lt-places" aria-label="Known places">
      <h2>Known places</h2>
      {facts.length ? (
        <ul>
          {facts.map((f) => (
            <Fact key={f.id} f={f} cameras={cameras} />
          ))}
        </ul>
      ) : (
        <p className="lt-quiet">Places you name while asking are kept here, so nothing is asked twice.</p>
      )}
    </section>
  );
}

function Fact({ f, cameras }: { f: MemoryFact; cameras: CameraInfo[] }) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(f.canonical);
  const [confirmDelete, setConfirmDelete] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const apply = async (fn: () => Promise<unknown>) => {
    setError(null);
    try {
      await fn();
      await useEvora.getState().refreshMemory();
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "That change did not save.");
    }
  };

  const rename = () => {
    const n = name.trim();
    setEditing(false);
    if (n && n !== f.canonical) void apply(() => endpoints.patchMemory(f.id, { canonical: n }));
    else setName(f.canonical);
  };

  return (
    <li className="lt-fact">
      {editing ? (
        <input
          className="lt-fact-name"
          value={name}
          autoFocus
          aria-label="New name for this place"
          onChange={(e) => setName(e.target.value)}
          onBlur={rename}
          onKeyDown={(e) => {
            if (e.key === "Enter") rename();
            if (e.key === "Escape") {
              setName(f.canonical);
              setEditing(false);
            }
          }}
        />
      ) : (
        <button type="button" className="lt-marker" onClick={() => setEditing(true)} title="Rename">
          {f.canonical}
        </button>
      )}
      <span className="lt-place-meta">
        {describeBinding(f.binding, cameras)}. {SOURCE[f.source]} {day(f.created_at)} {clock(f.created_at).slice(0, 5)}
        {f.use_count ? `. Used ${f.use_count}×` : ""}
      </span>
      {f.aliases?.length ? (
        <span className="lt-aliases" aria-label="Also called">
          {f.aliases.map((a) => (
            <span key={a} className="lt-alias">
              {a}
              <button type="button" aria-label={`Forget the name ${a}`} onClick={() => void apply(() => endpoints.patchMemory(f.id, { aliases: (f.aliases ?? []).filter((x) => x !== a) }))}>
                ×
              </button>
            </span>
          ))}
        </span>
      ) : null}
      {f.inferred_aliases?.length ? (
        <span className="lt-aliases is-guess" aria-label="Names EVORA guessed">
          {f.inferred_aliases.map((a) => (
            <span key={a} className="lt-alias">
              {a}?
              <button type="button" aria-label={`Confirm ${a} means ${f.canonical}`} onClick={() => void apply(() => endpoints.patchMemory(f.id, { confirm_aliases: [a] }))}>
                ✓
              </button>
            </span>
          ))}
        </span>
      ) : null}
      <span className="lt-fact-actions">
        {confirmDelete ? (
          <>
            <span>Forget “{f.canonical}”? EVORA will ask again next time.</span>
            <button type="button" className="lt-danger" onClick={() => void apply(() => endpoints.deleteMemory(f.id))}>
              Forget it
            </button>
            <button type="button" onClick={() => setConfirmDelete(false)}>
              Keep
            </button>
          </>
        ) : (
          <>
            <button type="button" onClick={() => setEditing(true)}>
              Rename
            </button>
            <button type="button" onClick={() => setConfirmDelete(true)}>
              Delete
            </button>
          </>
        )}
      </span>
      {error ? (
        <span className="lt-error" role="alert">
          {error}
        </span>
      ) : null}
    </li>
  );
}

function describeBinding(binding: Record<string, unknown>, cameras: { id: string; name: string }[]): string {
  if (typeof binding.camera_id === "string") {
    const cam = cameras.find((c) => c.id === binding.camera_id);
    return `${cam?.name ?? binding.camera_id}${binding.zone_id ? ", marked line" : ", whole view"}`;
  }
  if (typeof binding.tod_after === "string" || typeof binding.tod_before === "string") return `${binding.tod_after ?? "…"} to ${binding.tod_before ?? "…"}`;
  if (typeof binding.global_id === "string") return "A tracked object";
  return "Learned";
}
