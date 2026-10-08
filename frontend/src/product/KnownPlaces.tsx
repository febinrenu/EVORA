"use client";

// Known places ledger (P4.9): every fact EVORA learned, where it came from and
// how often it is used, and in plain words what it does to answers. Rename,
// change what it means (recorded as a correction), drop an alias, confirm a
// guessed alias, or delete it (asks first). Memory is always yellow.
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
  const [changing, setChanging] = useState(false);
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
      <p className="lt-fact-means">{meaning(f, cameras)}</p>
      <span className="lt-place-meta">
        {SOURCE[f.source]} {day(f.created_at)} {clock(f.created_at).slice(0, 5)}
        {f.use_count ? `. Used in ${f.use_count} answer${f.use_count === 1 ? "" : "s"}` : ". Not used yet"}
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
            {editable(f) ? (
              <button type="button" onClick={() => setChanging((c) => !c)} aria-expanded={changing}>
                Change what it means
              </button>
            ) : null}
            <button type="button" onClick={() => setConfirmDelete(true)}>
              Delete
            </button>
          </>
        )}
      </span>
      {changing ? (
        <BindingEditor
          f={f}
          cameras={cameras}
          onCancel={() => setChanging(false)}
          onSave={(binding) => {
            setChanging(false);
            void apply(() => endpoints.patchMemory(f.id, { binding }));
          }}
        />
      ) : null}
      {error ? (
        <span className="lt-error" role="alert">
          {error}
        </span>
      ) : null}
    </li>
  );
}

const isTime = (b: Record<string, unknown>) => typeof b.tod_after === "string" || typeof b.tod_before === "string";
const editable = (f: MemoryFact) => isTime(f.binding) || typeof f.binding.camera_id === "string";

/** What the fact does to answers, in one sentence. */
function meaning(f: MemoryFact, cameras: { id: string; name: string }[]): string {
  const b = f.binding;
  const name = `“${f.canonical}”`;
  if (isTime(b)) {
    const after = typeof b.tod_after === "string" ? b.tod_after : null;
    const before = typeof b.tod_before === "string" ? b.tod_before : null;
    const overnight = after && before && after > before ? ", overnight" : "";
    return `When you say ${name}, EVORA only searches ${after ?? "midnight"} to ${before ?? "midnight"} of each day${overnight}.`;
  }
  if (typeof b.camera_id === "string") {
    const cam = cameras.find((c) => c.id === b.camera_id);
    return `${name} means the ${cam?.name ?? b.camera_id} camera, ${b.zone_id ? "only where its line or area is marked" : "its whole view"}.`;
  }
  if (typeof b.global_id === "string" || typeof b.track_id === "string") return `${name} means one tracked object, followed across cameras where it was linked.`;
  return `${name} was learned from a conversation.`;
}

/** Hours for a time fact, a camera for a place fact; saved as a correction of the old fact. */
function BindingEditor({ f, cameras, onCancel, onSave }: { f: MemoryFact; cameras: CameraInfo[]; onCancel: () => void; onSave: (b: Record<string, unknown>) => void }) {
  const b = f.binding;
  const [after, setAfter] = useState(typeof b.tod_after === "string" ? b.tod_after : "08:00");
  const [before, setBefore] = useState(typeof b.tod_before === "string" ? b.tod_before : "18:00");
  const [camera, setCamera] = useState(typeof b.camera_id === "string" ? b.camera_id : (cameras[0]?.id ?? ""));
  const time = isTime(b);
  const save = () => {
    if (time) return onSave({ ...b, tod_after: after, tod_before: before });
    // a marked line belongs to one camera: moving the place drops it
    if (camera === b.camera_id) return onSave(b);
    const rest = Object.fromEntries(Object.entries(b).filter(([k]) => k !== "zone_id"));
    onSave({ ...rest, camera_id: camera });
  };
  return (
    <form
      className="lt-fact-edit"
      onSubmit={(e) => {
        e.preventDefault();
        save();
      }}
    >
      {time ? (
        <div className="lt-row">
          <label>
            From <input type="time" value={after} onChange={(e) => setAfter(e.target.value)} required />
          </label>
          <label>
            to <input type="time" value={before} onChange={(e) => setBefore(e.target.value)} required />
          </label>
        </div>
      ) : (
        <label className="lt-row">
          Camera
          <select value={camera} onChange={(e) => setCamera(e.target.value)}>
            {cameras.map((c) => (
              <option key={c.id} value={c.id}>
                {c.name}
              </option>
            ))}
          </select>
        </label>
      )}
      {!time && b.zone_id && camera !== b.camera_id ? <span className="lt-place-meta">The marked line stays with the old camera. Mark it again from a question if it matters.</span> : null}
      <div className="lt-row">
        <button type="submit" className="lt-primary">
          Save the correction
        </button>
        <button type="button" className="lt-link" onClick={onCancel}>
          Cancel
        </button>
      </div>
    </form>
  );
}
