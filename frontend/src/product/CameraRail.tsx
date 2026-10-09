"use client";

// Camera rail and load-footage flow (P4.7). Drop files, check each detected
// clock and where it came from (the on-screen clock is read in the background
// after upload), correct it by hand, name the cameras, start indexing, and
// watch per-layer progress with throughput.
import { useCallback, useState } from "react";
import { ApiError, endpoints, frameUrl, liveUrl, type CameraInfo } from "@/lib/api/client";
import { clock, day, fromWall, wallInput } from "./format";
import { useEvora } from "./store";
import { Frame } from "./Frame";

const CLOCK_SOURCE: Record<CameraInfo["t0_source"], string> = {
  filename: "from the file name",
  metadata: "from the file's metadata",
  osd: "from the on-screen timestamp",
  slate: "from the clock slate",
  manual: "clock unknown, using the file time",
  live: "live",
};

const LAYER_COPY: Record<string, string> = {
  L0: "scenes",
  L1: "people and vehicles",
  L2: "colours, carrying and crossings",
  L3: "descriptions",
};

const ACCEPT = ".mp4,.mov,.mkv,.avi";

export function CameraRail() {
  const cameras = useEvora((s) => s.cameras);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState<string | null>(null);
  // what the server says while it works on the upload, e.g. converting an odd codec
  const bulletin = useEvora((s) => s.bulletin);
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(async (files: File[]) => {
    const videos = files.filter((f) => /\.(mp4|mov|mkv|avi)$/i.test(f.name));
    if (!videos.length) {
      setError("Those are not video files. evora reads mp4, mov, mkv and avi.");
      return;
    }
    setError(null);
    useEvora.getState().setBulletin(null);
    setUploading(videos.length === 1 ? `Reading ${videos[0].name}…` : `Reading ${videos.length} files…`);
    try {
      const cams = await endpoints.upload(videos);
      useEvora.getState().upsertCameras(cams);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The upload did not reach this machine's API.");
    } finally {
      setUploading(null);
      useEvora.getState().setBulletin(null);
    }
  }, []);

  const pending = cameras.filter((c) => c.status === "pending");

  return (
    <aside
      className={`lt-rail${over ? " is-over" : ""}`}
      aria-label="Cameras"
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        void upload(Array.from(e.dataTransfer.files));
      }}
    >
      <h2>Cameras</h2>
      {cameras.length ? (
        <ul className="lt-cams">
          {cameras.map((c) => (
            <CameraRow key={c.id} cam={c} />
          ))}
        </ul>
      ) : null}
      {pending.length ? (
        <button type="button" className="lt-primary" onClick={() => void startIngest(pending.map((c) => c.id), setError)}>
          Start indexing {pending.length === 1 ? "this camera" : `${pending.length} cameras`}
        </button>
      ) : null}
      <label className="lt-drop">
        <input type="file" accept={ACCEPT} multiple onChange={(e) => void upload(Array.from(e.target.files ?? []))} />
        <span>{uploading ? (bulletin ? `${uploading} ${bulletin[0].toUpperCase()}${bulletin.slice(1)}…` : uploading) : cameras.length ? "Drop more footage, or choose files" : "Drop footage here, or choose files"}</span>
      </label>
      {error ? (
        <p className="lt-error" role="alert">
          {error}
        </p>
      ) : null}
    </aside>
  );
}

async function startIngest(ids: string[], onError: (m: string) => void) {
  try {
    const jobs = await endpoints.ingest(ids);
    for (const j of jobs) useEvora.getState().setJob(j);
    for (const id of ids) useEvora.getState().setCameraStatus(id, "ingesting");
  } catch (e) {
    onError(e instanceof ApiError ? e.message : "Indexing could not start.");
  }
}

function CameraRow({ cam }: { cam: CameraInfo }) {
  const job = useEvora((s) => s.jobs[cam.id]);
  const live = useEvora((s) => s.live[cam.id]);
  const analysis = useEvora((s) => s.analysis[cam.id]);
  const reading = useEvora((s) => s.clock[cam.id]);
  const [liveError, setLiveError] = useState<string | null>(null);
  const streaming = cam.status === "live" || live === "running" || live === "retrying" || live === "starting";
  const toggleLive = async () => {
    setLiveError(null);
    try {
      if (streaming) {
        await endpoints.stopReplay([cam.id]);
        useEvora.getState().setLive(cam.id, "stopped");
      } else {
        // replay with live analysis so watches fire; without the perception stack, replay alone
        try {
          await endpoints.replay([cam.id], true);
        } catch (e) {
          if (!(e instanceof ApiError && e.status === 503)) throw e;
          await endpoints.replay([cam.id], false);
          setLiveError("Replaying without live analysis: the perception stack is not installed (start.bat setup).");
        }
        useEvora.getState().setLive(cam.id, "starting");
      }
    } catch (e) {
      setLiveError(e instanceof ApiError ? e.message : "Replay could not start.");
    }
  };
  const [name, setName] = useState(cam.name);
  const [prevName, setPrevName] = useState(cam.name);
  if (cam.name !== prevName) {
    setPrevName(cam.name);
    setName(cam.name);
  }
  const mid = cam.t0 + (cam.duration_s ?? 0) / 2;
  const saveName = async () => {
    const n = name.trim();
    if (!n || n === cam.name) return setName(cam.name);
    try {
      useEvora.getState().upsertCameras([await endpoints.renameCamera(cam.id, n)]);
    } catch {
      setName(cam.name);
    }
  };

  return (
    <li className={`lt-cam is-${cam.status}`}>
      {streaming ? (
        // MJPEG: the browser keeps the multipart stream open; frames are face-blurred server side
        <Frame src={liveUrl(cam.id)} alt={`${cam.name}, live`} osd={live === "retrying" ? "RETRYING" : "● LIVE"} className="lt-live" />
      ) : (
        <Frame src={frameUrl(cam.id, mid)} alt={`${cam.name}, frame from the middle of the recording`} />
      )}
      <div className="lt-cam-body">
        <input
          className="lt-cam-name"
          value={name}
          aria-label={`Name of camera ${cam.id}`}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => void saveName()}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
        <ClockLine cam={cam} reading={reading} />
        <p className="lt-cam-state">{stateLine(cam, job, reading?.state === "reading")}</p>
        {cam.kind === "file" && cam.status === "ready" ? (
          <button type="button" className="lt-link" onClick={() => void toggleLive()}>
            {streaming ? "Stop the live replay" : "Replay as live"}
          </button>
        ) : null}
        {streaming ? (
          <p className="lt-cam-state">{analysis === "running" ? "Live replay, analysed as it plays: watches can fire" : analysis === "error" ? "Live replay; analysis stopped with an error" : "Live replay"}</p>
        ) : null}
        {liveError ? <p className="lt-error">{liveError}</p> : null}
        {cam.status === "error" ? (
          <button type="button" className="lt-link" onClick={() => void startIngest([cam.id], setLiveError)}>
            Try again
          </button>
        ) : null}
        {cam.status === "ingesting" && job ? (
          <span className="lt-progress" style={{ ["--p" as string]: String(job.progress ?? 0) }} aria-hidden="true">
            <i />
          </span>
        ) : null}
      </div>
    </li>
  );
}

/** The camera's clock, where it came from, and a by-hand correction that moves everything already indexed. */
function ClockLine({ cam, reading }: { cam: CameraInfo; reading: { state: "reading" | "failed"; error?: string } | undefined }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState("");
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // the API reports a typed clock and the file time both as "manual"; this session knows which
  const [byHand, setByHand] = useState(false);
  const source =
    reading?.state === "reading"
      ? "reading the on-screen clock…"
      : reading?.state === "failed"
        ? "the on-screen clock could not be read, using the file time"
        : byHand && cam.t0_source === "manual"
          ? "set by hand"
          : CLOCK_SOURCE[cam.t0_source];
  const open = () => {
    setValue(wallInput(cam.t0));
    setError(null);
    setEditing(true);
  };
  const save = async () => {
    const t0 = fromWall(value);
    if (t0 === null) return setError("Give a date and a time to the second.");
    if (Math.abs(t0 - cam.t0) < 0.5) return setEditing(false);
    setSaving(true);
    setError(null);
    try {
      useEvora.getState().upsertCameras([await endpoints.setClock(cam.id, t0)]);
      useEvora.getState().setClock(cam.id, null);
      useEvora.getState().clockCorrected(cam.id);
      setByHand(true);
      setEditing(false);
    } catch (e) {
      // 409 while the camera is being indexed: the server's sentence says why
      setError(e instanceof ApiError ? e.message : "The clock could not be changed.");
    } finally {
      setSaving(false);
    }
  };

  if (!editing) {
    return (
      <>
        <p className={`lt-cam-clock${reading?.state === "reading" ? " is-reading" : ""}`} aria-live="polite">
          {day(cam.t0)} {clock(cam.t0)}, {source}
        </p>
        {cam.kind === "file" ? (
          <button type="button" className="lt-link" onClick={open}>
            Correct the clock
          </button>
        ) : null}
      </>
    );
  }
  return (
    <form
      className="lt-clock-edit"
      onSubmit={(e) => {
        e.preventDefault();
        void save();
      }}
    >
      <label>
        <span>First frame of {cam.name} was recorded at</span>
        <input type="datetime-local" step={1} value={value} onChange={(e) => setValue(e.target.value)} disabled={saving} autoFocus />
      </label>
      <p className="lt-cam-state">Everything already indexed for this camera moves with it.</p>
      <div className="lt-clock-actions">
        <button type="submit" className="lt-primary" disabled={saving}>
          {saving ? "Moving…" : "Set the clock"}
        </button>
        <button type="button" className="lt-link" onClick={() => setEditing(false)} disabled={saving}>
          Cancel
        </button>
      </div>
      {error ? (
        <p className="lt-error" role="alert">
          {error}
        </p>
      ) : null}
    </form>
  );
}

function stateLine(cam: CameraInfo, job: ReturnType<typeof useEvora.getState>["jobs"][string] | undefined, reading: boolean): string {
  if (cam.status === "error") return job?.error ? `Stopped: ${job.error}` : "Indexing stopped. Check the file.";
  if (cam.status === "pending") return reading ? "Ready to index; indexing starts once the clock is read" : "Ready to index";
  if (cam.status === "ingesting" && reading) return "Waiting for the clock reading, then indexing";
  if (cam.status === "live") return "Live";
  const layers = cam.layers ?? [];
  const searchable = layers.includes("L0");
  if (cam.status === "ingesting") {
    const layer = job?.layer ?? null;
    const pct = job?.progress !== undefined ? ` ${Math.round((job.progress ?? 0) * 100)}%` : "";
    const rate = job?.video_s_per_s ? `, ${job.video_s_per_s.toFixed(1)}× real time` : "";
    const what = layer ? LAYER_COPY[layer] ?? layer : "footage";
    return `${searchable ? "Searchable now. " : ""}Refining ${what}${pct}${rate}`;
  }
  return layers.length ? `Indexed: ${layers.map((l) => LAYER_COPY[l] ?? l).join(", ")}` : "Ready";
}
