"use client";

// Camera rail and load-footage flow (P4.7). Drop files, check each detected
// clock and where it came from, name the cameras, start indexing, and watch
// per-layer progress with throughput.
import { useCallback, useState } from "react";
import { ApiError, endpoints, frameUrl, type CameraInfo } from "@/lib/api/client";
import { clock, day } from "./format";
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
  const [error, setError] = useState<string | null>(null);

  const upload = useCallback(async (files: File[]) => {
    const videos = files.filter((f) => /\.(mp4|mov|mkv|avi)$/i.test(f.name));
    if (!videos.length) {
      setError("Those are not video files. evora reads mp4, mov, mkv and avi.");
      return;
    }
    setError(null);
    setUploading(videos.length === 1 ? `Reading ${videos[0].name}…` : `Reading ${videos.length} files…`);
    try {
      const cams = await endpoints.upload(videos);
      useEvora.getState().upsertCameras(cams);
    } catch (e) {
      setError(e instanceof ApiError ? e.message : "The upload did not reach this machine's API.");
    } finally {
      setUploading(null);
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
        <span>{uploading ?? (cameras.length ? "Drop more footage, or choose files" : "Drop footage here, or choose files")}</span>
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
      <Frame src={frameUrl(cam.id, mid)} alt={`${cam.name}, frame from the middle of the recording`} />
      <div className="lt-cam-body">
        <input
          className="lt-cam-name"
          value={name}
          aria-label={`Name of camera ${cam.id}`}
          onChange={(e) => setName(e.target.value)}
          onBlur={() => void saveName()}
          onKeyDown={(e) => e.key === "Enter" && (e.target as HTMLInputElement).blur()}
        />
        <p className="lt-cam-clock">
          {day(cam.t0)} {clock(cam.t0)}, {CLOCK_SOURCE[cam.t0_source]}
        </p>
        <p className="lt-cam-state">{stateLine(cam, job)}</p>
        {cam.status === "ingesting" && job ? (
          <span className="lt-progress" style={{ ["--p" as string]: String(job.progress ?? 0) }} aria-hidden="true">
            <i />
          </span>
        ) : null}
      </div>
    </li>
  );
}

function stateLine(cam: CameraInfo, job: ReturnType<typeof useEvora.getState>["jobs"][string] | undefined): string {
  if (cam.status === "error") return job?.error ? `Stopped: ${job.error}` : "Indexing stopped. Check the file.";
  if (cam.status === "pending") return "Ready to index";
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
