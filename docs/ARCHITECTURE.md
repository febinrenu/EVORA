# Architecture

How the running system fits together, written from the code. For the reasons behind the design see the plan; this page
is the map.

## Processes

`make up` starts one Python process (the API, FastAPI on 127.0.0.1:8700) and, when Node is installed, the web interface
(Next.js on 127.0.0.1:3000). Local models run in Ollama (a separate service) and replay-as-live uses a loopback
MediaMTX plus one ffmpeg per camera. Nothing listens on the network unless asked to.

## One workspace per site

A workspace is a folder under `workspaces/<name>/`:

- `evora.sqlite`: cameras, tracks, track points, events, zones, memory facts, pending questions, standing queries,
  alerts, evidence registry, audit log, query log, settings (WAL mode, one writer connection per file).
- `vectors/` (LanceDB): crop and scene embeddings, re-identification embeddings, caption and alias embeddings.
- `media/`, `clips/`, `uploads/`, `exports/`, `live/`: crops, rendered clips and thumbnails (size-capped cache),
  uploaded files, evidence packs, restream configuration and logs.

Memory belongs to the workspace, so "main gate" learned on one site never leaks into another.

## Who owns what

| Area | Code | Owner |
|---|---|---|
| Platform | `core/`, `api/`, `memory/`, `alerts/`, `live/`, `evidence/`, `contracts/`, `config/` | M1 |
| Perception and identity | `perception/`, `reid/` | M2 |
| Language and retrieval | `llm/`, `query/`, `baseline/`, `eval/` | M3 |
| Interface | `frontend/` | M4 |

The contracts in `contracts/` (models, SQL schema, vector tables, API spec, fixtures, generated TypeScript types) are the
only shared surface.

## Footage in

1. `POST /api/cameras` saves the file (extension allow-list, size cap, ffprobe check, SHA-256, de-duplication, automatic
   conversion of unusual codecs to H.264) and detects the clock (file name, metadata, on-screen text, slate, manual).
2. `POST /api/ingest` queues one job per camera. The job runner calls the perception pipeline through
   `core/perception_adapter.py`, records only the layers that finished, and publishes progress on the event bus.
3. When the last camera of a batch finishes, identities are linked across cameras and alerts are re-checked.

The adapter finds M2's functions by name in their modules and passes the app's database and workspace, so the platform
runs (with honest "pending" states) even when the perception stack is not installed.

## A question, end to end

`POST /api/query` streams events: `plan`, then `clarify` (and stop) or `evidence*`, `answer`, `verified*`, `done`.

1. The planner turns the text into a typed plan (fast path, plan cache, then the language model).
2. Every referent in the plan goes through memory: exact alias, alias embedding, a language-model equivalence check for
   the grey band, camera names. A known place becomes a camera and a zone; an unknown or ambiguous one stops the stream
   with one clarification, stored so it survives a restart.
3. Retrieval, temporal and spatial logic, then a deterministic answer built from the evidence (the model plans, it does
   not write evidence). Verification of the top results streams after the answer.
4. The answer's evidence is registered, so thumbnails, clips and evidence packs can be rendered for it; the first three
   are rendered in the background.

`POST /api/clarify` binds the user's choice (camera, optional drawn line or polygon, known place, track or hours) to a
memory fact and resumes the question. The same flow serves standing questions.

## Evidence

Every evidence item carries wall-clock time and the offset into its file. The media service cuts clips and frames with
ffmpeg, blurs faces unless a short-lived, audited token says otherwise, and caches the result. An evidence pack is a
zip with the clip, three frames, provenance and a `SHA256SUMS` file anyone can check offline.

## Zones and events

A zone (line, polygon or whole frame) is stored on its camera. Saving one clears its old events and asks perception to
recompute them from stored trajectories, so crossings exist immediately without re-ingesting.

## Alerts

A standing question compiles into a rule (classes, event kinds, zone, direction, hours, cooldown). The engine evaluates
events (stored or live) and raises each alert once; alerts appear on `/api/events`, in the alert list, and as a text-only
phone push through the gateway.

## Privacy

On-prem mode is a process-wide guard on sockets, DNS and asyncio connections: only loopback and Unix sockets are allowed.
The single outbound gateway is the only module that talks to cloud services. Faces are blurred in all served media;
exports are blurred or refused. Queries, unblur requests, exports, setting changes and corrections are in the audit log.

## Extension points

- Hooks in `core/perception_adapter.py` for the perception and identity functions.
- The event bus (`core/bus.py`): notes with a `kind` (`ingest`, `camera`, `alert`, `privacy`, `live`, `zone`, `reid`)
  delivered on `GET /api/events`.
- Config in `config/default.yaml` and `config/profiles/*.yaml`; secrets only in `.env`.
