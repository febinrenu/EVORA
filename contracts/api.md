# HTTP API (contract v1)

FastAPI on `:8700`, UI dev server on `:5173`.

| Method & path | Body / params | Returns |
|---|---|---|
| `GET /api/health` | — | `{ok, version, workspace, profile, onprem, layers_ready}` |
| `GET /api/workspaces` · `POST /api/workspaces` · `POST /api/workspaces/{slug}/activate` | `{name}` | workspace list / created / active |
| `GET /api/cameras` | — | `CameraInfo[]` |
| `POST /api/cameras` | multipart file(s) **or** `{uri, name}` | `CameraInfo[]` (clock detected, status `pending`) |
| `PATCH /api/cameras/{id}` | `{name?, t0?, site_xy?}` | `CameraInfo` |
| `GET /api/cameras/{id}/frame?t=` | epoch | JPEG (faces blurred unless unblur token) |
| `GET /api/cameras/{id}/live.mjpg` | — | MJPEG for live tiles |
| `POST /api/ingest` | `{camera_ids, layers?}` | `IngestJob[]` |
| `POST /api/query` | `{text, session_id}` | **SSE** stream of `StreamEvent` |
| `POST /api/clarify` | `ClarifyResponse` | **SSE** stream continuing the paused query |
| `GET /api/memory` · `POST /api/memory` · `PATCH /api/memory/{id}` · `DELETE /api/memory/{id}` | `MemoryFact` | facts |
| `GET /api/zones?camera_id=` · `POST /api/zones` | `Zone` | zones (POST triggers retroactive event recompute) |
| `GET /api/tracks/{id}` | — | track + points + attrs |
| `GET /api/tracks/{id}/similar?k=20` | — | `Evidence[]` (query by example) |
| `GET /api/globals/{gid}/path` | — | `PathHop[]` |
| `GET /api/media/thumb/{evidence_id}.jpg` · `GET /api/media/clip/{evidence_id}.mp4` | range requests | rendered on demand, cached, faces blurred by default |
| `POST /api/media/unblur` (v1.1) | `{reason}` | `{token, expires_at}`: 5-minute token; pass `?unblur=<token>` on media and frame routes. Audited. |
| `POST /api/evidence/{id}/pack` | — | zip (clip, frames, provenance JSON, SHA-256 manifest) |
| `POST /api/standing` · `GET /api/standing` · `PATCH /api/standing/{id}` | `{text}` | `StandingQuery` (compiled) |
| `GET /api/alerts` · `POST /api/alerts/{id}/ack` | — | `Alert[]` |
| `GET /api/events` | — | **SSE**: ingest progress, alerts, camera status, notes |
| `POST /api/settings` | `{onprem?, blur_faces?, reference_now?}` | settings |
| `POST /api/voice` | audio blob | `{text}` (Groq Whisper, or local fallback when on-prem) |
| `GET /api/report` | — | latest eval + ablation JSON for the report page |
| `POST /api/dev/gt` | `{query, camera_id, window}` | appends a ground-truth item (dev builds only) |

**SSE order for a query:** `plan` → (`clarify` and stop) **or** `evidence`* → `answer` → `verified`* → `done`. The UI must render `answer` before `verified` arrives.

**v1.1 notes.** Media responses carry `X-Evora-Blur: applied | off | unavailable`. `unavailable` means blur was requested but the face-blur model is not installed; the UI should warn. Schema adds table `evidence` (see `schema.sql`); M3 registers every `Evidence` it returns via `evora.evidence.store.register`.

**v1.2 notes.** `MemoryFact.inferred_aliases` lists aliases the system learned silently so the Known places ledger can mark them as guesses. `PATCH /api/memory/{id}` accepts `confirm_aliases: string[]` to turn a guess into a confirmed alias. Changing a fact's binding is a correction and drops its guessed aliases.

**v1.3 notes.** `POST /api/zones` saves (or redraws, when the id exists) a zone and recomputes its events at once; the response carries `X-Evora-Events: <n>` or `pending` when the perception pipeline cannot compute them yet. `DELETE /api/zones/{id}` removes a zone, its events and its pointer in any memory fact. Both are additive.

**v1.4 notes.** `POST /api/standing` answers `409 {clarify: ClarifyRequest}` when the place or time word is not known yet: answer it through `POST /api/clarify`, then post the same text again. `GET /api/alerts` accepts `?acknowledged=true|false`. `/api/events` carries `note` events with `kind="alert"` (`{alert: Alert, historical: bool}`); alerts found while scanning already-processed footage have `historical: true` and `"found in earlier footage"` in `evidence.why`. Phone pushes (ntfy) are text only, never sent in on-prem mode, and go through the gateway.

**v1.5 notes.** `GET /api/health` is real and gains `egress_blocked` (outside connections refused since start) and `blur` (`applied | off | unavailable`, checked by running the blur model). `POST /api/settings` validates types (422), keeps `onprem`, `blur_faces` and `reference_now` across restarts, and publishes a bus note `kind="privacy"` `{onprem}` when on-prem changes. In on-prem mode every connection that leaves this machine is refused; `POST /api/voice` then answers `503` so the UI can fall back to the browser microphone.

**v1.6 notes.** `POST /api/evidence/{id}/pack[?unblur=<token>]` returns a zip attachment: `clip.mp4`, three frames (start, peak with the box, end), `evidence.json` (dual timestamps, clock source, the question or watch that produced it), `manifest.json` (provenance, source file hashes, per-file hashes), `SHA256SUMS` (`sha256sum -c SHA256SUMS` after unzipping) and `README.txt`. Headers: `X-Evora-Blur` (`applied` or `off`) and `X-Evora-Pack-SHA256`. Packs are blurred, or not exported: when face blur is on but unavailable the answer is `409` unless an unblur token (with a reason) is supplied. Verify offline with `python -m evora.evidence.pack verify <pack.zip>`.

**v1.7 notes.** Replay-as-live. `GET /api/live` returns `{server:{ready, port, binary_found}, streams:[{camera_id, url, state, speed, restarts, transcoding, started_at, error}]}`. `POST /api/live/replay {camera_ids, speed?}` (speed 0.25 to 8) starts a local MediaMTX and loops each recorded file as `rtsp://127.0.0.1:8554/<camera_id>`; `POST /api/live/replay/stop {camera_ids?}` stops some or all. Errors: 404 unknown camera, 409 a camera that is already a live source or a busy port or too many streams, 422 bad speed, 503 MediaMTX not installed (the message contains the install command). Stream state changes arrive on `/api/events` as `note` `kind="live"` `{camera_id, state}` (`starting`, `running`, `retrying`, `failed`, `stopped`). Streams bind to loopback only.

**v1.8 notes.** Live analysis and live tiles (additive). `POST /api/live/replay` takes an optional `analyze` (default false): it also starts live analysis of the same cameras. `POST /api/live/analyze {camera_ids}` and `POST /api/live/analyze/stop {camera_ids?}` start and stop analysis of cameras that are live (an RTSP camera, or a recorded file whose replay is running); errors: 404 unknown camera, 409 a file whose replay is not running, 403 on-prem mode and the stream is not on this machine, 422 bad id, 503 perception stack not installed (the message contains the install command). `GET /api/live` adds `analyzers:[{camera_id, state, error}]` with state `starting`, `running`, `retrying`, `stopped` or `error`; changes arrive on `/api/events` as `note` `kind="analysis"` `{camera_id, state, error}`. Analysis runs on the same camera id, so remembered places, zones and standing queries apply, and a live event raises alerts exactly like recorded footage (but a phone may be notified). `GET /api/cameras/{id}/live.mjpg?fps=2&unblur=<token>` is now real: a `multipart/x-mixed-replace; boundary=frame` stream of JPEGs (fps 1 to 5, width 640), every frame face-blurred under the same policy and `X-Evora-Blur` header (`applied`, `off`, `unavailable`) as the media routes; a frame that cannot be blurred is dropped, never sent raw. At most `live.max_tiles` tiles at once (503 beyond), 409 when the camera is not live, 403 as above. Evidence of replayed-live footage maps wall-clock time to a position in the file (the replay loops), so thumbnails and clips of live alerts play; `offset_s` is that position.

**v1.9 notes.** Operator views (read-only, additive). `GET /api/audit?action=&limit=100` newest first: `[{id, actor, action, detail, t}]`. `GET /api/queries?limit=50&full=` past questions newest first: `[{id, text, intent, verdict, count, n_results, confidence, timings_ms, created_at}]`; `full=true` adds `plan` and `answer`; a damaged stored row still lists with null fields. `GET /api/zones/{zone_id}/events?kind=&since=&until=&limit=200` events of that zone newest first with `payload` parsed (404 unknown zone). `GET /api/globals?cls=&limit=100` and `GET /api/globals/{gid}`: `{id, cls, label, n_tracks, cameras, t_first, t_last}` (404 unknown identity); `/api/globals/{gid}/path` is unchanged. `GET /api/doctor` runs the quick health checks of `make doctor` against this running app and returns `{verdict, ok, checks:[{id, title, status, detail, fix}], took_s}` (no network probes; on-prem follows the app setting).
