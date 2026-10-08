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
