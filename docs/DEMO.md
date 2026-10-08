# Demo runbook

The six-minute demo, written as an operator's checklist: what to do, what to say, what proves it, and what to do when it
goes wrong. Everything here uses commands and routes that exist today; anything that depends on unfinished work is
marked **partial** with the fallback to use.

Numbers quoted on stage come from `make eval` on the tagged commit, never from memory.

## 1. Before the judges arrive (30 minutes)

1. `make doctor` on the ingestion box. Fix every `FAIL`; read every `WARN` and decide if it matters today. Each row prints
   the command that fixes it. With Wi-Fi off use `make doctor ARGS="--on-prem"`.
2. Decide the privacy mode: cloud allowed, or on-prem (`evora_ONPREM=1`, or the toggle in the UI). With on-prem, Ollama
   must have its models (`ollama list` shows `qwen3.5:4b` and `qwen3-vl:2b` or `4b`).
3. Start from a fresh workspace so nothing pre-names "main gate": `evora_WORKSPACE=judge-set-1 make up ARGS="--live --open"`.
   The banner shows the API address (http://127.0.0.1:8700), the interface address, privacy mode and what is missing.
4. Put the judges' footage in `data/judge/` first and work from there, not from a USB stick.
5. Phone: subscribe to your ntfy topic and set `NTFY_TOPIC` in `.env` (skipped automatically in on-prem mode).
6. Backup video on the desktop. Browser at full screen; projector resolution checked.
7. Know your numbers: open the report page (or `eval/reports`) and have the main table, ablations and latency ready.

## 2. The script

| # | Time | You do | You say | What proves it |
|---|---|---|---|---|
| 1 | 0:00 | Drop the footage (`POST /api/cameras`, multipart). Read each detected clock and its source aloud; start `POST /api/ingest`. | "Four cameras, forty minutes, one question." | Camera rows show `t0_source` (filename, metadata, on-screen text, slate or manual). Progress arrives on `/api/events`; "Searchable now" is layer L0. |
| 2 | 0:40 | Ask "Did a red car pass through the main gate in the last hour?" (`POST /api/query`). A clarify card appears; choose the camera and draw the line (`POST /api/clarify`). Play the clip. | "It asks once, because it has never heard of the main gate." | Evidence has `t_peak` and `offset_s` (wall clock and time into the file), the circled object, a clip that plays at `/api/media/clip/{evidence_id}.mp4`. |
| 3 | 1:50 | Stop the server (Ctrl-C) and run `make up` again. Ask "any red vehicles through the main entrance after nine?" | "Nothing is asked twice, even after a restart." | No `clarify` event. `GET /api/memory` shows the place with the new alias marked as learned. |
| 4 | 2:40 | Ask "Where did the person with the large black bag go?" Open the path (`GET /api/globals/{gid}/path`). Click a frame and find similar (`GET /api/tracks/{id}/similar`). | "Identity across cameras, with the time it took to walk between them." | Hops in order with times; every hop plays. |
| 5 | 3:40 | `POST /api/standing` with "Notify me if anyone enters the parking zone after 8 pm". Start the replay (`POST /api/live/replay`). | "A standing question, running over replayed footage." | An alert note (`kind=alert`) on `/api/events`, the alert drawer, and the phone. |
| 6 | 4:30 | Switch on-prem (`POST /api/settings {"onprem": true}`), turn Wi-Fi off, ask again. Export an evidence pack (`POST /api/evidence/{id}/pack`), unzip, run `sha256sum -c SHA256SUMS`. | "Nothing leaves this machine. Faces are blurred. Every export is checkable." | `/api/health` shows `onprem` and `egress_blocked`; `X-Evora-Blur: applied`; the checksum command prints OK for every file. |
| 7 | 5:10 | Open the report page. | One sentence per contribution; quote the table. | `GET /api/report`. |

### Status of each beat today

- 1, 2, 3, 6: ready on the API. The interface for upload, clarify card and evidence sheet belongs to M4.
- 4: ready when re-identification has run (`make setup-perception`, models downloaded). Without it the path and similar
  routes answer an empty list with `X-Evora-Reid: pending`.
- 5: standing queries, alerts and the replay stream are ready. The live runner that consumes the replay stream waits for
  the perception team's live ingestion, so today alerts come from already processed footage (marked "found in earlier
  footage"). Say "replay of recorded footage" and do not claim live camera processing until the runner exists.
- 7: the report depends on M3's `make eval` output.

## 3. When something breaks

| Drill | What happens | What you do |
|---|---|---|
| Groq answers 429 on every key | The planner falls back to the fast path, then the local model; the answer carries a note. | Say so. Or switch on-prem (`POST /api/settings`). |
| Wi-Fi off | On-prem works end to end. Voice answers 503. | Type the question or use the browser microphone. |
| No GPU | CPU profile: L0 finishes first, L1 progressively. | Start answering as soon as "Searchable now" shows; mention refinement continues. |
| A file will not decode | The upload is rejected with the reason (no video stream, truncated file). | A truncated file cannot be repaired; ask for the original. Other odd codecs are converted automatically. |
| Clocks unknown | The clock source says `manual` and uses the file time. | Type the start time with `PATCH /api/cameras/{id}` and continue; both timestamps stay checkable. |
| Night or infrared footage | Colour is suppressed with a note. | Ask by class and place instead of colour. |
| Zero matches | A grounded "not found" with the nearest miss. | Show the nearest miss and its score. |
| Server restarted mid-clarification | The pending question is stored. | Answer the card again; it resumes. |
| Face blur unavailable | Media is served with `X-Evora-Blur: unavailable`; evidence packs refuse to export (409). | `python scripts/models_download.py --only yunet`, or ask for an unblur token with a reason. |
| Replay will not start | `POST /api/live/replay` says why (MediaMTX missing: install it; port in use: stop the other program). | `make doctor` shows both. |
| Total failure | | Play the backup video, show the report page, then `make up`. |

## 4. Cut list (decided at the second checkpoint)

Cut a beat rather than risk it live. In order of what to drop first: the phone push, the live replay (keep the alert on
screen from processed footage), the path animation, the report page. Never cut the clarify-once restart or the privacy
beat: they carry the most weight.

## 5. Judge-day sequence

1. Copy the footage, upload it, read each clock aloud, fix by hand if the judges give start times.
2. Name cameras exactly as the judges name them. Do not pre-name places: let the system ask.
3. Start ingest; begin answering when "Searchable now" appears.
4. Read each question back, type it verbatim, let clarifications happen, point at both timestamps and the circled object,
   play the clip, offer the evidence pack. For a not-found answer show the nearest miss.
5. If anything breaks: Groq errors, switch on-prem; ingest stalls, check `/api/events` and restart that camera's job;
   interface frozen, refresh (state is on the server); total failure, backup video then `make up`.
