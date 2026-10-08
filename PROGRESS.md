# PROGRESS — evora (HNX26EPS05)

Single source of truth for live team state. Committed. Merged with `merge=union` (see `.gitattributes`).

How to use this file:
- Edit only your own status block. Add new lines to shared sections; never rewrite other people's lines.
- `git pull --rebase` before editing, then commit (`docs(progress): ...`) and push immediately.
- Times in IST, 24-hour, `[HH:MM]`. Member tags: `[M1]` `[M2]` `[M3]` `[M4]`.
- Task IDs follow the plan (P1.x, P2.x, P3.x, P4.x).

Team: M1 Adhu (Platform, Memory & Integration) · M2 Johann (Perception & Identity) · M3 ________ (Reasoning, Retrieval & Science) · M4 ________ (Experience)
Ingestion box: ________ (GPU: ________) · Start time: ____ · Freeze: start + 12:45 · Submit: start + 15:00

---

## Status board (each member edits only their own block)

### M1 — Platform, Memory & Integration
- State: on track
- Doing: P1.7 done; starting P1.8 memory KB + resolve
- Next: P1.8, P1.9 wire /api/query and /api/clarify to M3's router
- Blockers:

### M2 — Perception & Identity
- State: on track
- Doing: P2.4-P2.10 perception pipeline (decode, clock, motion, detect, track, crops, embed, pipeline); detect_clock and ingest for M1
- Next: P2.3 detector spike on the RTX 4060, then golden mini-index for M3
- Blockers: none; MEVA and WILDTRACK downloads still running

### M3 — Reasoning, Retrieval & Science
- State: on track
- Doing: P3.13 (path/describe/by-example), P3.17 ablation runner skeleton; waiting on M1 memory (resolver/clarifier) and M2 mini-index to wire `ours` into eval
- Next: wire Router into M1 /api/query (P1.9), run retrieval on the M2 mini-index and calibrate (P3.16), register ours/b0 in eval/harness.py, first make eval
- Blockers: M2 mini-index (+ SigLIP2 text embedder) for real retrieval numbers; M1 memory.resolve for the router. Groq: 4 keys verified live (gpt-oss-20b/120b, qwen3.8-27b, whisper all listed)

### M4 — Experience
- State: not started
- Doing:
- Next: P4.1–P4.3
- Blockers:

State values: not started · on track · at risk · blocked · done

---

## Checkpoints
| Gate | Target | Result | Notes |
|---|---|---|---|
| cp0 contracts + fixtures | start + 1:00 | pass | tag cp0; `make check` green (26 tests) |
| CP1 thin slice | start + 4:20 | | |
| CP2 eval #1 + dry run | start + 9:30 | | |
| Judge simulation | start + 12:45 | | |
| Freeze tag | start + 14:00 | | |

---

## Eval scoreboard (append rows; never edit old ones)
| Time | Commit | Split | Hit@1 | Hit@5 | MRR | Cam acc | Ts err (s) | Neg prec | Re-asks | TTFA p50 (ms) | Baseline Hit@1 | Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

## Throughput (append rows)
| Time | Commit | Machine | Profile | Layer | Video-s per s per camera | Notes |
|---|---|---|---|---|---|---|

---

## Decisions (append only)
- [HH:MM] [M1] DECISION: project name evora, Python package `evora`. Why: short, local, meaningful. Impact: none.
- [16:06] [M2] DECISION: naive times in file names are read in Asia/Kolkata (clock.default_tz). Why: the team and the judging venue are in IST. Impact: M3 time anchoring uses epoch seconds, so only the file-name interpretation depends on it.

## Contract change requests (append only)
- [11:40] [M1] REQUEST (self-approved, additive): table `evidence` in schema.sql and `POST /api/media/unblur`; media routes accept `?unblur=<token>` and answer with header `X-Evora-Blur`. Why: media routes must resolve an evidence id; unblur must be audited. Affects: M3 (register evidence), M4 (blur header). → [11:40] [M1] APPROVED v1.1 (07c7e79)
<!-- - [HH:MM] [M3] REQUEST: add optional `Answer.followups: list[str]`. Why: UI suggestions. Affects: M1, M4. → [HH:MM] [M1] APPROVED v1.1 -->

## Requests to other areas (append only)
- [12:10] [M1] → M3: after composing an answer, call `ctx.prerender.schedule([e.id for e in answer.evidence])` (M1 wires this in P1.9 when it relays your `answer` event, so no action needed from you unless you call the router directly).
- [11:40] [M1] → M3: call `evora.evidence.store.register(db, evidence)` for every `Evidence` you return (including `nearest_miss`). Until then media falls back to scanning stored answers, which only works for answers saved in `query_log`.
- [11:40] [M1] → M2: expose `blur_faces(jpeg: bytes) -> bytes` from `evora.perception` (or `.clock` / `.pipeline`). Until it exists, media is served unblurred with header `X-Evora-Blur: unavailable`.
- [11:40] [M1] → M4: show a visible warning when a media response has `X-Evora-Blur: unavailable`; the unblur flow is `POST /api/media/unblur {reason}` then `?unblur=<token>` (5 minute token, audited).
- [10:55] [M1] → M2: expose `detect_clock(path) -> (t0, source)` and `ingest(cam, profile, layers, on_progress)` from `evora.perception` (or `.clock` / `.pipeline`). M1 calls them through `core/perception_adapter.py`, which uses a simulated stub until they exist. `on_progress` takes an `IngestJob` (layer + progress 0..1; progress 1.0 marks that layer finished).
<!-- - [HH:MM] [M4] → M1: /api/cameras should include thumbnail URL. → [HH:MM] [M1] done (abc1234) -->
- [16:06] [M2] → M1: add perception dependencies to backend/pyproject.toml (av, opencv-python-headless, numpy, pillow, torch, torchvision, ultralytics, transformers, tzdata, boxmot) and an `ingest:` block in config/default.yaml (keys in backend/evora/perception/settings.py, all have defaults). Also: jobs.py should call add_layers only for layers ingest finished (progress 1.0); L2 and L3 are not implemented yet and ingest skips them.
- [16:06] [M2] → M1: media paths I store (crops.crop_path, scenes.frame_path, tracks.best_crop) are relative to the workspace media/ directory, e.g. crops/cam_01/t000001_0.jpg. Tell me if the media service expects another base.

## Known issues (append; mark fixed with commit)

## Datasets and models status (append)

---
- [16:06] [M2] EPFL terrace1, passageway1 and 6p (12 clips) downloaded and normalised to H.264 mp4 under data/norm/epfl; manifest at data/manifest.json. MEVA 2018-03-05 11:05-11:10 slice and WILDTRACK zip downloading.

## Log (append only, newest at the bottom)
- [HH:MM] [M1] Repository created; PROGRESS.md initialised.
- [09:57] [M1] P1.1 scaffold done (a0b9cd5): uv project, config profiles, Makefile (setup dev check test types), README.
- [09:57] [M1] P1.2 contracts v1 + fixtures + generated TS types done (e70d847). CONTRACT v1: pull and code against `contracts/`; use `contracts/fixtures/*.json` for mocks.
- [09:57] [M1] P1.3 API skeleton done (cea238f): every §5.5 route on :8700 serving fixtures; `make dev`. Tagged cp0.
- [09:57] [M1] HANDOFF: next is P1.4 (core/db.py, vectors.py, workspace.py), then P1.5 upload. Windows note: `make` via winget ezwinports.make; `uv` via `python -m uv` (Makefile handles both).
- [10:20] [M1] P1.4 core db/vectors/workspace done (29b0148): migrations from contracts/schema.sql, WAL + one writer per file, LanceDB tables sized from meta dims, workspaces confined to workspaces/. 40 tests in make check.
- [15:45] [M3] P3.1 done: scripts/check_groq.py lists models + rate-limit headers per key, keys masked (e227b00)
- [15:45] [M3] P3.2 done: llm gateway with key pool, 429 rotation, circuit, repair retry, Ollama fallback, on-prem path; 34 tests (b794ec8)
- [15:45] [M3] P3.3 done: planner prompt v0, 12 examples, static prefix ~2.2k tokens, byte-identical across calls (7efb1a2)

- [15:50] [M3] P3.4 done: query/fastpath.py, 93 table cases (phrasings + refusals), conservative: unknown word -> LLM (1bc99fd)
- [15:50] [M3] P3.5 done: query/planner.py + timeparse.py: fast path -> plan cache (sqlite plan_cache, keyed on text + camera set) -> gateway; time anchored to reference_now (meta.reference_now, else end of latest footage; meta.tz); 218 tests pass in full backend suite (9b891a5)
- [10:55] [M1] P1.5 camera upload done (1d565b9): extension allow-list, size cap, ffprobe validation, SHA-256, rtsp registration, DB-backed /api/cameras. Fixed `make dev` (contracts import) on the way.
- [10:55] [M1] P1.6 ingest job runner done (1d2d26c): resumable by layer, one failing camera never stops the others, progress as `note` events on /api/events (kind=ingest|camera). 257 tests in make check after merging M3.
- [16:30] [M3] P3.7 done: query/logic.py (pass_through/enter/exit/dwell/appear over line/polygon/frame zones, event-time matching, tod filters that wrap midnight, first/last/count distinct); 25 tests (83e2ec4)
- [16:30] [M3] P3.8 done: query/compose.py deterministic answers, dual timestamps (clock + offset into file), notes for unindexed/IR cameras, evidence-id validator; 27 tests (358a46e)
- [16:30] [M3] P3.10 done: eval/queries.py + metrics.py + harness.py (all section 9.3 metrics, PROGRESS scoreboard rows) and baseline/b0_frames.py core with injected embedder; 54 tests (01af89e, 764ff52, 7369acd)
- [16:30] [M3] NOTE: ours/b0 systems are not yet registered in eval/harness.py build_system (needs P3.6 retrieval, P3.9 router and an indexed workspace). No new contract changes needed so far.
- [11:40] [M1] CONTRACT v1.1: evidence table + unblur endpoint (07c7e79). Pull and regenerate nothing: TS types are unchanged.
- [11:40] [M1] P1.7 media service done (31e0d13, store e7ebd64): real frames, cached thumbnails with box, clips (3 s pre/post roll, faststart, HTTP range), face blur hook with honest `unavailable` state, audited unblur tokens. Bug fixed on the way: seeking past the last frame returned nothing.
- [16:06] [M2] P2.1 and P2.2 done: fetch_epfl, fetch_wildtrack, fetch_meva, transcode, manifest and models_download scripts (0dbf4b8). One-line fix in .gitignore: `data/` -> `/data/` because it also ignored scripts/data.
- [12:10] [M1] P1.5/P1.7 improvements done (88f5477), approved by Adhu: byte-identical uploads reuse the camera (`X-Evora-Duplicate`), unusual codecs (Indeo, MPEG-2...) are converted to H.264 on upload with an audit entry, media cache capped (LRU, `media.cache_max_bytes`), top-3 thumbnails and clips pre-rendered via `ctx.prerender`.
- [17:40] [M3] P3.6 done: query/retrieve.py + fuse.py. Track-centric (max-mean over crops), attributes (IR-aware), BM25 captions, scene support, per-camera fallback to scene windows while L1 is missing; switches unit/attributes/expansion; 24 tests on a real LanceDB (cfd7dc7)
- [17:40] [M3] DECISION: final score is a weighted blend of absolute signals, not reciprocal-rank fusion. Why: the not-found threshold (P3.12) needs a magnitude and RRF throws it away. Impact: weights/calibration are config, to be tuned on dev (P3.16).
- [17:40] [M3] DECISION: planner prompt uses a compact plan shape + plain JSON mode (about 2.1k tokens) instead of sending the full JSON schema too (about 3.1k). Measured live: 5/5 valid plans. Why: 8K tokens per minute per org. Also keypool now sticks to one key while it has room. Impact: none for others.
- [17:40] [M3] Approved improvements done: LLM record/replay (zero Groq calls on eval reruns), no-model share metric in eval, synonym expansion (off|lexicon|llm). Keys: GROQ_KEY-1..N and GROQ_KEYS both accepted.
- [17:40] [M3] → M1 REQUEST (additive): add an optional `retrieval:` section to config/default.yaml: unit (track|frame), attributes (true|false), expansion (off|lexicon|llm), weights {image, attributes, caption, scene}, calibration {midpoint, scale}, ann_k, scene_k, pool_limit. M3 reads it with RetrievalConfig.from_cfg; every key has a default, so nothing breaks until it exists.
- [17:40] [M3] → M2 REQUEST: expose the SigLIP2 TEXT encoder as `embed_text(str) -> np.ndarray` (L2-normalised, same space as the crops/scenes vectors) and set meta.embed_dim_image; fill tracks.attrs with the TrackAttrs JSON and tracks.best_t. Retrieval needs only those. Crops with cls 'person' etc. must match the detector class names in the plan.
- [17:40] [M3] → M1: noted your evidence.store.register and prerender requests; the router (P3.9) will call both for every Evidence including nearest_miss.
- [18:05] [M3] P3.15 done: scripts/meva_to_queries.py (find + generate) tested on the real kitware annotations; 11 tests (see git log).
- [18:05] [M3] → M2 IMPORTANT: the MEVA slice being downloaded (2018-03-05 11:05-11:10) has NO annotations. Kitware annotations for 2018-03-05 exist only for hours 13 and 14. For ground truth pick a window that is annotated: `python scripts/meva_to_queries.py find --min-cams 6` after cloning gitlab.kitware.com/meva/meva-data-repo into data/raw/meva-annotations. Best: 2018-03-09 10-10-00 school, 11 annotated cameras, 469 activity instances (also 2018-03-11 16-35-01 with 10, 2018-03-11 11-20-00 with 9 cameras and 935 instances). Please fetch that window (>= 4 of its cameras) and tell me the camera ids you give them, so I can map MEVA codes (G340...) to workspace ids with --camera-map.
- [18:50] [M3] P3.9 + P3.12 done: query/router.py streams plan, clarify|evidence*, answer, verified*, done in the contract order; asks once and stops, resume re-runs the original question; accept threshold (RouterConfig.accept, tune on dev) turns weak matches into a nearest miss; every Evidence incl. nearest_miss is registered with evidence.store; query_log written; 19 tests incl. a real SQLite + LanceDB workspace
- [18:50] [M3] P3.11 done: query/verify.py (numbered contact sheet, one vision call, streams after the answer). Checked live through the Groq vision fallback. Needs OpenCV in the backend environment (M2's detector stack brings it; M1: please add opencv-python-headless to backend/pyproject.toml if not already).
- [18:50] [M3] → M1: to wire P1.9 build `Router(db, Planner(gateway, SqlitePlanCache(db)), Retriever(db, store, text_embedder, RetrievalConfig.from_cfg(cfg), gateway), resolver, clarifier, verifier, RouterConfig(), reference_override=lambda: settings['reference_now'])`; `async for ev in router.answer(text, session_id)` and `router.resume(resp)` yield StreamEvent. Protocols you implement (evora/query/router.py): Resolver.resolve(ref) -> object with .status in bound|ambiguous|unknown and .facts (list[MemoryFact]); Clarifier.ask(query_id, text, plan, ref, resolution, options) -> ClarifyRequest (persist the pending query) and Clarifier.resume(resp) -> (text, plan) | None (bind the fact first). Place fact binding {camera_id, zone_id}, time fact {tod_after, tod_before}, object fact {global_id}.
