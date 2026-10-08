# PROGRESS — evora (HNX26EPS05)

Single source of truth for live team state. Committed. Merged with `merge=union` (see `.gitattributes`).

How to use this file:
- Edit only your own status block. Add new lines to shared sections; never rewrite other people's lines.
- `git pull --rebase` before editing, then commit (`docs(progress): ...`) and push immediately.
- Times in IST, 24-hour, `[HH:MM]`. Member tags: `[M1]` `[M2]` `[M3]` `[M4]`.
- Task IDs follow the plan (P1.x, P2.x, P3.x, P4.x).

Team: M1 Adhu (Platform, Memory & Integration) · M2 ________ (Perception & Identity) · M3 ________ (Reasoning, Retrieval & Science) · M4 ________ (Experience)
Ingestion box: ________ (GPU: ________) · Start time: ____ · Freeze: start + 12:45 · Submit: start + 15:00

---

## Status board (each member edits only their own block)

### M1 — Platform, Memory & Integration
- State: on track
- Doing: P1.5/P1.6 done; starting P1.7 media service
- Next: P1.7 thumbs/clips/range/blur, P1.8 memory KB + resolve
- Blockers:

### M2 — Perception & Identity
- State: not started
- Doing:
- Next: P2.1–P2.3
- Blockers:

### M3 — Reasoning, Retrieval & Science
- State: on track
- Doing: P3.7 logic.py, P3.8 compose.py (pure, tested on synthetic rows)
- Next: P3.10 eval harness + B0, then P3.6 retrieve+fuse once M2 mini-index lands
- Blockers: need M1 core/db.py + vectors.py (P1.4) and M2 mini-index (~T+3:00) for real retrieval; Groq key not yet run through check_groq

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

## Contract change requests (append only)
<!-- - [HH:MM] [M3] REQUEST: add optional `Answer.followups: list[str]`. Why: UI suggestions. Affects: M1, M4. → [HH:MM] [M1] APPROVED v1.1 -->

## Requests to other areas (append only)
- [10:55] [M1] → M2: expose `detect_clock(path) -> (t0, source)` and `ingest(cam, profile, layers, on_progress)` from `evora.perception` (or `.clock` / `.pipeline`). M1 calls them through `core/perception_adapter.py`, which uses a simulated stub until they exist. `on_progress` takes an `IngestJob` (layer + progress 0..1; progress 1.0 marks that layer finished).
<!-- - [HH:MM] [M4] → M1: /api/cameras should include thumbnail URL. → [HH:MM] [M1] done (abc1234) -->

## Known issues (append; mark fixed with commit)

## Datasets and models status (append)

---

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
