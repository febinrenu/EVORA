# Session rules for evora (local file, never commit)

evora is a multi-camera video intelligence system with conversational queries, built for hackathon problem HNX26EPS05 by a team of four (M1 Platform & Memory, M2 Perception & Identity, M3 Reasoning & Science, M4 Experience). The full plan is in `PLAN.md` (local-only). Live team state is in `PROGRESS.md` (committed).

If the human has not said which member they are, ask: "Which member are you, M1, M2, M3 or M4?" before doing anything else.

## Read order at the start of every session

1. This file.
2. `PROGRESS.md`: your status block and your latest HANDOFF line first, then Decisions, Contract change requests, Requests, then the rest.
3. `PLAN.md` §0–§6, §9, §12, and your own section (§8.1 M1, §8.2 M2, §8.3 M3, §8.4 M4; M4 also all of §10).

## Hard rules

1. Never mention Claude, Anthropic, AI assistants, language models used for coding, or how code was produced, in any commit message, PR text, code comment, docstring, doc, `PROGRESS.md` entry, or any other tracked file. Write as the human engineer who owns the change. (Describing the product's own use of language models, such as the Groq planner, is fine.)
2. Never stage or commit `CLAUDE.md`, `PLAN.md`, `.claude/`, `.env`, or anything under `data/`, `models/`, `workspaces/`.
3. `contracts/` is frozen v1. Only M1 edits it, via PLAN.md §5.8. Others request changes in `PROGRESS.md` under "Contract change requests".
4. Write only inside your ownership area (PLAN.md §5.1) plus your own `PROGRESS.md` block and new lines in shared `PROGRESS.md` sections. For changes elsewhere, add a line under "Requests".
5. Run `make check` before every commit. Commit small with Conventional Commits, for example `feat(query): fast-path parser for time phrases`.
6. After each finished task, add one `PROGRESS.md` log line with the task ID and short commit hash. Before the session ends, before a model switch, or when context is getting long, add a HANDOFF line (PLAN.md §12.5). Always `git pull --rebase` before editing `PROGRESS.md`, then commit and push it immediately.
7. Never read videos, images, large logs or whole datasets into context. Use `ffprobe`, `head`, counts and small summaries.
8. Secrets live only in `.env`. Never print, log or commit API keys.
9. In product code, every outbound network call goes through `backend/evora/llm/gateway.py` (the privacy guard depends on it).
10. Use plan mode for any task longer than about 30 minutes or touching more than three files. In plans, add an "Improvements" section for ideas beyond PLAN.md inside this member's area, with impact, cost in minutes, and cross-area effects. Implement improvements only after the human approves.

## Quality bar

- Python 3.11, typed, pydantic v2 at every boundary, `ruff` clean, `pytest` for logic, no bare `except`, structured logging, config via `config/*.yaml` and `.env` (no hard-coded paths or thresholds).
- Times are epoch seconds UTC; image coordinates are normalized 0..1.
- Frontend: TypeScript strict, no `any` at API boundaries, types generated from `contracts/`, design tokens and rules from PLAN.md §10 only. No default component-kit look, no gradient washes, no all-caps labels.
- Tests prove behaviour, not just imports. A task is done when its tests pass in `make check` or `make test`.

## Commands

`make setup` · `make dev` · `make check` · `make test` · `make test-e2e` · `make eval` · `make ablate` · `make models` · `make doctor` · `make up`
