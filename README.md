# evora

Multi-camera video intelligence with conversational queries (HNX26EPS05).

## Quick start

```
make setup     # install backend deps (uv, Python 3.11+)
make doctor    # is this machine ready to demo? every problem comes with the command that fixes it
make up        # check, start local services, serve the API and the built UI on http://127.0.0.1:8700
make check     # ruff + pytest + generated TS types in sync
```

More targets: `make dev` (API with reload; `evora_MOCK=1 make dev` serves fixtures for UI work), `make models`
(download local models while online), `make offline-test` (the suite with Wi-Fi-off rules), `make eval`, `make ablate`.
Pass options with `ARGS`, for example `make up ARGS="--live --open"` or `make doctor ARGS="--on-prem"`.

Judge day: follow Appendix C of the plan; the first step is `make doctor` all green.

## Layout

- `contracts/` frozen v1 contracts: pydantic models, SQLite schema, vector tables, API spec, fixtures, generated TS types. Changes go through PROGRESS.md (contract change requests).
- `backend/evora/` Python package. `config/` tunables and profiles. `docs/` write-up and demo script.
- `PROGRESS.md` live team state.

Copy `.env.example` to `.env` and fill in your own keys. Never commit `.env`.
