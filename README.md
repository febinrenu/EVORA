# evora

Multi-camera video intelligence with conversational queries (HNX26EPS05).

## Quick start

```
make setup     # install backend deps (uv, Python 3.11+)
make dev       # API on http://127.0.0.1:8700 (serves contract fixtures until services land)
make check     # ruff + pytest + generated TS types in sync
```

## Layout

- `contracts/` frozen v1 contracts: pydantic models, SQLite schema, vector tables, API spec, fixtures, generated TS types. Changes go through PROGRESS.md (contract change requests).
- `backend/evora/` Python package. `config/` tunables and profiles. `docs/` write-up and demo script.
- `PROGRESS.md` live team state.

Copy `.env.example` to `.env` and fill in your own keys. Never commit `.env`.
