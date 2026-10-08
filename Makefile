UV ?= $(shell command -v uv >/dev/null 2>&1 && echo uv || echo python -m uv)
BACKEND = $(UV) --directory backend

.PHONY: setup dev check test test-e2e eval ablate types types-check models doctor offline-test up

setup:
	$(BACKEND) sync

dev:
	$(BACKEND) run uvicorn evora.api.app:create_app --factory --reload --host 127.0.0.1 --port 8700

types:
	$(BACKEND) run python ../scripts/gen_types.py ../contracts/ts/schema.json
	npx --yes json-schema-to-typescript@15 contracts/ts/schema.json -o contracts/ts/evora-types.ts

types-check:
	$(BACKEND) run python ../scripts/gen_types.py ../contracts/ts/.schema.check.json
	@cmp -s contracts/ts/schema.json contracts/ts/.schema.check.json || { rm -f contracts/ts/.schema.check.json; echo "contracts/ts is stale: run make types"; exit 1; }
	@rm -f contracts/ts/.schema.check.json

check:
	$(BACKEND) run ruff check .
	$(BACKEND) run pytest -q
	$(MAKE) types-check

test:
	$(BACKEND) run pytest -q

test-e2e:
	$(BACKEND) run pytest -q tests/e2e

eval ablate models:
	@echo "make $@: owned by M2/M3, not wired yet"; exit 1

doctor offline-test up:
	@echo "make $@: M1 phase 3, not wired yet"; exit 1
