UV ?= $(shell command -v uv >/dev/null 2>&1 && echo uv || echo python -m uv)
BACKEND = $(UV) --directory backend

.PHONY: setup setup-perception frontend-check dev check test test-e2e eval ablate types types-check models doctor offline-test up

setup:
	$(BACKEND) sync

setup-perception:
	$(BACKEND) sync --extra perception --extra embed

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
	$(MAKE) frontend-check

frontend-check:
	@if [ -f frontend/package.json ] && command -v npm >/dev/null 2>&1; then 		cd frontend && { [ -d node_modules ] || npm ci; } && npm run check; 	else echo "frontend-check skipped (no frontend/ folder or no npm)"; fi

test:
	$(BACKEND) run pytest -q

test-e2e:
	$(BACKEND) run pytest -q tests/e2e

models:
	EVORA_MODELS_DIR=../models $(BACKEND) run python ../scripts/models_download.py $(ARGS)

eval:
	PYTHONPATH=.. $(BACKEND) run python -m eval.harness $(ARGS)

ablate:
	PYTHONPATH=.. $(BACKEND) run python -m eval.ablate $(ARGS)

offline-test:
	HF_HUB_OFFLINE=1 evora_ONPREM=1 $(BACKEND) run pytest -q tests/e2e tests/privacy

doctor:
	EVORA_MODELS_DIR=../models $(BACKEND) run python ../scripts/doctor.py $(ARGS)

up:
	EVORA_MODELS_DIR=../models $(BACKEND) run python ../scripts/up.py $(ARGS)
