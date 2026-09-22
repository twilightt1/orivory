.PHONY: dev build up down test lint lint-full lint-fix format security-check check lite-build lite-run quickstart

# ── Lite mode (one container, zero external services) ────────────────────────

lite-build:
	docker build -t ghcr.io/twilightt1/orivory:lite .

# The whole memory hub in one container: API + MCP server, SQLite, in-process
# Qdrant, in-memory caches, filesystem uploads. No Postgres/Redis/MinIO/Qdrant server.
lite-run:
	docker run -d --name orivory-lite -p 127.0.0.1:8000:8000 -v orivory-data:/data \
		-e OPENAI_API_KEY=$${OPENAI_API_KEY:-} ghcr.io/twilightt1/orivory:lite

# One command from clone to running hub (builds first if no image yet).
quickstart: lite-build lite-run
	@echo "Orivory is up:  http://localhost:8000  (MCP: /mcp)"

# ── Compose (the same single-container lite stack) ──────────────────────────

dev:
	uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

# CI-safe unit tests only (skips the live-infra integration suite).
test:
	python -m pytest tests -q --ignore=tests/integration

# Lint the paths CI lints (the same full-repo check).
lint: lint-full

# Full-repo lint (Phase 1-3 remediation pass).
lint-full:
	python -m ruff check app tests eval scripts

# Auto-fix safe lint issues across app/, tests/, eval/, and scripts/.
lint-fix:
	python -m ruff check app tests eval scripts --fix

format:
	ruff format app/

security-check:
	python scripts/security_check.py

# Convenience: run tests + lint + security check together.
check: test lint-full security-check
