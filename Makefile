.PHONY: dev build up down migrate seed test lint lint-full lint-fix format demo-smoke security-check check lite-build lite-run quickstart

# ── Lite mode (one container, zero external services) ────────────────────────

lite-build:
	docker build -f Dockerfile.lite -t ghcr.io/twilightt1/orivory:lite .

# The whole memory hub in one container: API + MCP server, SQLite, in-process
# Qdrant, in-memory caches, filesystem uploads. No Postgres/Redis/MinIO/Qdrant server.
lite-run:
	docker run -d --name orivory-lite -p 8000:8000 -v orivory-data:/data \
		-e OPENAI_API_KEY=$${OPENAI_API_KEY:-} ghcr.io/twilightt1/orivory:lite

# One command from clone to running hub (builds first if no image yet).
quickstart: lite-build lite-run
	@echo "Orivory is up:  http://localhost:8000  (MCP: /mcp)"

# ── Full stack (Postgres + Redis + Qdrant + MinIO + workers + UI) ───────────

dev:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build

migrate:
	alembic upgrade head

makemigration:
	alembic revision --autogenerate -m "$(name)"

worker:
	celery -A app.tasks.celery_app worker -Q default,ingestion,email -c 4 -l INFO

beat:
	celery -A app.tasks.celery_app beat -l INFO

# CI-safe unit tests only (skips live infra and Redis-bound auth tests).
test:
	python -m pytest tests -q --ignore=tests/integration --ignore=tests/api/test_auth_api.py --ignore=tests/api/test_admin_api.py --ignore=tests/test_auth.py

# Targeted lint matching CI's narrow path list.
lint:
	ruff check app/main.py app/config.py app/agents app/services/health_service.py app/services/diagnostics_service.py app/storage.py app/tasks/ingestion_tasks.py app/retrieval app/api/v1/chat.py app/api/v1/sse.py app/api/v1/admin.py eval/run_eval.py eval/live_api_eval.py eval/metrics.py eval/reporting.py tests/api/test_health_api.py tests/api/test_sse.py tests/api/test_chat_streaming.py tests/api/test_admin_diagnostics.py tests/api/conftest.py tests/rag/test_graph_routing.py tests/rag/test_evaluation.py tests/rag/test_integration.py tests/rag/test_ai_hardening.py tests/rag/conftest.py tests/services/test_health_service.py tests/services/test_diagnostics_service.py tests/services/conftest.py tests/eval/test_eval_metrics.py tests/eval/test_live_api_eval.py tests/config/test_settings_validation.py tests/integration

# Full-repo lint (Phase 1-3 remediation pass).
lint-full:
	python -m ruff check app tests eval scripts

# Auto-fix safe lint issues across app/, tests/, eval/, and scripts/.
lint-fix:
	python -m ruff check app tests eval scripts --fix

format:
	ruff format app/

demo-smoke:
	python scripts/demo_smoke.py

security-check:
	python scripts/security_check.py

# Convenience: run tests + lint + security check together.
check: test lint-full security-check
