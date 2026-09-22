# Contributing to Orivory

Thank you for your interest in contributing to Orivory! This guide will help you get started.

---

## 🎯 How Can I Contribute?

### 1. 🐛 Reporting Bugs

Before opening a bug report:
- Search [existing issues](https://github.com/twilightt1/orivory/issues) to avoid duplicates
- Use the [bug report template](.github/ISSUE_TEMPLATE/bug_report.md)
- Include: steps to reproduce, expected/actual behavior, environment details

### 2. 💡 Suggesting Features

- Check the [roadmap](docs/ROADMAP.md)
- Open a [feature request](.github/ISSUE_TEMPLATE/feature_request.md)
- Explain the use case and benefits

### 3. 🔧 Pull Requests

We welcome code contributions! Follow these steps:

```bash
# 1. Fork the repository
# 2. Clone your fork
git clone https://github.com/YOUR_USERNAME/orivory.git
cd orivory

# 3. Create a feature branch
git checkout -b feature/your-feature-name

# 4. Make your changes
# ... write code ...

# 5. Run tests
python -m pytest tests -q --ignore=tests/integration

# 6. Run linting
python -m ruff check app tests eval scripts

# 7. Commit with clear messages
git commit -m "feat: add new feature"

# 8. Push and create PR
git push origin feature/your-feature-name
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.13 (see `.python-version`)
- Docker & Docker Compose

### Quick Start

```bash
# Clone repository
git clone https://github.com/twilightt1/orivory.git
cd orivory

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Start development server — no services to start first: the SQLite schema
# ladder and the embedded Qdrant folder are created on boot; durable state
# lands in the path DATABASE_URL / QDRANT_LOCAL_PATH point at.
uvicorn app.main:app --reload
```

`docker compose up -d` is the same product containerised (one service, `app`);
`make quickstart` builds and runs it.

### Running Tests

```bash
# The CI-safe suite (skips the live-stack integration modules)
python -m pytest tests -q --ignore=tests/integration

# Run specific test file
python -m pytest tests/api/test_memories_router.py -q

# The live-stack modules need the API on :8000 and an explicit opt-in
RUN_LIVE_INTEGRATION=1 python -m pytest tests/integration -q
```

### Code Style

We use **Ruff** for linting:

```bash
# Check code style
python -m ruff check app tests eval scripts

# Auto-fix issues
python -m ruff check app tests eval scripts --fix
```

---

## 📝 Commit Message Format

We follow [Conventional Commits](https://www.conventionalcommits.org/):

| Type | Description |
|------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `style:` | Formatting, no code change |
| `refactor:` | Code refactoring |
| `test:` | Adding tests |
| `chore:` | Maintenance tasks |

**Examples:**
```bash
git commit -m "feat: add timeline tool to the MCP hub"
git commit -m "fix: keep the erase receipt open when the readback fails"
git commit -m "docs: update API documentation"
```

---

## 📁 Project Structure

```
orivory/
├── app/                    # Main application
│   ├── api/v1/            # API endpoints: router.py + users, memories,
│   │                      #   agents, erasure, imports
│   ├── mcp_hub/           # MCP server, tools, agent-token identity
│   ├── models/            # SQLAlchemy models (SQLite-shaped)
│   ├── services/          # Business logic (erasure, imports, digest, …)
│   ├── retrieval/         # RAG retrieval, memory spine, drain loop
│   ├── ingestion/         # Upload/import pipelines
│   ├── observability/     # Cost/salience tracking
│   └── middleware/        # Logging, rate limiting
├── tests/                # Test suite
├── scripts/              # Utility scripts
├── eval/                 # Offline evaluation + benchmarks
└── docs/                 # Documentation
```

## 🌱 Starter tasks (`good first issue` candidates)

Scoped, verified entry points — each names files + done-criteria. Maintainers:
promote these to GitHub issues with the `good first issue` label.

| # | Task | Files | Done when |
|---|---|---|---|
| 1 | Startup embedding-dim check (fail-fast at boot, today the guard only fires on next write/query) | `app/retrieval/embedder.py`, `app/main.py` lifespan | Mismatched `EMBED_*` config vs the active vector-store generation refuses boot with a clear message; test with fake collection |
| 2 | HTTP-level scoping tests: a token without `memory:write` must be refused by `POST /api/v1/imports` over real HTTP, and the ledger row must carry the refusal | `tests/api/`, `tests/conftest.py` | ASGI-transport matrix over the mounted routers (local owner vs each scope), all 401/403/404 as appropriate |
| 3 | Convert one legacy eval one-shot into an entrypoint CLI | `eval/run_real_sample.py` (or siblings listed in `eval/README.md`) | Same outputs via argparse flags; README table updated; no new scripts |
| 4 | Document the single-process cost ledger | `app/observability/` cost tracker + `docs/OPERATIONS_RUNBOOK.md` | Limitation + workaround (keep workers at 1 / read the SQLite ledger directly) written where an operator will find it |

Rules for all of them: TDD (RED test first), `ruff` clean, no new infra.

---

## 🏷️ Issue Labels

| Label | Description |
|-------|-------------|
| `bug` | Bug reports |
| `enhancement` | New features |
| `documentation` | Documentation improvements |
| `good first issue` | Easy tasks for newcomers |
| `help wanted` | Seeking contributors |
| `question` | Questions and discussions |

---

## ❓ Questions?

- Open a [Discussion](https://github.com/twilightt1/orivory/discussions)
- Check [existing issues](https://github.com/twilightt1/orivory/issues)

---

## 📜 Code of Conduct

By participating, you agree to maintain a welcoming and respectful environment for everyone.

---

<div align="center">

**Thank you for contributing to Orivory!** 🚀

*Your AI Second Brain from the Stars* ⭐🐺

</div>
