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
pytest tests/ -v

# 6. Run linting
ruff check app/ tests/

# 7. Commit with clear messages
git commit -m "feat: add new feature"

# 8. Push and create PR
git push origin feature/your-feature-name
```

---

## 🛠️ Development Setup

### Prerequisites

- Python 3.12+
- Docker & Docker Compose
- PostgreSQL 15+, Redis 7+

### Quick Start

```bash
# Clone repository
git clone https://github.com/twilightt1/orivory.git
cd orivory

# Start infrastructure
docker compose up -d

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt -r requirements-dev.txt

# Run migrations
alembic upgrade head

# Start development server
uvicorn app.main:app --reload
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/api/test_auth.py -v

# Run with coverage
pytest tests/ --cov=app --cov-report=html
```

### Code Style

We use **Ruff** for linting:

```bash
# Check code style
ruff check app/ tests/

# Auto-fix issues
ruff check --fix app/ tests/
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
git commit -m "feat: add referral system"
git commit -m "fix: resolve auth token refresh issue"
git commit -m "docs: update API documentation"
```

---

## 📁 Project Structure

```
orivory/
├── app/                    # Main application
│   ├── api/v1/            # API endpoints
│   │   ├── auth.py        # Authentication
│   │   ├── chat.py        # Chat & recall
│   │   ├── memories.py    # Memory management
│   │   ├── insights.py    # AI insights
│   │   ├── discovery.py   # Knowledge discovery
│   │   ├── referral.py    # Referral system
│   │   └── analytics.py   # Analytics
│   ├── models/            # Database models
│   ├── services/          # Business logic
│   ├── agents/            # LangGraph agents
│   └── retrieval/         # RAG retrieval
├── frontend/              # Next.js frontend
├── tests/                # Test suite
├── scripts/              # Utility scripts
└── docs/                 # Documentation
```

## 🌱 Starter tasks (`good first issue` candidates)

Scoped, verified entry points — each names files + done-criteria. Maintainers:
promote these to GitHub issues with the `good first issue` label.

| # | Task | Files | Done when |
|---|---|---|---|
| 1 | Startup embedding-dim check (fail-fast at boot, today the guard only fires on next write/query) | `app/retrieval/embedder.py`, `app/main.py` lifespan | Mismatched `EMBED_*` config vs the active vector-store generation refuses boot with a clear message; test with fake collection |
| 2 | HTTP-level behavioral security tests (ROADMAP #2 remainder: cross-user 404, ledger scoping over real HTTP) | `tests/api/`, `tests/conftest.py` | Matrix of user-A vs user-B resource access over ASGI transport, all 403/404 as appropriate |
| 3 | Migrate raw `fetch(` calls to the shared `apiClient` | `frontend/src/app/settings/page.tsx`, `forgot-password`, `login`, `share/[id]`, `ProactiveInsightToast/Widget`, `ReferralDashboard`, `AuthProvider` | No direct `fetch(` outside `lib/api-client.ts`; `tsc --noEmit` clean |
| 4 | Convert one legacy eval one-shot into an entrypoint CLI | `eval/run_real_sample.py` (or siblings listed in `eval/README.md`) | Same outputs via argparse flags; README table updated; no new scripts |
| 5 | Document cost-ledger multi-worker limitation | `app/observability/` cost tracker + `docs/OPERATIONS_RUNBOOK.md` | Limitation + workaround (single worker / external DB) written where an operator will find it |

Rules for all five: TDD (RED test first), `ruff` clean, no new infra.

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
