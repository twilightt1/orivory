# Orivory — The Open Memory Hub

<div align="center">

![Python](https://img.shields.io/badge/Python-3.13+-blue?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js-14-000000?style=for-the-badge&logo=next.js&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**Your memories, your infrastructure, any AI agent — with receipts.**

*Orivory = Open + Memory + Discovery*

</div>

---

## What is Orivory?

Orivory is an open-source, self-hosted **memory hub for AI agents**: a single
place where everything you know gets stored once — and every AI agent you use
(Claude Desktop, Cursor, OpenClaw, your own scripts) reads and writes it
through MCP.

Three ideas make it different from a chat-with-docs app:

1. **Proactive, not just reactive.** Orivory surfaces your memories back to
   you — salience-weighted recall, "on this day" digests, knowledge-graph
   connections — instead of letting notes rot in a landfill.
2. **Governed, not vibes.** Every agent gets its own scoped token
   (`memory:read` / `memory:write`); every access lands in an audit ledger
   ("which AI saw what, when"); forgetting returns a **verifiable erasure
   receipt**.
3. **Provable, not marketed.** The eval harness ships with the repo
   (LongMemEval-S + MemoryAgentBench adapters) so quality claims can be
   checked, not just claimed — including two **negative** results we
   published anyway. Current best: **0.570** LongMemEval-S (n=100, single
   pass + semantic rerank) vs 0.490 baseline; results with Wilson 95% CIs
   in [`eval/benchmarks/results/`](eval/benchmarks/results/), history in
   [CHANGELOG](CHANGELOG.md).

Your data stays on your infrastructure. MIT-licensed, self-hosted, plain
Postgres + ChromaDB under the hood.

## Core features

| Area | What you get |
|---|---|
| **🧠 Memory store** | Evidence-first correctable memory: sửa 1 lần mọi agent nhớ đúng (supersede có scope/thời gian hiệu lực), stale view tự chặn, time-aware recall, hybrid vector + keyword search |
| **🕸️ Knowledge graph** | Automatic entity extraction, relation mapping, cluster detection |
| **🔌 MCP hub** | 7 tools — `search/get/timeline/list_recent/add/correct/delete/forget`: any MCP-capable agent (Claude Desktop, Cursor, OpenClaw…) connects with a scoped per-agent token — see [skills/orivory](skills/orivory/SKILL.md) |
| **📜 Access ledger** | Append-only audit log: which agent read or wrote which memory, when |
| | |
| **🗑️ Erasure receipts** | Right-to-be-forgotten with verification: cascade deletion across rows, links and vectors, re-checked and receipted |
| **📥 Import paths** | One-shot upload of ChatGPT / Claude / PAM / generic-JSON exports with dedup |
| **📊 Benchmarks** | LongMemEval-S + MemoryAgentBench harness, 0.570 (n=100) with CIs — [results](eval/benchmarks/results/), [how to run](eval/README.md) |
| **👥 Workspaces** | Shared knowledge bases with workspace-level access control |
| **📊 Analytics** | Usage tracking, DAU metrics, cost monitoring |

## Quick start

### Lite mode — one container, zero external services (personal/demo use)

The whole memory hub — API + MCP server + SQLite + in-process Chroma — in a
single container. No Postgres, no Redis, no MinIO, no workers.

```bash
curl -fsSL https://raw.githubusercontent.com/twilightt1/orivory/main/install.sh | bash
```

or plain docker:

```bash
docker run -d --name orivory -p 8000:8000 -v orivory-data:/data \
  -e OPENAI_API_KEY=sk-... ghcr.io/twilightt1/orivory:lite
```

Or from a clone: `make quickstart`. Then:

- **App**: http://localhost:8000 · MCP endpoint: http://localhost:8000/mcp
- Connect an agent below — that's the whole setup.

Lite mode is single-user by design (personal brain). Data persists in the
`orivory-data` volume; the JWT secret is ephemeral per container.
It is the fastest way to try Orivory — not a production tier: anything
multi-user, multi-instance, or load-bearing belongs on the full stack below.

### Full stack — Postgres + Redis + ChromaDB + MinIO + workers + UI

```bash
git clone https://github.com/twilightt1/orivory.git
cd orivory
cp .env.example .env            # add your LLM API key(s)

docker compose up -d            # migrations run automatically (migrate gate),
                                # then app, celery workers, frontend
```

- **App**: http://localhost:8000 · API docs: http://localhost:8000/docs
- **Frontend**: http://localhost:3000 · **Flower**: http://localhost:5555

Health & self-diagnosis: `/health` (liveness) and `/ready` — deployment-aware
per-dependency checks (postgres/sqlite, redis, minio/storage, chroma, mcp_hub).

### Connect an AI agent (both modes)

```bash
# 1. Register an agent client (as your logged-in user) — token shown ONCE
curl -X POST http://localhost:8000/api/v1/agents \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"name": "Claude Desktop", "scopes": ["memory:read", "memory:write"]}'
```

```json
// 2. Point the agent at the MCP endpoint (Claude Desktop / OpenClaw / …)
{
  "mcp": {
    "servers": {
      "orivory": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable-http",
        "headers": { "Authorization": "Bearer ${ORIVORY_TOKEN}" }
      }
    }
  }
}
```

Every tool call (`search_memory`, `add_memory`, `forget_memory`, …) is now
scoped to that token and recorded in the ledger.

### Bring your old brain along

```bash
curl -X POST http://localhost:8000/api/v1/imports \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -F "file=@conversations.json" -F "source_format=chatgpt"
```

ChatGPT, Claude, PAM bundles and generic JSON are supported. Leave anytime —
plain Postgres + JSON everywhere, export or query your data directly.

## Project layout

```
orivory/
├── app/                    # FastAPI backend
│   ├── api/v1/             # REST (memories, agents, erasure, imports, …)
│   ├── agents/             # LLM client + parsing seams (LangGraph agents removed)
│   ├── mcp_hub/            # MCP server: identity, scoped tools, ledger
│   ├── ingestion/          # connectors + import format adapters
│   ├── models/             # SQLAlchemy models
│   ├── retrieval/          # hybrid retrieval + memory vector store
│   └── services/           # domain services
├── skills/orivory/         # OpenClaw/ClawHub skill package
├── eval/                   # RAG eval framework + benchmarks/
├── docs/                   # architecture, API reference, guides, research
└── docker-compose.yml      # full stack with migrate gate + healthchecks
```

## Documentation

| Doc | Contents |
|---|---|
| [docs/API.md](docs/API.md) | Full API reference: auth, chat, memories, MCP hub, erasure, imports |
| [docs/how-it-works.html](docs/how-it-works.html) | The one-page explainer — how the hub works, honestly compared | 
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture — hub spine, agents, retrieval, data model |
| [docs/ROADMAP.md](docs/ROADMAP.md) | Shipped milestones and open follow-ups |
| [docs/ideas/open-source-positioning.md](docs/ideas/open-source-positioning.md) | Positioning: one-line definition, competitor matrix, hub-first narrative |
| [docs/research/](docs/research/) | Market / user / platform / papers research behind the pivot |
| [docs/EVALUATION_GUIDE.md](docs/EVALUATION_GUIDE.md) | RAG evaluation + benchmarks |
| [docs/LITE_MODE.md](docs/LITE_MODE.md) | One-container lite mode |
| [docs/DEPLOYMENT_GUIDE.md](docs/DEPLOYMENT_GUIDE.md) · [docs/OPERATIONS_RUNBOOK.md](docs/OPERATIONS_RUNBOOK.md) · [docs/BACKUP_RESTORE.md](docs/BACKUP_RESTORE.md) | Ops |

## Contributing

We welcome contributions — see [CONTRIBUTING.md](CONTRIBUTING.md). The
`good first issue` label marks self-contained starting points. Security
issues: see [SECURITY.md](SECURITY.md) (responsible disclosure, no public
issues).

## License

MIT — see [LICENSE](LICENSE).
