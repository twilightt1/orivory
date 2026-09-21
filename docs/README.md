# Orivory Documentation

Start here. Everything else is organized by purpose:

## For users

| Doc | Contents |
|---|---|
| [../README.md](../README.md) | Product overview + quick start |
| [LITE_MODE.md](LITE_MODE.md) | One container (SQLite + embedded Qdrant, no external services) — the whole product |
| [LOCAL_RUN_GUIDE.md](LOCAL_RUN_GUIDE.md) | Run the stack locally (dev mode) |
| [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) | Production deployment |
| [OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) | Day-2 ops: logs, backups rotation, common failures |
| [BACKUP_RESTORE.md](BACKUP_RESTORE.md) | Backup/restore procedures |
| [ONBOARDING.md](ONBOARDING.md) | Product onboarding flow design + tours |

## Reference

| Doc | Contents |
|---|---|
| [API.md](API.md) | Full REST + MCP reference (§13 MCP hub, §14 erasure receipts, §15 import paths) |
| [ARCHITECTURE.md](ARCHITECTURE.md) | How the system works — memory spine, MCP hub, erasure, imports, agents, eval |
| [ROADMAP.md](ROADMAP.md) | Shipped milestones + open follow-ups |
| [../CHANGELOG.md](../CHANGELOG.md) | Notable changes per release |
| [../SECURITY.md](../SECURITY.md) | Responsible disclosure |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Contribution guide |

## Deep dives

| Doc | Contents |
|---|---|
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | RAG eval framework + benchmarks |
| [RAG_TECHNIQUES.md](RAG_TECHNIQUES.md) | Retrieval techniques used in the pipeline |
| [architecture/rag_pipeline_optimization.md](architecture/rag_pipeline_optimization.md) | RAG pipeline hardening notes |
| [../eval/benchmarks/README.md](../eval/benchmarks/README.md) | Benchmark protocol + leaderboard hygiene |

## Research & plans (private)

Strategy research, positioning one-pagers, and SDD implementation plans live
in the private companion repo:
[orivory-private](https://github.com/twilightt1/orivory-private)
([research/](https://github.com/twilightt1/orivory-private/tree/main/docs/research) ·
[ideas/](https://github.com/twilightt1/orivory-private/tree/main/docs/ideas) ·
[plans/](https://github.com/twilightt1/orivory-private/tree/main/docs/superpowers/plans)).
