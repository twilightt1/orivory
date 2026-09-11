---
name: orivory-memory
description: Persistent memory for OpenClaw agents via a self-hosted Orivory MCP server. Use when the user asks to remember, forget, or recall past decisions and history.
version: 1.1.0
homepage: https://github.com/twilightt1/orivory
emoji: "🧠"
metadata:
  openclaw:
    primaryEnv: ORIVORY_TOKEN
    envVars:
      - name: ORIVORY_TOKEN
        required: true
        description: Bearer token for your self-hosted Orivory MCP endpoint (register an agent client at POST /api/v1/agents to get one).
      - name: ORIVORY_URL
        required: false
        description: Override the MCP endpoint URL if it is not set in the mcp.servers config (default http://localhost:8000/mcp).
---

# Orivory Memory

Give your OpenClaw agent a persistent, user-owned second brain. Orivory is a
self-hosted memory hub: the agent reads and writes long-term memory through
MCP tools, every access is recorded in a ledger the user can audit
("which AI saw what, when"), and forgetting is verifiable (erasure receipts).

## Quick Start

### Install

The skill itself installs from this folder (or ClawHub once published):

```bash
openclaw skills install ./skills/orivory
```

### Configure

Orivory's memory tools reach the agent through an **MCP server** entry, not
through the skill (skills cannot declare MCP endpoints). Add Orivory once in
`~/.openclaw/openclaw.json`:

```json
{
  "mcp": {
    "servers": {
      "orivory": {
        "url": "http://localhost:8000/mcp",
        "transport": "streamable-http",
        "headers": {
          "Authorization": "Bearer ${ORIVORY_TOKEN}"
        }
      }
    }
  }
}
```

Or via the CLI:

```bash
openclaw mcp add orivory \
  --url http://localhost:8000/mcp \
  --transport streamable-http \
  --header "Authorization: Bearer ${ORIVORY_TOKEN}"
```

Set the token in your OpenClaw runtime env (never in shell profiles):

```bash
# one agent token per agent — see what each one accessed in the ledger
export ORIVORY_TOKEN="oa_<32-hex from POST /api/v1/agents>"
```

### Verify

```bash
openclaw mcp doctor orivory --probe
```

Expect the seven Orivory tools to list: `search_memory`, `get_memory`,
`list_recent`, `add_memory`, `correct_memory`, `delete_memory`, `forget_memory`. Then ask your
agent: *"What tools do you have for memory?"*

## Core tasks

- **Remember** — "remember that we decided to use pgvector for the hub" →
  `add_memory` with a clear title; confirm once, then stop narrating.
- **Recall** — "what did we decide about the vector store last week?" →
  `search_memory` with the user's words (not paraphrased); cite what you find
  or say you don't recall — never invent.
- **Forget** — "forget the note about the old API key" → `forget_memory`
  (or `delete_memory` for a single known id); the erasure is receipted.
- **Resurface** — "anything I saved about Postgres indexing?" →
  `search_memory`, then `get_memory` before quoting at length.

See `examples/` for full prompts and `references/tool-catalog.md` for every
tool's arguments and when to use which.

## Auto-capture (optional, recommended)

Hook your agent's session activity into Orivory so memories flow in without
manual calls. Two supported paths:

1. **`scripts/openclaw_capture.py` (stdlib-only daemon)** — watches your
   OpenClaw session/memory directory for new and changed Markdown files,
   converts them to the OpenClaw session-JSON shape, and POSTs them to
   `POST /api/v1/imports` using an agent token (`memory:write` scope).
   Dedup is end-to-end: the local state file skips unchanged files and the
   endpoint skips `(user, type, ref)` duplicates.

   ```bash
   python3 scripts/openclaw_capture.py --watch ~/.openclaw/workspace \
       --url http://localhost:8000 --token "$ORIVORY_TOKEN" --interval 30
   ```

   `--once` runs a single scan; `--dry-run` prints what would be captured.
   See `scripts/openclaw_capture.py --help` for all flags.

2. **MCP write tools directly** — agents that can call tools should use
   `add_memory` per fact (the tool-call path is already ledgered).

## Error handling

`references/error-handling.md` covers: `401` (token expired/revoked — re-register
the agent client), empty results (say so; offer to save), timeouts (Orivory may
be starting — probe with `openclaw mcp doctor`), and `422` invalid ids.

## Security notes

- The token is scoped (`memory:read` / `memory:write`) and revocable instantly
  (`DELETE /api/v1/agents/{id}`) — one token per agent, never shared.
- Every read/write lands in the user's access ledger; the user can see which
  agent accessed what. Do not batch speculative reads.
- Forgetting is the user's call: prefer `forget_memory` (receipted) over raw
  deletes when the intent is "erase this".
- Orivory is self-hosted — the memory never leaves the user's infrastructure.
