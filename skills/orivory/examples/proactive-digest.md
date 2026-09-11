# Example: Proactive weekly digest (scheduled agent workflow)

Goal: every Monday morning, the agent reviews the week's saved memories and
sends the user a short digest — without being asked.

## Setup (one-time, user-side)

Use OpenClaw's cron/scheduled-task support (or the user's own scheduler) to
run a prompt like the one below every Monday 08:00, with the Orivory MCP
server connected.

## The prompt

> It's Monday morning. Run my weekly memory digest:
> 1. `list_recent` with `limit: 30` — everything saved in the last 7 days.
> 2. Group by topic (title prefixes are the hint).
> 3. For each group, one line: the topic, the count, and the single most
>    important memory (its `get_memory` content, first sentence).
> 4. End with one question back to me: which of these should we go deeper on?
>
> Do not invent items. If nothing was saved, say "a quiet week — nothing
> captured" and stop.

## Why this is Orivory's shape of "proactive"

- **Read-only by default** — the digest agent needs only `memory:read`; give
  it a separate agent token so its every access is separately visible in the
  ledger ("which AI saw what").
- **The user can see the schedule**: digests are just scheduled reads — no
  background writes, no surprise mutations.
- **Actionable, not noise**: one question back per digest ("deeper on what?")
  is what keeps a digest from becoming homework — the failure mode that
  killed ChatGPT Pulse (see [PAPERS_HCI_PRIVACY.md §2](https://github.com/twilightt1/orivory-private/blob/main/docs/research/PAPERS_HCI_PRIVACY.md) (private):
  digests survive when user-configured + action-attached).

## Variations

- **"On this day"**: add `list_recent` filtered by the user manually scanning
  a year-old window (Orivory's digest endpoint does this server-side —
  `GET /api/v1/memories/digest`).
- **Project-scoped**: pass a `query` to `search_memory` ("project atlas")
  instead of `list_recent` for a per-project digest.
