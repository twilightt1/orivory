# Orivory memory tools — catalog

All tools are exposed at the Orivory MCP endpoint (`/mcp`, streamable HTTP).
Read tools require the `memory:read` scope; write tools require
`memory:write`. Every authorized call is recorded in the user's access
ledger with your agent's name — act as if the user is watching, because
they can.

| Tool | Scope | Args | Returns | When to use |
|---|---|---|---|---|
| `search_memory` | read | `query: str`, `limit: int = 8` | `{results: [{id, title, content, salience, captured_at}], query}` | The default entry point for "what do I know about X". Pass the user's own words. |
| `list_recent` | read | `limit: int = 20` | recent memories, newest first | "What have I saved lately?" / browsing after a save. |
| `get_memory` | read | `memory_id: str` | full memory row | You have an id (from search) and need the whole content before quoting. |
| `add_memory` | write | `title: str`, `content: str`, `tags?: [str]` | created memory summary | "Remember this…" — one fact per memory, title = the searchable handle. Never use add to overwrite a changed fact — use `correct_memory`. |
| `correct_memory` | write | `content: str`, `title?: str`, `subject?: str`, `attribute?: str`, `scope?: str`, `memory_id?: str`, `valid_from?: str`, `evidence_ids?: [str]` | new version summary (`status`, `superseded`, provenance) | "That fact changed…" — creates a linked new version; the old one stays as history. Pass `memory_id` when you know the row you are correcting: the tool then refuses with `status: "conflict"` if another writer moved that slot since you read it (nothing is superseded — re-read and decide) instead of silently un-learning the newer fact. |
| `delete_memory` | write | `memory_id: str` | deletion confirmation | Removing one known memory (hard delete + receipt). For "erase this about me", prefer `forget_memory` (soft, receipted). |
| `forget_memory` | write | `memory_ids: [str]` | soft-forget summary (`receipt_id`, `status`, `invalidated`, `suppressed`, `skipped`, `invalid`) | Right-to-be-forgotten: invalidates the memory and its derived chain, pins every affected source against re-import, keeps provenance, and returns a receipt. Rows are NOT deleted — serving stops; `erase`/`delete_memory` is the hard path. |

## Choosing between similar tools

- `search_memory` vs `list_recent`: search when the user names a topic;
  list when they mean "lately/recently".
- `delete_memory` vs `forget_memory`: delete is the HARD path for one
  identified item (row + descendants + vectors, verification receipt);
  forget is the SOFT one when the user's intent is "I don't want this
  known/stored" — it invalidates the target and its derived chain, preserves
  the provenance, blocks re-import of the affected sources, and returns a
  serving-verified receipt. Say which one you mean when the user is
  ambiguous: forget leaves the row in history, delete does not.
- `correct_memory` vs `add_memory`: correct when a stored fact changed
  (new version links back to the old one, which stays as history); add
  when the fact is new. Never overwrite via add.
- Honesty: Orivory versions its own store — copies the agent already
  repeated outside Orivory are unrecoverable; say so when asked.
- Never write without a clear title: the title is what `search_memory`
  ranks on next week.

## Answer discipline

1. Search with the user's phrasing first; refine once if empty.
2. Quote memories with their `title` so the user can trace them.
3. If results are empty or off-topic: say "I don't recall that in your
   memories" and offer to save — do not fill gaps from general knowledge
   while wearing the memory-hat.
