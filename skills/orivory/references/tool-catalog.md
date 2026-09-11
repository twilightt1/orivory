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
| `correct_memory` | write | `content: str`, `title?: str`, `subject?: str`, `attribute?: str`, `scope?: str`, `memory_id?: str`, `valid_from?: str`, `evidence_ids?: [str]` | new version summary (`status`, `superseded`, provenance) | "That fact changed…" — creates a linked new version; the old one stays as history. |
| `delete_memory` | write | `memory_id: str` | deletion confirmation | Removing one known memory. For "erase this about me", prefer `forget_memory` (receipted). |
| `forget_memory` | write | `memory_ids: [str]` | receipt summary (id, status, erased, skipped, invalid) | Right-to-be-forgotten: cascades across rows, links, vectors and returns a verifiable receipt. |

## Choosing between similar tools

- `search_memory` vs `list_recent`: search when the user names a topic;
  list when they mean "lately/recently".
- `delete_memory` vs `forget_memory`: delete for surgical removal of one
  item you already identified; forget when the user's intent is erasure
  ("I don't want this known/stored") — it cascades children, links and
  vectors and the user gets a receipt.
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
