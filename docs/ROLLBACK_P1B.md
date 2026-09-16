# Rolling a P1b install back to the pre-P1b stack

**This escape hatch exists for ONE release** (spec §12). After that release the
Qdrant path is the only path, §1-§4 and `scripts/rollback_to_chroma.py` are
deleted, and the P1b-compatible restore is `verify --restore-drill` alone. §6
(P2 -> pre-P2) and §7 (P4a -> pre-P4) are the exceptions: see §5 for their
removal condition.

P1b cuts over by flipping the generation pointer: the SQLite database keeps
serving, but the vectors moved to Qdrant and the local embedding contract went
from masked-mean to CLS. Going back therefore needs **two** things, not one:

1. the **pre-P1b binary** (its readers resolve `Orivory_memories` and
   `rag_conv_<conversation_id>` in a local Chroma directory), and
2. a **Chroma store built at the legacy mean contract** from the live database —
   which is what `scripts/rollback_to_chroma.py` produces.

A Qdrant snapshot, a `cp -r` of the old Chroma directory, or restoring the
`.pre-p1b.bak` snapshot alone all fail: the first two do not speak the old
binary's protocol, the third silently drops every write made after the backup.

---

## 1. Run the rebuild (isolated venv)

`chromadb` is not a runtime dependency after P1b (Task 7 removed it from
`pyproject.toml` and `uv.lock`; `requirements-rollback.txt` is now its only
pin), so run the tool from its own venv — **never** install the pin into the
runtime image:

```bash
python -m venv .venv-rollback
.venv-rollback/bin/pip install -r requirements.txt -r requirements-rollback.txt

# stop the app first: the tool refuses while APP_PORT answers
.venv-rollback/bin/python scripts/rollback_to_chroma.py \
    --db /data/orivory.db \
    --chroma-path /data/chroma-rollback
```

Arguments:

| flag | meaning |
| --- | --- |
| `--db` | the **live** SQLite database (never the snapshot) |
| `--chroma-path` | where the rebuilt Chroma store is written; must be a free DIRECTORY (a file, or a non-empty directory, is refused) |
| `--fingerprint` | `legacy-mean` only — the contract the old binary expects |
| `--out-db` | where the SQLite copy goes (default: beside the store); must not already exist |
| `--batch` | rows per embedding batch (default 64) |
| `--namespace` | which memory namespace to export (default `personal` — P4a's only namespace) |

Exit codes: `0` rebuilt and every gate passed; `1` rebuilt, a gate failed (see
the marker below); `2` refused before doing anything.

`--chroma-path` must name a **free directory**: a path that is a file, or a
directory that already holds anything, is refused before a byte is written.
Both artifacts belong to the run that made them: a **failed** build removes the
store directory it created (an empty directory you passed in is removed with
it) and the emitted copy — the source database is untouched either way — and
neither artifact of a completed build is ever overwritten or merged into.

### What it emits

- `<chroma-path>/` — `Orivory_memories` (one collection, `user_id` filter) plus
  one `rag_conv_<conversation_id>` collection per conversation that has eligible
  chunks, all stamped with the legacy mean contract
  (`orivory_embed_fingerprint` = the mean canonical contract, `orivory_embed_dim`
  = 384, `hnsw:space = cosine`).
- `<db>.rollback.db` (or `--out-db`) — a copy of the **live** database with:
  - the eligible set's rows only, as index data — a superseded, dirty,
    suppressed or unowned row has no vector, so a correction cannot be undone
    and a forgotten source cannot be resurrected;
  - `index_generations` holding exactly one row per kind — `Orivory_memories`
    (kind `memory`) and the chunk family row — at the mean token; **every P1b
    row is deleted**, so the copy never claims a generation whose data is not in
    this store. Rolling forward again re-creates those rows (`cutover`);
  - the `memory_suppressions` ledger carried over verbatim, so the old binary
    keeps refusing to re-import a forgotten source (spec §5.4/§12.3);
  - `PRAGMA user_version = 2` — without this the old ladder refuses to boot the
    file at all.

The tool never edits the live database or the `.pre-p1b.bak` snapshot beside
it: it reads the live database once through a read-only handle (a killed
process can leave a WAL that only opens read-write — that fallback may
checkpoint the WAL, but it never changes committed content), and every edit
lands on the emitted copy. Post-cutover writes are preserved because the source
is the live DB.

The chunk collections are rebuilt from `document_chunks` (never skipped): every
row the live reader would index — the migration CLI's eligibility rule, one
definition — grouped by its conversation. Note that this set includes parent
chunks, which the pre-P1b ingest path only put in Redis/BM25; they carry
`chunk_type: "parent"` in their metadata if an operator wants to filter them
out downstream.

### Readiness gates

Before the report says ready, the rebuilt store is read back and checked:
tenant isolation (per-tenant ID sets, plus a filtered query that must not cross
a tenant), the ID set against the eligible set, the legacy mean contract on
every point and collection, corrections (superseded/dirty absent) and forgets
(suppressed absent). The contract check is the OLD binary's own
`check_collection_dim` guard, not a weaker cousin: `orivory_embed_backend =
local-arctic`, `orivory_embed_dim = 384`, the mean fingerprint,
`orivory_embed_generation = fingerprint_generation(fingerprint)` and
`hnsw:space = cosine` on every collection, plus the mean generation on every
point. A failure writes `NOT-READY.json` **into the store** with the findings
and exits `1` — do not point the old binary at a store carrying that file.

Re-running is always a **fresh build**, never an overwrite: the store directory
and the emitted copy (`<db>.rollback.db` by default) both belong to the run that
made them. Give the second run its own paths (`--chroma-path
/data/chroma-rollback-2 --out-db /data/orivory.rollback-2.db`) or move/delete the
previous pair first — the refusal names the same remedy.

The arctic-mean assumption is also cross-checked against the **install's own
records**: the pointer in `<db>.p1b-expand-record.json` and any
`index_generations` row still naming the mean generation, which carries the
token it was built at. The report's `contract_evidence.mean` is `true` (the
records agree), `false` (they name a different contract — the tool refuses with
exit `2`) or `null` (no record names a pre-P1b contract: the assumption is
**unverified**, not confirmed).

## 2. What it is rolling back FROM

The report's `rollback_from` says which generation the install serves:

1. `<db>.p1b-rollback-marker.json` — `active` and `cutover_at` (written by
   `cutover`, **after** the transaction commits, so it can be missing if the
   process died in that window);
2. else `<db>.p1b-expand-record.json` — the pointer recorded before the expand;
3. else `source: "none"` — no sidecar exists (a hand-migrated install). The
   rollback still runs; the report just cannot name what it replaced.

If the marker is missing, the drill-down fallback is the expand record. If both
are missing, treat the rollback as un-narrated but still valid: the rebuilt
store and the emitted copy are self-describing (the manifest rows name the
store, and `user_version = 2` makes the era explicit).

## 3. Swap in and start the old binary

1. Stop the P1b app.
2. Point the pre-P1b binary's configuration at the emitted artifacts:
   `CHROMA_MODE=local`, `CHROMA_LOCAL_PATH=<chroma-path>`, and
   `DATABASE_URL=sqlite+aiosqlite:///<db>.rollback.db` — or move them into the
   deployment's canonical paths.
3. Start the old binary and confirm: a recall returns hits (not an empty list),
   the dimension guard does not raise `EmbeddingDimensionMismatch`, and a
   forgotten source stays forgotten.
4. Keep the Qdrant directory and the migrated database untouched until the
   decision is final — nothing here deletes them, and rolling forward again is
   `migrate_qdrant.py` on the live database as before.

Writes made **after** the rebuild are not in the emitted copy (it is a copy of
the live database at run time). Record the cutover point in the incident log
and re-run the rebuild if the rollback is delayed — a re-run is a fresh build,
so it needs its own `--chroma-path` and `--out-db`, or the previous pair moved
out of the way first (§1).

## 4. Restore drill (before you need it)

`verify --restore-drill` proves that the backup the migration CLI already took
can actually be restored — checksums, `PRAGMA integrity_check`,
`PRAGMA foreign_key_check`, the recorded embedding fingerprint and the
deletion/suppression ledger — and never writes into the backup volume:

```bash
python scripts/migrate_qdrant.py verify --restore-drill --dir /backups/p1b
# optional: --target /tmp/restore-check
```

The restore lands in a **new** directory (`<dir>.restore` by default), and the
report's `checks` are `checksum`, `integrity`, `foreign_keys`, `fingerprint`,
`ledger`, `absence`. It reports ready only when all of them pass:

- `ledger` fails when the manifest carries no ledger record, when the ledger
  table is missing from the snapshot, or when the ledger in the restored bytes
  differs from the record taken at backup time — a restore that cannot prove
  forgotten content stays forgotten is not a restore;
- `absence` fails when a `completed` erasure receipt names a memory the restore
  brings back, or when chunk rows have no document row (GC residue);
- `fingerprint` fails when the restored schema's `user_version` or its active
  generation contract differs from what the backup recorded.

Backups taken before this release have no `fingerprint`/`deletion_ledger`
records; re-running `backup --dir <same dir>` verifies the existing snapshot
and adds those records from the bytes (it never re-snapshots or overwrites).
That refresh can only recompute what the bytes still carry: a snapshot from a
pre-v2-era install has **no `memory_suppressions` table at all**, so there is no
ledger to recompute and the drill's `ledger` check fails closed — intended
(R33), because a restore that cannot prove forgotten content stays forgotten is
not a restore. Such a snapshot is not drillable: take a fresh `backup --dir
<new dir>` from the migrated install and keep the old snapshot as a last-resort
artifact only.

The drill's `--target` must be a directory that is empty or does not exist yet,
and it must sit **outside** `--dir` — the volume being verified is never
written to. Both rules are enforced before a byte is copied (default target:
`<dir>.restore`, beside the backup).

## 5. Removal condition

Keep this escape hatch for **exactly one release**. Delete §1-§4 (the Chroma
rebuild), `scripts/rollback_to_chroma.py`, `requirements-rollback.txt` and
`tests/migration/test_p1b_rollback.py` in the release after P1b (spec §12), and
drop the retired store itself — the `LEGACY_CHROMA_PATH` directory
(`Orivory_memories` + `rag_conv_*`) — once the window closes: nothing serves from
it after `cutover`, so it is only disk at that point. From that release the only
supported restore is `verify --restore-drill` plus a Qdrant snapshot restore.

**§6 is NOT part of that deletion.** The P2 rollback recipe (drop the FTS5
objects, re-stamp `user_version = 3`) has no other home and is the only
documented way back from a v4 file to a pre-P2 binary — deleting it in the
release after P1b would remove the recipe in exactly the release that ships P2.
The same holds for §7 (the P4a recipe): it ships with the phase that creates the
need. Before §1-§4 go, move §6 and §7 into
[OPERATIONS_RUNBOOK.md](OPERATIONS_RUNBOOK.md) (and re-point this document's
references to them).

## 6. Rolling a P2 install back to a pre-P2 binary

A P2 (lite) install runs SQLite `user_version = 4`: the v3 schema plus the P2
lexical index — `memory_fts`, an FTS5 virtual table with its shadow tables, and
the three `memories_fts_*` triggers that maintain it. **None of that is model
metadata**: the virtual table is created by the ladder's own DDL, so the v4-only
objects are exactly one virtual table (plus SQLite's shadow tables) and three
triggers.

A pre-P2 binary cannot open such a file: its ladder accepts `0..3` and refuses
with `unsupported SQLite schema version 4; expected 3`. Going back to the P1b
stack therefore needs the same two moves the rebuild makes for the pre-P1b
stack (`user_version = 2`, §1) — **stamp the version the target ladder tops out
at, and take the objects it cannot know about with it**:

```bash
# On a COPY, never on the live file. Verify with PRAGMA integrity_check after.
sqlite3 /data/orivory.rollback-p2.db <<'SQL'
DROP TRIGGER IF EXISTS memories_fts_ai;
DROP TRIGGER IF EXISTS memories_fts_au;
DROP TRIGGER IF EXISTS memories_fts_ad;
DROP TABLE IF EXISTS memory_fts;     -- drops the FTS5 shadow tables too
PRAGMA user_version = 3;             -- the P1b ladder's top
SQL
```

- `user_version = 3` is what a P1b-era reader boots: its ladder stops there, and
  the v2/v3 objects in the copy are the ones it built. `2` (the copy
  `scripts/rollback_to_chroma.py` emits) is the pre-P1b binary's value — use
  whatever the binary you are going back to understands, and nothing higher.
- Drop the FTS objects rather than leaving them inert: the triggers fire on
  **every** memory write, so an old binary would be writing into a table it does
  not know and an SQLite build without FTS5 would fail the write outright.
- Rolling forward again is a boot with a P2 binary: the ladder re-runs the
  `v3 -> v4` step (the DDL is `IF NOT EXISTS`, the backfill is coverage-driven),
  so the index comes back without a rebuild of the memories table.
- If a full restore is acceptable instead, the ladder's own `.pre-p2.bak`
  snapshot beside the database is the pre-P2 bytes — and, like `.pre-p1b.bak`,
  restoring it silently drops every write made after it was taken.

## 7. Rolling a P4a install back to a pre-P4 binary

A P4a (lite) install runs SQLite `user_version = 5`: the v4 schema plus
`memories.namespace` (`VARCHAR(32) NOT NULL DEFAULT 'personal'`) and the
`ix_memories_namespace_user(namespace, user_id)` index. Both come from the
ladder's own DDL (model metadata does not create them on an existing file).

A pre-P4 binary cannot open such a file: its ladder accepts `0..4` and refuses
with `unsupported SQLite schema version 5; expected 4`. Going back needs the
same two moves as §6 — **stamp the version the target ladder tops out at, and
take the objects it cannot know about with it**:

```bash
# On a COPY, never on the live file. Verify with PRAGMA integrity_check after.
sqlite3 /data/orivory.rollback-p4.db <<'SQL'
DROP INDEX IF EXISTS ix_memories_namespace_user;  -- SQLite refuses to drop an
ALTER TABLE memories DROP COLUMN namespace;       -- INDEXED column: index first
PRAGMA user_version = 4;                          -- the pre-P4 ladder's top
SQL
```

- The stamp ALONE (`user_version = 4`, column left in place) is the minimal
  rollback and is safe for a pre-P4 binary: it never names `namespace`, so its
  `SELECT`s simply read the extra column and any insert that omits it gets the
  column default `'personal'`. The test that pins the round trip
  (`tests/retrieval/test_p4a_gate.py`) does exactly this. The DDL above is the
  conservative variant: the old binary then sees the v4 shape it built itself.
- Rolling forward again is a boot with a P4a binary: the ladder re-runs the
  `v4 -> v5` step, which INSPECTS the column instead of re-adding it (SQLite has
  no `ADD COLUMN IF NOT EXISTS`), reuses the operator's existing `.pre-p4.bak`
  rather than overwriting it, and never re-asserts a namespace an operator moved
  by hand.
- `.pre-p4.bak` beside the database is the pre-namespace bytes — and, like the
  other milestone snapshots, restoring it silently drops every write made after
  it was taken. On an install whose ladder carried several steps in ONE boot,
  the snapshot's NAME is not its version stamp (mid-ladder `VACUUM INTO` copies
  can hold later objects at an earlier `PRAGMA user_version`): check the stamp
  inside before pointing an older binary at it.

