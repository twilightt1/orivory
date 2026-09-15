# Rolling a P1b install back to the pre-P1b stack

**This escape hatch exists for ONE release** (spec §12). After that release the
Qdrant path is the only path, this document and `scripts/rollback_to_chroma.py`
are deleted, and the P1b-compatible restore is `verify --restore-drill` alone.

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

`chromadb` is not a runtime dependency after P1b, so run the tool from its own
venv — **never** install the pin into the runtime image:

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
| `--chroma-path` | where the rebuilt Chroma store is written; must be free |
| `--fingerprint` | `legacy-mean` only — the contract the old binary expects |
| `--out-db` | where the SQLite copy goes (default: beside the store) |
| `--batch` | rows per embedding batch (default 64) |

Exit codes: `0` rebuilt and every gate passed; `1` rebuilt, a gate failed (see
the marker below); `2` refused before doing anything.

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

The tool never writes the live database or the `.pre-p1b.bak` snapshot beside
it; the copy is read once, read-only, and all edits land on the emitted file.
Post-cutover writes are preserved because the source is the live DB.

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
(suppressed absent). A failure writes `NOT-READY.json` **into the store** with
the findings and exits `1` — do not point the old binary at a store carrying
that file. Re-run the rebuild (with another `--chroma-path`) once the finding is
understood.

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
and re-run the rebuild if the rollback is delayed.

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

## 5. Removal condition

Delete this document, `scripts/rollback_to_chroma.py`,
`requirements-rollback.txt` and `tests/migration/test_p1b_rollback.py` in the
release after P1b (spec §12). From that point the only supported restore is
`verify --restore-drill` plus a Qdrant snapshot restore.
