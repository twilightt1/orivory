# Backup and Restore Guide

Orivory is ONE container and stores ALL durable state under `/data`:

| Component | Path | Data | Backup priority |
|---|---|---|---|
| SQLite | `/data/orivory.db` (+ `-wal`/`-shm`) | users, conversations, memories, documents, chunks, outbox | Critical |
| Uploads | `/data/uploads` | uploaded source documents | Critical |
| Qdrant | `/data/qdrant` | vector index | Important, rebuildable from documents/chunks |
| Caches / rate limits | process memory | nothing durable (`InMemoryRedis`) | none |

There is no Postgres, MinIO or Redis service to back up any more — one archive
covers the whole system.

## Backup (the whole /data directory)

```bash
# Stop the app first: a copy of a LIVE SQLite file can miss the WAL.
docker compose stop app
docker run --rm -v orivory-data:/data -v "$PWD/backups":/backup alpine \
  tar czf /backup/orivory-data_$(date +%Y%m%d_%H%M%S).tgz -C /data .
docker compose start app
```

A dev checkout bind-mounts `./data` instead of the named volume; tar that
directory the same way.

Prefer a SQLite-consistent snapshot WITHOUT stopping the app? Take the database
on its own with `VACUUM INTO` (it includes committed WAL frames), then copy
`uploads/` and `qdrant/` separately:

```bash
docker compose exec app python -c \
  "import sqlite3; sqlite3.connect('/data/orivory.db').execute(\"VACUUM INTO '/data/backup.db'\")"
```

## Restore

```bash
docker compose stop app
docker run --rm -v orivory-data:/data -v "$PWD/backups":/backup alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/orivory-data_<stamp>.tgz -C /data"
docker compose start app
```

Boot runs the versioned SQLite ladder (`bootstrap_sqlite()`), so an older
database is adopted and upgraded on start; a NEWER schema than the image knows
is refused loudly rather than repaired.

## Qdrant

Embedded Qdrant owns `/data/qdrant` exclusively and is included in the archive
above. If it is lost (or the archive is from a different database), the index
can be rebuilt: re-ingest the source documents, or run
`python scripts/migrate_qdrant.py backfill`. A restored Qdrant folder is only
consistent with the database it was taken with — the generation pointers live
in SQLite — so restore both or re-run `cutover`.

## Caches

`InMemoryRedis` is process-local: restarting the container drops caches,
rate-limit windows and the query cache. Nothing to back up — they are rebuilt
on the next request.

## Safe Restore Order

1. Stop the app.
2. Restore the `/data` archive (database + uploads + Qdrant together).
3. Start the app — the schema ladder upgrades an older database on boot.
4. Check `/ready`.
5. Run the offline eval smoke: `python eval/run_eval.py --mode offline
   --output-dir eval/results --top-k 5`.

## The migration CLI's own backups

The Qdrant cutover has its own offline tooling — see
[ROLLBACK_P1B.md](ROLLBACK_P1B.md):

- take the backup with `python scripts/migrate_qdrant.py backup --dir /backups/p1b`
  (VACUUM INTO snapshot + checksum manifest + copies of the uploads directory
  and the retired pre-P1b store when they exist);
- prove it restores with
  `python scripts/migrate_qdrant.py verify --restore-drill --dir /backups/p1b`
  — it restores into a NEW directory and refuses to report ready unless the
  checksums, `integrity_check`, `foreign_key_check`, the recorded embedding
  fingerprint and the deletion/suppression ledger all hold;
- going back to the pre-P1b stack is the **retired-Chroma rollback tool**, kept
  for exactly ONE release: `scripts/rollback_to_chroma.py` runs from its own
  isolated venv built with `requirements-rollback.txt` (the runtime lock no
  longer carries `chromadb`), never from the runtime image.

The migration also writes sidecar artifacts next to the database — all safe to
keep, none of them part of the DB backup: `<db>.pre-p1b.bak` (+
`.manifest.json`, the pre-cutover snapshot), `migrate.lock` (the single-migration
lock), `<db>.backfill-checkpoint.json` (resumable keyset checkpoint),
`<db>.p1b-expand-record.json` (the pointer the install served before the expand)
and `<db>.p1b-rollback-marker.json` (what `cutover` activated).

### Dropping the retired store

Once `cutover` has flipped the generation pointers, nothing serves from the old
Chroma store any more. Keep it until the one-release rollback window closes,
then delete the retired collection/store (`LEGACY_CHROMA_PATH`, holding
`Orivory_memories` and `rag_conv_*`) and drop any backups that only carry it —
the Qdrant data plus the database are the whole system from that point on.
`verify --restore-drill` is the supported restore afterwards.
