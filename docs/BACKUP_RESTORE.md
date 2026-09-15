# Backup and Restore Guide

Orivory stores durable state in Postgres, MinIO, and ChromaDB volumes.
Redis is used for queues/cache/session-like data and is usually not the primary source of truth.

## What to Back Up

| Component | Data | Backup priority |
|---|---|---|
| Postgres | users, conversations, messages, document metadata, chunks | Critical |
| MinIO | uploaded source documents | Critical |
| ChromaDB | vector index | Important, rebuildable from docs/chunks |
| Redis | Celery queues, cache, refresh tokens, rate limits | Optional / operational |

## Postgres Backup

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres \
  pg_dump -U "$POSTGRES_USER" "$POSTGRES_DB" > backups/postgres_$(date +%Y%m%d_%H%M%S).sql
```

For Windows PowerShell:

```powershell
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec postgres pg_dump -U $env:POSTGRES_USER $env:POSTGRES_DB > backups/postgres_backup.sql
```

## Postgres Restore

Stop app services first:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop app celery_worker celery_beat
```

Restore:

```bash
cat backups/postgres_backup.sql | docker compose -f docker-compose.yml -f docker-compose.prod.yml exec -T postgres \
  psql -U "$POSTGRES_USER" "$POSTGRES_DB"
```

Run migrations after restore if needed:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml exec app alembic upgrade head
```

## MinIO Backup

Recommended options:

1. use MinIO Client (`mc`) mirror to object storage or local disk
2. snapshot the Docker volume `miniodata`

Example with `mc` from a configured host:

```bash
mc alias set Orivory http://localhost:9000 "$MINIO_ACCESS_KEY" "$MINIO_SECRET_KEY"
mc mirror Orivory/rag-docs backups/minio/rag-docs
```

## MinIO Restore

```bash
mc mirror backups/minio/rag-docs Orivory/rag-docs
```

Ensure `MINIO_BUCKET` matches the restored bucket name.

## ChromaDB Backup

ChromaDB stores persistent index data in the `chromadata` Docker volume.

Snapshot the volume while writes are stopped:

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop app celery_worker
docker run --rm -v orivory_chromadata:/data -v "$PWD/backups":/backup alpine \
  tar czf /backup/chromadata_backup.tgz -C /data .
```

## ChromaDB Restore

```bash
docker compose -f docker-compose.yml -f docker-compose.prod.yml stop chromadb
docker run --rm -v orivory_chromadata:/data -v "$PWD/backups":/backup alpine \
  sh -c "rm -rf /data/* && tar xzf /backup/chromadata_backup.tgz -C /data"
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d chromadb
```

If ChromaDB backup is missing, the index can be rebuilt by re-ingesting source documents, but that takes longer and needs provider keys.

## Redis Notes

Redis contains Celery queue state, cache, refresh tokens, and rate limit keys.
In most deployments, do not rely on Redis as the backup source of truth.

If Redis is lost:

- active Celery jobs may need to be retried
- users may need to log in again
- BM25/parent caches can rebuild through ingestion paths

## Safe Restore Order

1. Stop API and Celery workers.
2. Restore Postgres.
3. Restore MinIO.
4. Restore ChromaDB if available.
5. Start infrastructure.
6. Run migrations.
7. Start API and Celery.
8. Check `/ready`.
9. Run offline or live API eval smoke.

## P1b cutover backups (SQLite / lite)

The Qdrant cutover has its own offline tooling — see
[ROLLBACK_P1B.md](ROLLBACK_P1B.md):

- take the backup with `python scripts/migrate_qdrant.py backup --dir /backups/p1b`
  (VACUUM INTO snapshot + checksum manifest + uploads/chroma copies);
- prove it restores with
  `python scripts/migrate_qdrant.py verify --restore-drill --dir /backups/p1b`
  — it restores into a NEW directory and refuses to report ready unless the
  checksums, `integrity_check`, `foreign_key_check`, the recorded embedding
  fingerprint and the deletion/suppression ledger all hold;
- going back to the pre-P1b stack for one release is
  `scripts/rollback_to_chroma.py` (isolated venv, `requirements-rollback.txt`).

