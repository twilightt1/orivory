# Final whole-branch P0 fix report

## Status

Implemented the final fix wave on branch `p0-audit-baseline`. The branch is locally committed, with the remaining P1/spec blockers explicitly fail-closed and documented below. No push was performed.

## Commit

- `0b377ea fix: close final P0 audit safety gaps`

## Changes

### Contract and vector safety

- Collection contract checks now distinguish genuinely empty new collections from populated unknown collections. Populated unknown, partial, unreadable, mismatched, and generation-mismatched collections fail with typed `EmbeddingDimensionMismatch`.
- Sync and async contract stamping now propagates metadata-write failures and verifies the resulting contract before an upsert can proceed.
- Chroma's immutable `hnsw:space` setting is excluded from metadata modification; the real local Chroma smoke path now stamps, writes two memories, reads them back, and searches successfully.
- Contract failures propagate through memory recall, write-back, import, and MCP write callers instead of becoming an ordinary empty result or an index-failure boolean.
- `only_missing` presence checks validate active fingerprint, dimension, backend, and the derived contract-generation token. A mismatch aborts safely and explicitly requires a fresh relevant collection rather than mixing contracts.
- Memory vector payloads now carry allowlisted backend, dimension, fingerprint, contract generation, model revision, and canonical-memory-revision provenance. A missing canonical revision is recorded as `unavailable`; `updated_at` is not misrepresented as a revision, and arbitrary memory metadata is not copied into the ACL/payload.

### Retrieval security and correctness

- SQL hydration and ownership/state eligibility now happen before semantic rerank. Rerank receives only current SQL-owned title/content; vector payload text is never sent to the remote reranker.
- Results are rehydrated and revalidated after rerank before scoring/serialization.
- Personal context excludes superseded/dirty rows before rewrite use.
- Zero-valued rerank scores remain zero instead of being replaced by vector scores.
- Memory vector filters use a principal-owned tenant clause plus allowlisted caller filters. Caller `user_id`, unknown fields, and unknown operators are rejected; caller filters cannot overwrite the tenant condition.

### Embedding contract provenance

- Fingerprints now include model/revision, artifact and tokenizer SHA-256 values where verified, graph output, pooling, prefixes, max-token/truncation/padding policy, normalization, dimension, precision, provider, and document-format version.
- Arctic XS and multilingual E5 downloads use pinned upstream revisions and verify expected artifact/tokenizer hashes. Production Arctic mean pooling remains unchanged.
- Retained Chroma MiniLM is represented explicitly as its own 384-dimensional contract with its bundled artifact/tokenizer provenance.
- Jina dimensions use `settings.JINA_EMBED_DIMENSIONS`; OpenAI dimensions use `settings.EMBED_DIMENSIONS` in the active fingerprint. Provider-defined API truncation/normalization is represented as unknown rather than asserted.

### SQLite and schema readiness

- Both canonical async and sync SQLite connection paths now set `foreign_keys=ON`, `journal_mode=WAL`, `busy_timeout=5000`, and `synchronous=FULL` through one shared connection policy.
- `bootstrap_sqlite` now uses `PRAGMA user_version`: fresh installs are initialized at schema version 1, while existing unversioned schemas and unsupported/incomplete versions fail closed instead of being silently evolved by `create_all`.
- The parity test uses an isolated temporary SQLite file, explicitly bootstraps schema, tests async-write/sync-read and sync-write/async-read, verifies FK/WAL/busy-timeout/FULL pragmas, checks FK rejection, cleans up rows, and never touches the deployment/test Postgres URL.

### Benchmark and dependency reproducibility

- Benchmark metadata now records resolved dataset path/source/digest, complete selected question IDs, seed, Git HEAD/dirty state, actual embedding backend/fingerprint/revision/artifact/provider, runtime/package versions, rerank configuration, answer/judge settings, timeout policy, concurrency requested versus actual, thread limits, warmup/cache state, and write/index timing totals.
- Per-question records now include ingest and recall durations.
- `uv lock` was regenerated from `pyproject.toml`; `uv lock --check` is clean.

## Verification

- Focused final safety/fingerprint/SQLite/benchmark suite: `38 passed, 1 warning`.
- Full suite: `560 passed, 55 skipped, 1 warning`.
- Ruff: `All checks passed!` for `app tests eval`.
- `uv lock --check`: exit 0; resolved 153 packages.
- `git diff --check`: exit 0 before commit.
- Real cached Arctic ONNX + local Chroma smoke: `REAL_LOCAL_CHROMA_SMOKE: PASS`, two sync upserts, async tenant-filtered search, and payload provenance readback.
- No paid LLM, embedding-provider, rerank-provider, or benchmark calls were made.

## Unresolved blockers and deliberate phase boundaries

1. **Existing SQLite schema migration is not implemented.** Version 1 is a fresh-install/bootstrap gate only. An existing unversioned database now fails startup with a concrete migration error. A reviewed SQLite migration, dry-run/backup/integrity workflow remains a P1 requirement.
2. **Full active-generation architecture is not implemented.** This wave provides a stable contract-generation token and makes `only_missing` fail closed, but it does not add the spec's SQLite generation manifest, canonical entity revision counter, outbox, fresh-collection active pointer, cutover, or durable backfill worker. A contract mismatch must be rebuilt in a separately provisioned fresh relevant collection before activation.
3. **Qdrant migration and complete Chroma removal are not part of this fix wave.** The branch remains on its existing Chroma transitional path; no new vector backend or unreviewed migration framework was introduced.
4. **Arctic XS production still uses the preserved `legacy-mean` contract.** The existing CLS probe remains a separate baseline-difference/parity gate; production CLS cutover requires the spec's independent parity/approval step. m-v2/INT8/256d remains out of scope.
5. **Canonical Memory revisions are unavailable in the current schema.** Payload provenance marks this explicitly rather than deriving a revision from `updated_at`; durable revision/outbox work remains part of the later phase.
6. **Postgres-backed integration coverage was not executable in this environment.** The suite reported the existing localhost:55432 Postgres-unavailable warning and skipped its dependent tests. No server parity or production migration claim is made.
7. **No benchmark quality or performance result was produced.** Benchmark provenance code and tests were exercised without provider calls; new measurements require the approved dataset/hardware/budget and paid-provider authorization.

## Report path

`/Users/twilight/Coding/WorkSpace/orivory/.worktrees/p0-audit-baseline/final-fix-report.md`
