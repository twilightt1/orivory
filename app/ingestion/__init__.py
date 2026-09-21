"""
Orivory ingestion package.

Document ingestion: `document_memory` projects uploaded documents into
memories, `pipeline` drives the chunk/embed/store path, and `import_formats`
parses the one-shot export uploads.

The connector layer + its dispatcher were removed on the slim branch; the
connector base/types (`base.py`, `types.py`) went with the D2 cleanup wave.
git history has both.
"""
