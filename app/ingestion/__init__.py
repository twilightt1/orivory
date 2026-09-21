"""
Orivory ingestion package.

Document ingestion: `document_memory` projects uploaded documents into
memories, `pipeline` drives the chunk/embed/store path, `import_formats`
parses the one-shot export uploads, and `backoff`/`types` are shared
primitives. The connector layer + its dispatcher were removed on the slim
branch; git history has them.
"""
from app.ingestion.base import BaseConnector
from app.ingestion.types import ConnectorItem, SyncResult

__all__ = [
    "ConnectorItem",
    "SyncResult",
    "BaseConnector",
]
