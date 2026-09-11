from fastapi import APIRouter

from app.api.v1 import (
    agents,
    auth,
    erasure,
    imports,
    memories,
    users,
)

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(auth.router)
api_router.include_router(users.router)

# Slim core (memory-for-agents): memories, agent tokens + ledger, erasure,
# one-shot imports. Everything else is dormant on the slim branch — files
# stay in tree, unmounted. See docs/ideas/open-source-positioning.md.
api_router.include_router(memories.router)
api_router.include_router(agents.router)
api_router.include_router(erasure.router)
api_router.include_router(imports.router)  # /imports - one-shot export upload

# Dormant (full-stack only, re-mount when needed):
#   chat, admin, system_settings, entities (+relations/graph), sources,
#   insights, discovery, workspaces, demo, analytics, referral
