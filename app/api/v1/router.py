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
# one-shot imports.
api_router.include_router(memories.router)
api_router.include_router(agents.router)
api_router.include_router(erasure.router)
api_router.include_router(imports.router)  # /imports - one-shot export upload

# The full-stack surface (chat, admin, analytics, demo, discovery, entities,
# experiments, hints, insights, referral, sources, system_settings,
# workspaces, sse) was removed on the slim branch; git history has it.
