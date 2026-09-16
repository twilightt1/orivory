"""
Entity / Graph API — second-brain knowledge graph.

Read endpoints expose the graph for visualization and retrieval. Phase 4 adds
write endpoints so users/agents can correct extracted entities and relations.
"""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.graph.clustering import detect_clusters
from app.graph.extraction import (
    normalize_entity_name,
    normalize_entity_type,
    normalize_relation_type,
)
from app.models.entity import Entity, MemoryEntity, Relation
from app.models.memory import Memory
from app.models.user import User
from app.retrieval.memory.correction import state_of
from app.retrieval.memory.namespaces import personal_namespace
from app.retrieval.memory.visibility import namespace_predicate
from app.schemas.Orivory import (
    EntityCreate,
    EntityListResponse,
    EntityResponse,
    EntityUpdate,
    GraphClustersResponse,
    GraphEdge,
    GraphNode,
    GraphSnapshot,
    MemoryResponse,
    RelationCreate,
    RelationResponse,
    RelationUpdate,
)
from app.utils.dependencies import get_current_verified_user

router = APIRouter(prefix="/entities", tags=["entities"])


# ─── response helpers ────────────────────────────────────────────────────────


def _entity_response(entity: Entity) -> EntityResponse:
    """Map ORM Entity.extra_metadata to API field `metadata`."""
    return EntityResponse(
        id=entity.id,
        name=entity.name,
        entity_type=entity.entity_type,
        aliases=entity.aliases or [],
        description=entity.description,
        first_seen_at=entity.first_seen_at,
        last_seen_at=entity.last_seen_at,
        mention_count=entity.mention_count or 0,
        metadata=entity.extra_metadata or {},
        created_at=entity.created_at,
        updated_at=entity.updated_at,
    )


def _relation_response(relation: Relation) -> RelationResponse:
    """Map ORM Relation.extra_metadata to API field `metadata`."""
    return RelationResponse(
        id=relation.id,
        user_id=relation.user_id,
        source_entity_id=relation.source_entity_id,
        target_entity_id=relation.target_entity_id,
        relation=relation.relation,
        weight=relation.weight,
        evidence_count=relation.evidence_count,
        last_evidence_at=relation.last_evidence_at,
        metadata=relation.extra_metadata or {},
        created_at=relation.created_at,
        updated_at=relation.updated_at,
    )


def _memory_response(memory: Memory) -> MemoryResponse:
    """Map ORM Memory.extra_metadata to API field `metadata`."""
    return MemoryResponse(
        id=memory.id,
        user_id=memory.user_id,
        parent_id=memory.parent_id,
        source_type=memory.source_type,
        source_ref=memory.source_ref,
        source_url=memory.source_url,
        title=memory.title,
        content=memory.content,
        summary=memory.summary,
        tags=memory.tags or [],
        salience=memory.salience,
        pinned=memory.pinned,
        recall_count=memory.recall_count,
        last_used_at=memory.last_used_at,
        captured_at=memory.captured_at,
        indexed_at=memory.indexed_at,
        updated_at=memory.updated_at,
        revision=memory.revision or 1,  # unsaved/detached rows carry the column default
        state=state_of(memory),
        metadata=memory.extra_metadata or {},
    )


def _merge_aliases(existing: list[str] | None, incoming: list[str] | None) -> list[str]:
    merged: list[str] = []
    for alias in list(existing or []) + list(incoming or []):
        alias = alias.strip()
        if alias and alias not in merged:
            merged.append(alias)
    return merged[:50]


async def _find_entity_by_name(
    db: AsyncSession,
    user_id: UUID,
    name: str,
    entity_type: str,
) -> Entity | None:
    return (await db.execute(
        select(Entity).where(
            Entity.user_id == user_id,
            func.lower(Entity.name) == name.casefold(),
            Entity.entity_type == entity_type,
        )
    )).scalar_one_or_none()


async def _get_user_entity(db: AsyncSession, user_id: UUID, entity_id: UUID) -> Entity | None:
    entity = await db.get(Entity, entity_id)
    if not entity or entity.user_id != user_id:
        return None
    return entity


# ─── /entities ──────────────────────────────────────────────────────────────


@router.post("", response_model=EntityResponse, status_code=status.HTTP_201_CREATED)
async def create_entity(
    body: EntityCreate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EntityResponse:
    """Create or upsert a user-owned entity."""
    name = normalize_entity_name(body.name)
    entity_type = normalize_entity_type(body.entity_type)
    if not name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Entity name cannot be empty.")

    existing = await _find_entity_by_name(db, current_user.id, name, entity_type)
    if existing:
        existing.aliases = _merge_aliases(existing.aliases, body.aliases)
        if body.description and not existing.description:
            existing.description = body.description
        existing.extra_metadata = {**(existing.extra_metadata or {}), **body.metadata}
        await db.commit()
        await db.refresh(existing)
        return _entity_response(existing)

    entity = Entity(
        user_id=current_user.id,
        name=name,
        entity_type=entity_type,
        aliases=_merge_aliases([], body.aliases),
        description=body.description,
        mention_count=0,
        extra_metadata=body.metadata,
    )
    db.add(entity)
    await db.commit()
    await db.refresh(entity)
    return _entity_response(entity)


@router.get("", response_model=EntityListResponse)
async def list_entities(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    entity_type: str | None = None,
    name: str | None = Query(default=None, description="Substring match on name (case-insensitive)"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
) -> EntityListResponse:
    base = select(Entity).where(Entity.user_id == current_user.id)
    count_base = select(func.count(Entity.id)).where(Entity.user_id == current_user.id)

    if entity_type:
        normalized_type = normalize_entity_type(entity_type)
        base = base.where(Entity.entity_type == normalized_type)
        count_base = count_base.where(Entity.entity_type == normalized_type)
    if name:
        pattern = f"%{name.lower()}%"
        base = base.where(func.lower(Entity.name).like(pattern))
        count_base = count_base.where(func.lower(Entity.name).like(pattern))

    total = (await db.execute(count_base)).scalar_one()
    rows = (await db.execute(
        base.order_by(Entity.mention_count.desc(), Entity.last_seen_at.desc())
            .offset(offset).limit(limit)
    )).scalars().all()

    return EntityListResponse(
        items=[_entity_response(e) for e in rows],
        total=total,
        limit=limit,
        offset=offset,
    )


@router.get("/{entity_id}", response_model=EntityResponse)
async def get_entity(
    entity_id: UUID,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EntityResponse:
    entity = await _get_user_entity(db, current_user.id, entity_id)
    if not entity:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Entity not found.")
    return _entity_response(entity)


@router.patch("/{entity_id}", response_model=EntityResponse)
async def update_entity(
    entity_id: UUID,
    body: EntityUpdate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> EntityResponse:
    entity = await _get_user_entity(db, current_user.id, entity_id)
    if not entity:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Entity not found.")

    data = body.model_dump(exclude_unset=True)
    new_name = normalize_entity_name(data.get("name", entity.name))
    new_type = normalize_entity_type(data.get("entity_type", entity.entity_type))
    if not new_name:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Entity name cannot be empty.")

    duplicate = await _find_entity_by_name(db, current_user.id, new_name, new_type)
    if duplicate and duplicate.id != entity.id:
        raise HTTPException(status.HTTP_409_CONFLICT, detail="Entity with this name/type already exists.")

    if "name" in data:
        entity.name = new_name
    if "entity_type" in data:
        entity.entity_type = new_type
    if "aliases" in data:
        entity.aliases = _merge_aliases([], data["aliases"] or [])
    if "description" in data:
        entity.description = data["description"]
    if "metadata" in data:
        entity.extra_metadata = data["metadata"] or {}

    await db.commit()
    await db.refresh(entity)
    return _entity_response(entity)


@router.get("/{entity_id}/memories", response_model=list[MemoryResponse])
async def list_memories_for_entity(
    entity_id: UUID,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=50, ge=1, le=200),
) -> list[MemoryResponse]:
    """Return all memories that link to this entity, newest first."""
    entity = await _get_user_entity(db, current_user.id, entity_id)
    if not entity:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Entity not found.")

    rows = (await db.execute(
        select(Memory)
        .join(MemoryEntity, MemoryEntity.memory_id == Memory.id)
        .where(
            MemoryEntity.entity_id == entity_id,
            Memory.user_id == current_user.id,
            namespace_predicate(personal_namespace(current_user.id)),
        )
        .order_by(Memory.captured_at.desc())
        .limit(limit)
    )).scalars().all()
    return [_memory_response(m) for m in rows]


# ─── /relations ─────────────────────────────────────────────────────────────


relations_router = APIRouter(prefix="/relations", tags=["relations"])


@relations_router.post("", response_model=RelationResponse, status_code=status.HTTP_201_CREATED)
async def create_relation(
    body: RelationCreate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RelationResponse:
    """Create or upsert a relation between two user-owned entities."""
    source = await _get_user_entity(db, current_user.id, body.source_entity_id)
    target = await _get_user_entity(db, current_user.id, body.target_entity_id)
    if not source or not target:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Source or target entity not found.")
    if source.id == target.id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Self-relations are not allowed.")

    relation_type = normalize_relation_type(body.relation)
    existing = (await db.execute(
        select(Relation).where(
            Relation.user_id == current_user.id,
            Relation.source_entity_id == source.id,
            Relation.target_entity_id == target.id,
            Relation.relation == relation_type,
        )
    )).scalar_one_or_none()
    if existing:
        existing.weight = body.weight
        existing.extra_metadata = {**(existing.extra_metadata or {}), **body.metadata}
        existing.last_evidence_at = datetime.now(UTC)
        await db.commit()
        await db.refresh(existing)
        return _relation_response(existing)

    relation = Relation(
        user_id=current_user.id,
        source_entity_id=source.id,
        target_entity_id=target.id,
        relation=relation_type,
        weight=body.weight,
        evidence_count=1,
        last_evidence_at=datetime.now(UTC),
        extra_metadata=body.metadata,
    )
    db.add(relation)
    await db.commit()
    await db.refresh(relation)
    return _relation_response(relation)


@relations_router.patch("/{relation_id}", response_model=RelationResponse)
async def update_relation(
    relation_id: UUID,
    body: RelationUpdate,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
) -> RelationResponse:
    relation = await db.get(Relation, relation_id)
    if not relation or relation.user_id != current_user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Relation not found.")

    data = body.model_dump(exclude_unset=True)
    if "relation" in data:
        new_type = normalize_relation_type(data["relation"])
        duplicate = (await db.execute(
            select(Relation).where(
                Relation.user_id == current_user.id,
                Relation.source_entity_id == relation.source_entity_id,
                Relation.target_entity_id == relation.target_entity_id,
                Relation.relation == new_type,
                Relation.id != relation.id,
            )
        )).scalar_one_or_none()
        if duplicate:
            raise HTTPException(status.HTTP_409_CONFLICT, detail="Duplicate relation already exists.")
        relation.relation = new_type
    if "weight" in data:
        relation.weight = data["weight"]
    if "metadata" in data:
        relation.extra_metadata = data["metadata"] or {}

    await db.commit()
    await db.refresh(relation)
    return _relation_response(relation)


# ─── /graph ─────────────────────────────────────────────────────────────────


graph_router = APIRouter(prefix="/graph", tags=["graph"])


@graph_router.get("/snapshot", response_model=GraphSnapshot)
async def graph_snapshot(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    min_weight: float = Query(default=0.0, ge=0.0, le=1.0),
    limit: int = Query(default=200, ge=1, le=1000),
) -> GraphSnapshot:
    """Return a snapshot of the knowledge graph (nodes + edges) for visualization."""
    entity_rows = (await db.execute(
        select(Entity)
        .where(Entity.user_id == current_user.id)
        .order_by(Entity.mention_count.desc())
        .limit(limit)
    )).scalars().all()
    entity_ids = {e.id for e in entity_rows}
    if not entity_ids:
        # No entities yet → no relations can match. Return early with an
        # empty snapshot to avoid issuing a Relation query with an empty
        # IN(...) clause (dialect-dependent behavior).
        return GraphSnapshot(
            nodes=[],
            edges=[],
            generated_at=datetime.now(UTC),
        )

    relation_rows = (await db.execute(
        select(Relation)
        .where(
            Relation.user_id == current_user.id,
            Relation.weight >= min_weight,
            Relation.source_entity_id.in_(entity_ids),
            Relation.target_entity_id.in_(entity_ids),
        )
        .order_by(Relation.weight.desc())
        .limit(limit * 2)
    )).scalars().all()

    # Build id -> name map for edge endpoints
    name_of = {e.id: e.name for e in entity_rows}

    return GraphSnapshot(
        nodes=[
            GraphNode(id=e.name, type=e.entity_type, mentions=e.mention_count)
            for e in entity_rows
        ],
        edges=[
            GraphEdge(
                source=name_of.get(r.source_entity_id, str(r.source_entity_id)),
                target=name_of.get(r.target_entity_id, str(r.target_entity_id)),
                relation=r.relation,
                weight=r.weight,
            )
            for r in relation_rows
        ],
        generated_at=datetime.now(UTC),
    )


@graph_router.get("/clusters", response_model=GraphClustersResponse)
async def graph_clusters(
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    min_weight: float = Query(default=0.2, ge=0.0, le=1.0),
    limit: int = Query(default=20, ge=1, le=100),
) -> GraphClustersResponse:
    """Return simple connected-component clusters for graph visualization."""
    return await detect_clusters(db, current_user.id, min_weight=min_weight, limit=limit)


@graph_router.get("/related/{entity_name}", response_model=list[EntityResponse])
async def related_entities(
    entity_name: str,
    current_user: Annotated[User, Depends(get_current_verified_user)],
    db: Annotated[AsyncSession, Depends(get_db)],
    limit: int = Query(default=20, ge=1, le=100),
) -> list[EntityResponse]:
    """Return entities directly related to the given entity name (1-hop neighbors)."""
    # Find source entity
    src = (await db.execute(
        select(Entity)
        .where(Entity.user_id == current_user.id, func.lower(Entity.name) == entity_name.lower())
    )).scalar_one_or_none()
    if not src:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail=f"Entity '{entity_name}' not found.")

    # Find both outgoing and incoming
    related_ids = (await db.execute(
        select(distinct(Relation.target_entity_id))
        .where(Relation.user_id == current_user.id, Relation.source_entity_id == src.id)
    )).scalars().all()
    incoming_ids = (await db.execute(
        select(distinct(Relation.source_entity_id))
        .where(Relation.user_id == current_user.id, Relation.target_entity_id == src.id)
    )).scalars().all()
    all_ids = set(related_ids) | set(incoming_ids)
    if not all_ids:
        return []

    rows = (await db.execute(
        select(Entity)
        .where(Entity.user_id == current_user.id, Entity.id.in_(all_ids))
        .order_by(Entity.mention_count.desc())
        .limit(limit)
    )).scalars().all()
    return [_entity_response(e) for e in rows]
