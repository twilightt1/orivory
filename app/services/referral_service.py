"""
Referral Service

Business logic for referral system.
"""

from __future__ import annotations

import logging
import secrets
import string
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import UUID

from sqlalchemy import and_, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.referral import Referral, ReferralCode, ReferralReward

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)


def generate_referral_code(length: int = 8) -> str:
    """Generate a unique referral code. Format: ML-XXXXXXXX"""
    chars = string.ascii_uppercase + string.digits
    code = ''.join(secrets.choice(chars) for _ in range(length))
    return f"ML-{code}"


async def get_or_create_referral_code(db: AsyncSession, user_id: UUID) -> ReferralCode:
    """Get existing referral code or create a new one for user.

    Race-safe: the DB enforces one active code per user
    (uq_referral_codes_user_active). A concurrent first call wins; the loser
    catches IntegrityError, rolls back, and re-reads the winner's row.
    """
    from sqlalchemy.exc import IntegrityError

    result = await db.execute(
        select(ReferralCode).where(
            and_(
                ReferralCode.user_id == user_id,
                ReferralCode.is_active
            )
        )
    )
    existing = result.scalars().first()

    if existing:
        return existing

    # Create new code
    while True:
        code = generate_referral_code()
        result = await db.execute(
            select(ReferralCode).where(ReferralCode.code == code)
        )
        if not result.scalar_one_or_none():
            break

    referral_code = ReferralCode(
        user_id=user_id,
        code=code,
        is_active=True,
        max_uses=10
    )
    db.add(referral_code)
    try:
        await db.commit()
    except IntegrityError:
        # Lost the race: another request created the user's code first.
        await db.rollback()
        winner = await db.scalar(
            select(ReferralCode).where(
                and_(
                    ReferralCode.user_id == user_id,
                    ReferralCode.is_active
                )
            ).order_by(ReferralCode.created_at.asc()).limit(1)
        )
        if winner is not None:
            log.info(f"Referral code race lost for user {user_id} — returning winner")
            return winner
        raise
    await db.refresh(referral_code)

    log.info(f"Created referral code {code} for user {user_id}")
    return referral_code


async def create_referral_link(
    db: AsyncSession,
    referrer_id: UUID,
    referee_email: str
) -> Referral | None:
    """Create a pending referral link when user shares."""
    referral_code = await get_or_create_referral_code(db, referrer_id)

    # Check if referral already exists
    result = await db.execute(
        select(Referral).where(
            and_(
                Referral.referrer_id == referrer_id,
                Referral.referee_email == referee_email.lower()
            )
        )
    )
    existing = result.scalar_one_or_none()

    if existing:
        return existing

    referral = Referral(
        referral_code_id=referral_code.id,
        referrer_id=referrer_id,
        referee_email=referee_email.lower(),
        status="pending"
    )
    db.add(referral)
    await db.commit()
    await db.refresh(referral)

    log.info(f"Created referral: {referrer_id} -> {referee_email}")
    return referral


async def complete_referral(
    db: AsyncSession,
    referee_id: UUID,
    referee_email: str
) -> Referral | None:
    """Complete a pending referral when referee signs up."""
    result = await db.execute(
        select(Referral).where(
            and_(
                Referral.referee_email == referee_email.lower(),
                Referral.status == "pending"
            )
        )
    )
    referral = result.scalar_one_or_none()

    if not referral:
        log.info(f"No pending referral found for {referee_email}")
        return None

    # Update referral
    referral.referee_id = referee_id
    referral.status = "completed"
    referral.completed_at = datetime.now(UTC)

    await db.commit()
    await db.refresh(referral)

    log.info(f"Completed referral: {referral.referrer_id} -> {referee_id}")

    # Create reward
    await create_referral_reward(db, referral)

    return referral


async def create_referral_reward(db: AsyncSession, referral: Referral) -> ReferralReward:
    """Create a reward for the referrer."""
    # Count existing rewards
    result = await db.execute(
        select(func.count(ReferralReward.id)).where(
            ReferralReward.user_id == referral.referrer_id
        )
    )
    reward_count = result.scalar() or 0

    # Tiered rewards
    if reward_count >= 4:
        reward_type, reward_value = "free_months", 3  # 3 months for 5+ referrals
    else:
        reward_type, reward_value = "free_months", 1  # 1 month for first 4

    reward = ReferralReward(
        referral_id=referral.id,
        user_id=referral.referrer_id,
        reward_type=reward_type,
        reward_value=reward_value
    )
    db.add(reward)
    await db.commit()
    await db.refresh(reward)

    log.info(f"Created reward: {reward_value} {reward_type} for user {referral.referrer_id}")
    return reward


async def get_referral_stats(db: AsyncSession, user_id: UUID) -> dict:
    """Get referral statistics for a user."""
    # Total completed referrals
    result = await db.execute(
        select(func.count(Referral.id)).where(
            and_(
                Referral.referrer_id == user_id,
                Referral.status == "completed"
            )
        )
    )
    total_referrals = result.scalar() or 0

    # Pending referrals
    result = await db.execute(
        select(func.count(Referral.id)).where(
            and_(
                Referral.referrer_id == user_id,
                Referral.status == "pending"
            )
        )
    )
    pending_referrals = result.scalar() or 0

    # Unclaimed rewards
    result = await db.execute(
        select(func.count(ReferralReward.id)).where(
            and_(
                ReferralReward.user_id == user_id,
                ReferralReward.is_claimed.is_(False),
            )
        )
    )
    unclaimed_rewards = result.scalar() or 0

    # Get referral code
    referral_code = await get_or_create_referral_code(db, user_id)

    return {
        "total_referrals": total_referrals,
        "pending_referrals": pending_referrals,
        "unclaimed_rewards": unclaimed_rewards,
        "referral_code": referral_code.code,
        "referral_link": f"https://Orivory.app/signup?ref={referral_code.code}"
    }


async def get_referral_leaderboard(db: AsyncSession, limit: int = 10) -> list[dict]:
    """Get top referrers for leaderboard."""
    result = await db.execute(
        select(
            Referral.referrer_id,
            func.count(Referral.id).label("count")
        )
        .where(Referral.status == "completed")
        .group_by(Referral.referrer_id)
        .order_by(func.count(Referral.id).desc())
        .limit(limit)
    )

    leaderboard = []
    for row in result.all():
        leaderboard.append({
            "user_id": str(row.referrer_id),
            "referral_count": row.count
        })

    return leaderboard
