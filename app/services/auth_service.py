"""Authentication business logic."""
from __future__ import annotations

import asyncio
import hashlib
import logging
import secrets
import string
from datetime import UTC, datetime, timedelta
from uuid import UUID

import bcrypt
from sqlalchemy import and_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models.email_verification import EmailVerification
from app.models.password_reset_session import PasswordResetSession
from app.models.user import User
from app.redis_client import get_redis
from app.utils.security import create_access_token

log = logging.getLogger(__name__)
OTP_MAX = 5


def _now() -> datetime:
    return datetime.now(UTC)


def _hash_refresh_token(token: str) -> str:
    """Hash a refresh token before it is used as a Redis key.

    Storing only the SHA-256 hex of the token means the raw token string
    is never used as a key (defense in depth) while still allowing O(1)
    lookup with the same value derived from the client-supplied token.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _hash(pw: str) -> str:
    pw_bytes = pw.encode('utf-8')
    if len(pw_bytes) > 72:
        pw_bytes = hashlib.sha256(pw_bytes).hexdigest().encode('utf-8')
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt()).decode('utf-8')

def _verify(pw: str, h: str) -> bool:
    pw_bytes = pw.encode('utf-8')
    if len(pw_bytes) > 72:
        pw_bytes = hashlib.sha256(pw_bytes).hexdigest().encode('utf-8')
    try:
        return bcrypt.checkpw(pw_bytes, h.encode('utf-8'))
    except Exception:
        return False


async def _hash_async(pw: str) -> str:
    """bcrypt costs ~100-300ms of CPU — must not block the event loop."""
    return await asyncio.to_thread(_hash, pw)


async def _verify_async(pw: str, h: str) -> bool:
    return await asyncio.to_thread(_verify, pw, h)
def _otp() -> str:
    return "".join(secrets.choice(string.digits) for _ in range(6))



async def register_email(db: AsyncSession, email: str, password: str) -> User:
    from fastapi import HTTPException
    from sqlalchemy.exc import IntegrityError
    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        detail = ("Email already registered via Google. Please sign in with Google."
                  if existing.auth_provider == "google"
                  else "Email already in use.")
        raise HTTPException(409, detail=detail)

    user = User(email=email, hashed_password=await _hash_async(password), auth_provider="email",
                # ponytail: single-user self-host with no email provider has no
                # way to deliver OTPs — verification mail would be a mock into
                # the void. Trust local registration when SendGrid is unset.
                is_verified=(not settings.SENDGRID_API_KEY),
                onboarding_done=False)
    db.add(user)
    try:
        await db.flush()
    except IntegrityError:
        # Two concurrent registrations for the same email both passed the
        # pre-check above; the unique constraint is the real arbiter.
        await db.rollback()
        raise HTTPException(409, detail="Email already in use.") from None

    otp, token = _otp(), secrets.token_urlsafe(64)
    db.add(EmailVerification(
        user_id=user.id, token=token, token_type="verify",
        otp_code=otp, otp_attempts=0,
        expires_at=_now() + timedelta(hours=24),
    ))

    from app.models.user_quota import UserQuota
    db.add(UserQuota(user_id=user.id))
    await db.commit()
    await db.refresh(user)

    # Best-effort send: the OTP row is already committed above so verification
    # can be retried via resend.
    from app.services.email_service import email_service
    try:
        email_service.send_verification(email, otp, token)
    except Exception as exc:
        log.warning("Verification email enqueue failed for %s: %s", email, exc)
    return user



async def verify_email_otp(db: AsyncSession, email: str, otp_code: str) -> User:
    from fastapi import HTTPException
    user = await db.scalar(select(User).where(User.email == email))
    if not user or user.is_verified:
        raise HTTPException(400, detail="Account not found or already verified.")

    ev = await db.scalar(
        select(EmailVerification).where(and_(
            EmailVerification.user_id == user.id,
            EmailVerification.token_type == "verify",
            EmailVerification.used_at.is_(None),
            EmailVerification.expires_at > _now(),
        ))
    )
    if not ev:
        raise HTTPException(400, detail="OTP expired. Please request a new one.")
    if ev.otp_attempts >= OTP_MAX:
        raise HTTPException(400, detail="Too many attempts. Please request a new code.")
    if ev.otp_code != otp_code:
        ev.otp_attempts += 1
        await db.commit()
        raise HTTPException(400, detail=f"Incorrect OTP. {OTP_MAX - ev.otp_attempts} attempts left.")

    user.is_verified = True
    ev.used_at = _now()
    await db.commit()
    await db.refresh(user)
    return user



async def verify_email_link(db: AsyncSession, token: str) -> User:
    from fastapi import HTTPException
    ev = await db.scalar(
        select(EmailVerification).where(and_(
            EmailVerification.token == token,
            EmailVerification.token_type == "verify",
            EmailVerification.used_at.is_(None),
            EmailVerification.expires_at > _now(),
        ))
    )
    if not ev:
        raise HTTPException(400, detail="Invalid or expired verification link.")
    user = await db.get(User, ev.user_id)
    if not user:
        raise HTTPException(400, detail="Account not found.")
    user.is_verified = True
    ev.used_at = _now()
    await db.commit()
    await db.refresh(user)
    return user



async def _count_with_window(redis, key: str, window_seconds: int) -> int:
    """Increment a rate counter with a TTL set atomically.

    ``INCR`` followed by ``EXPIRE`` is not atomic: a crash between the two
    leaves a key with no TTL that blocks the email forever. Ensure the
    window exists via ``SET NX`` first, then INCR — the first request
    yields 1, the second 2, and the key always carries a TTL.
    """
    await redis.set(key, 0, ex=window_seconds, nx=True)
    return await redis.incr(key)


async def resend_verification(db: AsyncSession, email: str) -> None:
    from fastapi import HTTPException
    redis = await get_redis()
    key = f"resend_limit:{email}"
    count = await _count_with_window(redis, key, 3600)
    if count > 3:
        raise HTTPException(429, detail="Too many resend requests. Try again in 1 hour.")

    user = await db.scalar(select(User).where(User.email == email))
    if not user or user.is_verified:
        return

    await db.execute(
        update(EmailVerification)
        .where(and_(EmailVerification.user_id == user.id,
                    EmailVerification.token_type == "verify",
                    EmailVerification.used_at.is_(None)))
        .values(used_at=_now())
    )
    otp, token = _otp(), secrets.token_urlsafe(64)
    db.add(EmailVerification(
        user_id=user.id, token=token, token_type="verify",
        otp_code=otp, otp_attempts=0,
        expires_at=_now() + timedelta(hours=24),
    ))
    await db.commit()
    # ponytail: same broker-absent guard as register_email — OTP row is
    # committed, so a dropped enqueue is retryable via resend.
    from app.services.email_service import email_service
    try:
        email_service.send_verification(email, otp, token)
    except Exception as exc:
        log.warning("Verification email enqueue failed for %s: %s", email, exc)



async def complete_onboarding(db: AsyncSession, user: User, display_name: str) -> tuple[User, str, str]:
    from fastapi import HTTPException
    if not user.is_verified:
        raise HTTPException(403, detail="Email not verified.")
    user.display_name    = display_name
    user.onboarding_done = True
    await db.commit()
    await db.refresh(user)
    access  = create_access_token({"sub": str(user.id), "role": user.role})
    refresh = await _create_refresh(user.id)
    return user, access, refresh



async def login_email(db: AsyncSession, email: str, password: str) -> tuple[User, str, str]:
    from fastapi import HTTPException
    user = await db.scalar(select(User).where(User.email == email))
    if (not user or user.auth_provider != "email"
            or not user.hashed_password
            or not await _verify_async(password, user.hashed_password)):
        raise HTTPException(401, detail="Invalid email or password.")
    if not user.is_verified:
        raise HTTPException(403, detail="Please verify your email first.")
    if not user.is_active or user.is_deleted:
        raise HTTPException(403, detail="Account deactivated.")
    access  = create_access_token({"sub": str(user.id), "role": user.role})
    refresh = await _create_refresh(user.id)
    return user, access, refresh



async def find_or_create_google_user(db: AsyncSession, info: dict) -> User:
    from fastapi import HTTPException
    sub, picture = info["sub"], info.get("picture")
    # Normalize to match the lowercased emails stored via the request schemas,
    # so a Google login resolves to the same row as an email signup.
    email = str(info["email"]).strip().lower()


    user = await db.scalar(select(User).where(User.google_id == sub))
    if user:
        if user.is_deleted or not user.is_active:
            raise HTTPException(403, detail="Account deactivated.")
        if picture and user.avatar_url != picture:
            user.avatar_url = picture
            await db.commit()
        return user


    existing = await db.scalar(select(User).where(User.email == email))
    if existing:
        if existing.is_deleted or not existing.is_active:
            raise HTTPException(403, detail="Account deactivated.")
        if existing.auth_provider == "email":
            raise HTTPException(409, detail="This email is registered with a password. Please log in with email.")
        existing.google_id  = sub
        existing.avatar_url = picture
        await db.commit()
        return existing


    from app.models.user_quota import UserQuota
    user = User(
        email=email, auth_provider="google", google_id=sub,
        avatar_url=picture, is_verified=True, is_active=True,
        onboarding_done=False, display_name=None,
    )
    db.add(user)
    await db.flush()
    db.add(UserQuota(user_id=user.id))
    await db.commit()
    await db.refresh(user)
    log.info("Google user created", extra={"user_id": str(user.id)})
    return user



async def create_password_reset_session(db: AsyncSession, email: str) -> None:
    from fastapi import HTTPException
    redis = await get_redis()
    key = f"forgot_pw:{email}"
    count = await _count_with_window(redis, key, 3600)
    if count > 3:
        raise HTTPException(429, detail="Too many requests. Try again in 1 hour.")

    user = await db.scalar(
        select(User).where(and_(User.email == email, User.auth_provider == "email"))
    )
    if not user or not user.is_active or user.is_deleted:
        return

    await db.execute(
        update(PasswordResetSession)
        .where(and_(PasswordResetSession.user_id == user.id,
                    PasswordResetSession.used_at.is_(None)))
        .values(used_at=_now())
    )
    otp, token = _otp(), secrets.token_urlsafe(64)
    db.add(PasswordResetSession(
        user_id=user.id, token=token, otp_code=otp,
        verified=False, expires_at=_now() + timedelta(minutes=15),
    ))
    await db.commit()
    # ponytail: same broker-absent guard — reset row is committed, caller
    # already rate-limited, so a dropped enqueue degrades to "try again".
    from app.services.email_service import email_service
    try:
        email_service.send_password_reset(email, otp, token)
    except Exception as exc:
        log.warning("Password-reset email enqueue failed for %s: %s", email, exc)


async def verify_reset_otp(db: AsyncSession, email: str, otp_code: str) -> str:
    from fastapi import HTTPException
    user = await db.scalar(select(User).where(User.email == email))
    if not user:
        raise HTTPException(400, detail="Invalid OTP or expired session.")

    session = await db.scalar(
        select(PasswordResetSession)
        .where(and_(
            PasswordResetSession.user_id == user.id,
            PasswordResetSession.verified.is_(False),
            PasswordResetSession.used_at.is_(None),
            PasswordResetSession.expires_at > _now(),
        ))
        .order_by(PasswordResetSession.created_at.desc())
    )
    if not session:
        raise HTTPException(400, detail="Reset session expired. Please start over.")
    if session.otp_attempts >= OTP_MAX:
        raise HTTPException(400, detail="Too many attempts. Request a new code.")
    if session.otp_code != otp_code:
        session.otp_attempts += 1
        await db.commit()
        raise HTTPException(400, detail=f"Incorrect OTP. {OTP_MAX - session.otp_attempts} attempts left.")

    session.verified = True
    await db.commit()
    return session.token


async def verify_reset_link(db: AsyncSession, token: str) -> str:
    from fastapi import HTTPException
    session = await db.scalar(
        select(PasswordResetSession).where(and_(
            PasswordResetSession.token == token,
            PasswordResetSession.verified.is_(False),
            PasswordResetSession.used_at.is_(None),
            PasswordResetSession.expires_at > _now(),
        ))
    )
    if not session:
        raise HTTPException(400, detail="Invalid or expired reset link.")
    session.verified = True
    await db.commit()
    return token


async def reset_password(db: AsyncSession, token: str, new_password: str) -> None:
    from fastapi import HTTPException
    session = await db.scalar(
        select(PasswordResetSession).where(and_(
            PasswordResetSession.token == token,
            PasswordResetSession.verified.is_(True),
            PasswordResetSession.used_at.is_(None),
            PasswordResetSession.expires_at > _now(),
        ))
    )
    if not session:
        raise HTTPException(400, detail="Invalid or expired reset session.")
    user = await db.get(User, session.user_id)
    if not user:
        raise HTTPException(400, detail="Account not found.")
    user.hashed_password = await _hash_async(new_password)
    session.used_at = _now()
    await db.commit()
    await _invalidate_all_refresh(user.id)
    log.info("Password reset", extra={"user_id": str(user.id)})



async def update_display_name(db: AsyncSession, user: User, display_name: str) -> User:
    user.display_name = display_name
    await db.commit()
    await db.refresh(user)
    return user


async def set_retention_settings(db: AsyncSession, user: User, *,
                                 enabled: bool, days: int | None) -> User:
    """Write the user's opt-in retention setting (P4b/T6, spec §8.1).

    The window is stored even while retention is OFF (a user may set it up
    before turning it on); ``None`` means "no window chosen". The boundary
    (``RetentionSettingsRequest``) refuses "enabled without a window", so the
    pair stored here is always runnable or switched off.
    """
    user.retention_enabled = enabled
    user.retention_days = days
    await db.commit()
    await db.refresh(user)
    return user


async def change_password(db: AsyncSession, user: User, current: str, new_pw: str) -> None:
    from fastapi import HTTPException
    if user.auth_provider != "email":
        raise HTTPException(400, detail="Google accounts do not use passwords.")
    if not user.hashed_password or not await _verify_async(current, user.hashed_password):
        raise HTTPException(400, detail="Current password is incorrect.")
    user.hashed_password = await _hash_async(new_pw)
    await db.commit()
    await _invalidate_all_refresh(user.id)
    log.info("Password changed", extra={"user_id": str(user.id)})



async def _create_refresh(user_id: UUID | str) -> str:
    """Create a refresh token and persist it hashed in Redis.

    The raw token is returned to the client once. Redis only ever sees the
    SHA-256 hash of the token plus a per-user set index used for O(1)
    invalidation (e.g. on password change).
    """
    redis = await get_redis()
    token = secrets.token_urlsafe(64)
    ttl   = settings.REFRESH_TOKEN_EXPIRE_DAYS * 86400
    token_hash = _hash_refresh_token(token)
    pipe = redis.pipeline()
    pipe.setex(f"refresh:{token_hash}", ttl, str(user_id))
    pipe.sadd(f"refresh_user:{user_id}", token_hash)
    pipe.expire(f"refresh_user:{user_id}", ttl)
    await pipe.execute()
    return token


async def _invalidate_all_refresh(user_id: UUID | str) -> None:
    """Invalidate every refresh token issued to ``user_id``.

    Looks up the per-user index set, deletes each hashed token key, and
    finally removes the index itself. O(N_user_tokens) instead of an
    O(total_tokens) keyspace scan.
    """
    redis = await get_redis()
    user_key = f"refresh_user:{user_id}"
    token_hashes = await redis.smembers(user_key)
    if token_hashes:
        keys = [f"refresh:{th}" for th in token_hashes]
        await redis.delete(*keys)
    await redis.delete(user_key)


async def _invalidate_one_refresh(refresh_token: str) -> None:
    """Invalidate a single refresh token (logout, rotation).

    Hashes the supplied token and removes the corresponding key. The
    index set is not modified here because the token is expected to be
    removed in the same atomic pair as creation during rotation.
    """
    redis = await get_redis()
    token_hash = _hash_refresh_token(refresh_token)
    await redis.delete(f"refresh:{token_hash}")
