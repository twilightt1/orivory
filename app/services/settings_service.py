import json
from typing import TYPE_CHECKING

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.system_setting import SystemSetting
from app.redis_client import get_redis

if TYPE_CHECKING:  # annotations only
    from redis.asyncio import Redis


class SettingsService:
    CACHE_PREFIX = "sys_setting:"
    CACHE_TTL = 300


    DEFAULTS = {
        "default_user_daily_limit": {"value": 50, "description": "Default daily request limit for new users"},
        "default_user_monthly_limit": {"value": 1000, "description": "Default monthly request limit for new users"},
        "maintenance_mode": {"value": False, "description": "If true, non-admins cannot access the system"},
    }

    @classmethod
    async def get_setting(cls, db: AsyncSession, key: str) -> dict:
        """
        Get a system setting, checking Redis cache first, then DB, then falling back to hardcoded defaults.
        """
        redis: Redis = await get_redis()
        cache_key = f"{cls.CACHE_PREFIX}{key}"


        cached_value = await redis.get(cache_key)
        if cached_value:
            return json.loads(cached_value)


        db_setting = await db.scalar(select(SystemSetting).where(SystemSetting.key == key))
        if db_setting:
            val = db_setting.value

            await redis.setex(cache_key, cls.CACHE_TTL, json.dumps(val))
            return val


        if key in cls.DEFAULTS:
            default_val = cls.DEFAULTS[key]["value"]

            await redis.setex(cache_key, cls.CACHE_TTL, json.dumps(default_val))
            return default_val

        return None

    @classmethod
    async def set_setting(cls, db: AsyncSession, key: str, value: dict, description: str | None = None) -> SystemSetting:
        """
        Create or update a system setting in the database and invalidate the cache.
        """
        db_setting = await db.scalar(select(SystemSetting).where(SystemSetting.key == key))

        if db_setting:
            db_setting.value = value
            if description is not None:
                db_setting.description = description
        else:
            if description is None and key in cls.DEFAULTS:
                description = cls.DEFAULTS[key]["description"]

            db_setting = SystemSetting(
                key=key,
                value=value,
                description=description
            )
            db.add(db_setting)




        redis: Redis = await get_redis()
        cache_key = f"{cls.CACHE_PREFIX}{key}"
        await redis.delete(cache_key)

        return db_setting

    @classmethod
    async def delete_setting(cls, db: AsyncSession, key: str) -> bool:
        """
        Delete a system setting from the database and invalidate the cache.
        """
        db_setting = await db.scalar(select(SystemSetting).where(SystemSetting.key == key))
        if db_setting:
            await db.delete(db_setting)


            redis: Redis = await get_redis()
            cache_key = f"{cls.CACHE_PREFIX}{key}"
            await redis.delete(cache_key)
            return True
        return False

    @classmethod
    async def list_settings(cls, db: AsyncSession, skip: int = 0, limit: int = 50) -> list[SystemSetting]:
        """
        List all settings in the database.
        """
        result = await db.execute(select(SystemSetting).offset(skip).limit(limit))
        return list(result.scalars().all())
