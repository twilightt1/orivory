import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context
from app.config import settings
from app.database import Base

# Orivory second-brain models

config = context.config
config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# Postgres gets NO FTS, and this is where that is decided. The P2 lexical index
# (``app/retrieval/memory/lexical_index``) is an FTS5 virtual table: SQLite-only
# DDL, deliberately absent from the metadata above, with no migration in this
# directory. It is created by the SQLite ladder's own ``exec_driver_sql`` step
# (``app.database._upgrade_v3_to_v4``), and on Postgres the lexical leg reports
# itself unavailable — typed, never a silent "no matches" (ruling R3(p2)).
# autogenerate must therefore never learn about it: a virtual table in the
# metadata would make every Postgres migration try to CREATE TABLE it.


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(url=url, target_metadata=target_metadata,
                      literal_binds=True, dialect_opts={"paramstyle": "named"})
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
