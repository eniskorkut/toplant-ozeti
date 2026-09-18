"""Async SQLAlchemy/SQLite setup shared by the API process and the worker."""

from __future__ import annotations

from collections.abc import AsyncIterator
from functools import lru_cache

from sqlalchemy import event
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.config import get_settings
from app.models import Base

SQLITE_PRAGMAS = (
    "PRAGMA journal_mode=WAL",
    "PRAGMA busy_timeout=5000",
    "PRAGMA foreign_keys=ON",
)


def configure_sqlite(engine: AsyncEngine) -> None:
    """Apply the shared SQLite hardening to every new connection.

    WAL keeps the API process and the worker from blocking each other, the busy
    timeout waits out short lock contention instead of failing, and foreign keys
    keep the meeting -> transcript/analysis relationships honest.
    """

    @event.listens_for(engine.sync_engine, "connect")
    def _set_pragmas(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        for pragma in SQLITE_PRAGMAS:
            cursor.execute(pragma)
        cursor.close()


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.engine: AsyncEngine = create_async_engine(url, future=True)
        configure_sqlite(self.engine)
        self.session_factory: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def init(self) -> None:
        async with self.engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)

    async def dispose(self) -> None:
        await self.engine.dispose()

    async def sessions(self) -> AsyncIterator[AsyncSession]:
        async with self.session_factory() as session:
            yield session


@lru_cache
def get_database() -> Database:
    return Database(get_settings().database_url)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding one session per request."""
    async for session in get_database().sessions():
        yield session
