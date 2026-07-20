"""Async PostgreSQL session lifecycle."""

from collections.abc import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine


def build_session_factory(database_url: str) -> async_sessionmaker[AsyncSession]:
    """Build a session factory without opening a connection eagerly."""

    engine = create_async_engine(database_url, pool_pre_ping=True)
    return async_sessionmaker(engine, expire_on_commit=False)


async def get_session() -> AsyncIterator[AsyncSession]:
    """FastAPI dependency yielding a transactional database session."""

    from backend.app.core.config.settings import get_settings

    factory = build_session_factory(get_settings().database_url)
    async with factory() as session:
        yield session

