from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from app.config import sqlalchemy_url


def make_engine(database_url: str) -> AsyncEngine:
    # Every timestamp is read and written in UTC, whatever the machine's time zone.
    return create_async_engine(sqlalchemy_url(database_url), pool_pre_ping=True,
                               connect_args={"options": "-c timezone=UTC"})


def make_sessionmaker(engine: AsyncEngine) -> async_sessionmaker:
    return async_sessionmaker(engine, expire_on_commit=False)
