"""Test fixtures: a real Postgres with the migrations applied.

Set TEST_DATABASE_URL to use an existing (empty, throwaway) database, for example the
Docker one. Otherwise an embedded Postgres from the `pgserver` package is started.
"""

import os
import tempfile
from pathlib import Path

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text

from app.db import make_engine, make_sessionmaker

SERVER_DIR = Path(__file__).resolve().parent.parent


def run_migrations(url: str, revision: str = "head") -> None:
    cfg = Config(str(SERVER_DIR / "alembic.ini"))
    cfg.set_main_option("script_location", str(SERVER_DIR / "migrations"))
    cfg.attributes["database_url"] = url
    cfg.attributes["configure_logging"] = False
    command.upgrade(cfg, revision)


@pytest.fixture(scope="session")
def database_url():
    url = os.environ.get("TEST_DATABASE_URL")
    server = None
    if not url:
        from pgserver import get_server  # pyright: ignore[reportPrivateImportUsage]

        server = get_server(tempfile.mkdtemp(prefix="lj-test-pg-"))
        url = server.get_uri()
    run_migrations(url)
    yield url
    if server is not None:
        server.cleanup()


@pytest.fixture(scope="session")
async def engine(database_url):
    engine = make_engine(database_url)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db(engine):
    async with engine.begin() as conn:
        await conn.execute(text("TRUNCATE item, segment, api_request, message, session RESTART IDENTITY CASCADE"))
    return make_sessionmaker(engine)
