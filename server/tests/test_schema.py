"""Rules the database enforces itself, whatever code talks to it."""

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError, IntegrityError


async def insert_session_and_message(engine, status: str) -> int:
    async with engine.begin() as conn:
        session_id = (await conn.execute(text("INSERT INTO session DEFAULT VALUES RETURNING id"))).scalar()
        return (await conn.execute(text(
            "INSERT INTO message (session_id, role, content, text, status) "
            "VALUES (:s, 'assistant', '[{\"type\": \"text\", \"text\": \"hi\"}]', 'hi', :st) RETURNING id"),
            {"s": session_id, "st": status})).scalar()


async def test_finished_messages_cannot_be_edited_or_deleted(db, engine):
    message_id = await insert_session_and_message(engine, "complete")
    for statement in (
        "UPDATE message SET text = 'edited' WHERE id = :id",
        "UPDATE message SET content = '[]' WHERE id = :id",
        "UPDATE message SET status = 'streaming' WHERE id = :id",
        "UPDATE message SET error = 'later note' WHERE id = :id",
        "UPDATE message SET injected_context = true WHERE id = :id",
        "DELETE FROM message WHERE id = :id",
    ):
        with pytest.raises(DBAPIError, match="append-only|cannot be edited"):
            async with engine.begin() as conn:
                await conn.execute(text(statement), {"id": message_id})


async def test_a_streaming_message_can_be_written_until_it_finishes(db, engine):
    message_id = await insert_session_and_message(engine, "streaming")
    async with engine.begin() as conn:
        await conn.execute(text("UPDATE message SET text = 'hi there' WHERE id = :id"), {"id": message_id})
        await conn.execute(text("UPDATE message SET status = 'complete' WHERE id = :id"), {"id": message_id})
    with pytest.raises(DBAPIError, match="cannot be edited"):
        async with engine.begin() as conn:
            await conn.execute(text("UPDATE message SET text = 'again' WHERE id = :id"), {"id": message_id})


async def test_external_ids_are_unique_per_source(db, engine):
    async with engine.begin() as conn:
        session_id = (await conn.execute(text("INSERT INTO session DEFAULT VALUES RETURNING id"))).scalar()
        insert = text("INSERT INTO message (session_id, role, status, source, external_id) "
                      "VALUES (:s, 'user', 'complete', :src, 'abc')")
        await conn.execute(insert, {"s": session_id, "src": "claude"})
        await conn.execute(insert, {"s": session_id, "src": "chatgpt"})
    with pytest.raises(IntegrityError):
        async with engine.begin() as conn:
            await conn.execute(insert, {"s": session_id, "src": "claude"})


async def test_row_level_security_hides_every_table_from_other_roles(db, engine):
    await insert_session_and_message(engine, "complete")
    async with engine.begin() as conn:
        await conn.execute(text(
            "DO $$ BEGIN IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'public_api') "
            "THEN CREATE ROLE public_api; END IF; END $$"))
        await conn.execute(text(
            "GRANT SELECT, INSERT ON session, message, api_request, alembic_version TO public_api"))
        await conn.execute(text("SET LOCAL ROLE public_api"))
        for table in ("session", "message", "api_request", "alembic_version"):
            assert (await conn.execute(text(f"SELECT count(*) FROM {table}"))).scalar() == 0
        with pytest.raises(DBAPIError, match="row-level security"):
            async with conn.begin_nested():
                await conn.execute(text("INSERT INTO session DEFAULT VALUES"))
