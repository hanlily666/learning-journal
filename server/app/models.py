"""Database tables. The schema itself is created by the Alembic migrations."""

from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import BigInteger, Boolean, DateTime, ForeignKey, Integer, Numeric, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class ChatSession(Base):
    __tablename__ = "session"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    name: Mapped[str | None] = mapped_column(Text)
    source: Mapped[str] = mapped_column(Text, default="app")
    # The label of the session's latest classified segment: ai, french, or messy.
    subject: Mapped[str | None] = mapped_column(Text)
    # ID of the last message extraction has processed.
    checkpoint: Mapped[int | None] = mapped_column(BigInteger)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Message(Base):
    __tablename__ = "message"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"))
    role: Mapped[str] = mapped_column(Text)  # user | assistant
    # Content blocks exactly as the API returned them (thinking blocks included).
    content: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    # Plain-text copy of the text blocks, for display and search.
    text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True),
                                                 server_default=func.clock_timestamp())
    status: Mapped[str] = mapped_column(Text)  # streaming | complete | interrupted | failed
    stop_reason: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    injected_context: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(Text, default="app")
    external_id: Mapped[str | None] = mapped_column(Text)


class ApiRequest(Base):
    __tablename__ = "api_request"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    step: Mapped[str] = mapped_column(Text)  # chat | classification | extraction
    session_id: Mapped[int | None] = mapped_column(ForeignKey("session.id"))
    message_id: Mapped[int | None] = mapped_column(ForeignKey("message.id"))
    model_requested: Mapped[str] = mapped_column(Text)
    model_served: Mapped[str | None] = mapped_column(Text)
    input_tokens: Mapped[int | None] = mapped_column(BigInteger)
    output_tokens: Mapped[int | None] = mapped_column(BigInteger)
    cache_read_tokens: Mapped[int | None] = mapped_column(BigInteger)
    cache_write_tokens: Mapped[int | None] = mapped_column(BigInteger)
    stop_reason: Mapped[str | None] = mapped_column(Text)
    request_id: Mapped[str | None] = mapped_column(Text)
    cost_usd: Mapped[Decimal | None] = mapped_column(Numeric(12, 6))
    error: Mapped[str | None] = mapped_column(Text)
    usage: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class Segment(Base):
    """A run of new messages of one session, processed together."""

    __tablename__ = "segment"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    session_id: Mapped[int] = mapped_column(ForeignKey("session.id"))
    first_message_id: Mapped[int] = mapped_column(ForeignKey("message.id"))
    last_message_id: Mapped[int] = mapped_column(ForeignKey("message.id"))
    label: Mapped[str | None] = mapped_column(Text)  # ai | french | messy
    label_reason: Mapped[str | None] = mapped_column(Text)
    # pending | running | done | held | skipped | needs_attention
    status: Mapped[str] = mapped_column(Text, default="pending")
    subject: Mapped[str | None] = mapped_column(Text)
    summary: Mapped[str | None] = mapped_column(Text)
    error: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())


class Item(Base):
    """One extracted list-field value, with the messages it came from."""

    __tablename__ = "item"
    __mapper_args__ = {"eager_defaults": True}

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    segment_id: Mapped[int] = mapped_column(ForeignKey("segment.id"))
    field: Mapped[str] = mapped_column(Text)  # explore | source | gap | concept
    text: Mapped[str] = mapped_column(Text)
    source_message_ids: Mapped[list[int]] = mapped_column(ARRAY(BigInteger))
    reason: Mapped[str] = mapped_column(Text)
    review_state: Mapped[str] = mapped_column(Text, default="unreviewed")
    agent_state: Mapped[str] = mapped_column(Text, default="queued")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), server_default=func.now())
