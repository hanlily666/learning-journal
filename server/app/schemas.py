"""Shapes the API returns. The browser's TypeScript types are generated from these."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

MessageStatus = Literal["streaming", "complete", "interrupted", "failed"]
Label = Literal["ai", "french", "messy"]
SegmentStatus = Literal["pending", "running", "done", "held", "skipped", "needs_attention"]
# The AI schema's list fields. Its two text fields, subject and summary, live on the segment.
ItemField = Literal["explore", "source", "gap", "concept"]


class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str | None
    source: str
    # The latest classified segment's label.
    subject: Label | None
    created_at: datetime
    updated_at: datetime


class MessageOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    role: Literal["user", "assistant"]
    content: list[dict[str, Any]]
    text: str
    created_at: datetime
    status: MessageStatus
    stop_reason: str | None
    error: str | None
    injected_context: bool


class SessionView(BaseModel):
    session: SessionOut
    messages: list[MessageOut]
    # Size of the conversation as of the latest reply (input + cached + output tokens).
    context_tokens: int
    # True once the conversation is big enough that a new session is the better choice.
    context_warning: bool
    # True when the latest reply failed or was interrupted and can be tried again.
    can_retry: bool


class SessionCreate(BaseModel):
    name: str | None = Field(default=None, max_length=200)


class SessionRename(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class MessageCreate(BaseModel):
    text: str = Field(min_length=1, max_length=200_000)


class SendResult(BaseModel):
    user_message: MessageOut
    assistant_message: MessageOut


class ItemOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    field: ItemField
    text: str
    source_message_ids: list[int]
    reason: str
    review_state: str
    agent_state: str


class SegmentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    session_id: int
    first_message_id: int
    last_message_id: int
    label: Label | None
    label_reason: str | None
    status: SegmentStatus
    subject: str | None
    summary: str | None
    error: str | None
    attempts: int
    created_at: datetime
    updated_at: datetime
    items: list[ItemOut] = []


class ExtractedSession(BaseModel):
    session: SessionOut
    # Oldest first.
    segments: list[SegmentOut]


class TriggerResult(BaseModel):
    # Segments queued by this call (sessions with unprocessed messages).
    queued: int
